from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, List, Tuple, Set, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

# Project helpers
from kv_cache_manager import KVCacheManager, KVCacheEntry, ChunkMetadata
from build_kv_cache import extract_texts, _tokenize_chunk, _prefill_get_past


# ------------------------------ Disk loader for precomputed KV ------------------------------

def _safe_load_tensor(path: str):
    return torch.load(path, map_location="cpu")

def _dir_has_kv(dir_path: str) -> bool:
    return all(os.path.isfile(os.path.join(dir_path, f)) for f in ["keys.pt", "values.pt", "valid_mask.pt"])

def _load_kv_cache_dir(
    cache_dir: str,
    kv: KVCacheManager,
    filter_prefix: str = "",
    load_to_gpu: bool = False,
    device: str = "cuda:0",
) -> int:
    if not cache_dir or not os.path.isdir(cache_dir):
        return 0

    loaded = 0
    for name in os.listdir(cache_dir):
        if filter_prefix and not name.startswith(filter_prefix):
            continue
        cdir = os.path.join(cache_dir, name)
        if not os.path.isdir(cdir) or not _dir_has_kv(cdir):
            continue

        meta_path = os.path.join(cdir, "metadata.json")
        keys = _safe_load_tensor(os.path.join(cdir, "keys.pt"))
        values = _safe_load_tensor(os.path.join(cdir, "values.pt"))
        valid_mask = _safe_load_tensor(os.path.join(cdir, "valid_mask.pt"))

        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)

        size_bytes = int(meta.get("size_bytes", 0))
        if size_bytes <= 0:
            size_bytes = keys.element_size() * keys.nelement() + values.element_size() * values.nelement()

        entry = KVCacheEntry(
            keys=keys,
            values=values,
            valid_mask=valid_mask,
            metadata=ChunkMetadata(
                chunk_id=meta.get("chunk_id", name),
                text=meta.get("text", ""),
                tokens=meta.get("tokens", []),
                relevance_score=float(meta.get("relevance_score", 0.0)),
                access_count=int(meta.get("access_count", 0)),
                last_access_time=float(meta.get("last_access_time", 0.0)),
                size_bytes=size_bytes,
                layer_count=int(meta.get("layer_count", 0)) or (keys.shape[0] if keys.ndim >= 3 else 0),
                is_on_gpu=bool(meta.get("is_on_gpu", False)),
            ),
        )

        if load_to_gpu:
            device_t = torch.device(device)
            entry.keys = entry.keys.to(device_t)
            entry.values = entry.values.to(device_t)
            entry.metadata.is_on_gpu = True
            kv.store_chunk(entry.metadata.chunk_id, entry, priority="gpu")
        else:
            entry.metadata.is_on_gpu = False
            kv.store_chunk(entry.metadata.chunk_id, entry, priority="cpu")

        loaded += 1
    return loaded


# ------------------------------ Lightweight bandit helpers ------------------------------

def _mean_token_embed(tokenizer, model, device, text: str, max_tokens: int) -> torch.Tensor:
    ids = tokenizer.encode(text or "", add_special_tokens=False)[:max_tokens]
    if not ids:
        H = model.get_input_embeddings().weight.shape[1]
        return torch.zeros(H, device=device)
    ids_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    emb = model.get_input_embeddings()(ids_t)
    return emb.mean(dim=1).squeeze(0)

def _compute_query_vec_local(tokenizer, model, device, question: str, max_q_tokens: int = 64) -> torch.Tensor:
    return _mean_token_embed(tokenizer, model, device, str(question or ""), max_q_tokens)

def _compute_chunk_centroids_local(tokenizer, model, device, texts: List[str], max_chunk_tokens: int = 32) -> torch.Tensor:
    H = model.get_input_embeddings().weight.shape[1]
    out = torch.zeros(len(texts), H, device=device)
    for i, t in enumerate(texts):
        out[i] = _mean_token_embed(tokenizer, model, device, t, max_chunk_tokens)
    return out

def _cosine_scores(qvec: torch.Tensor, mats: torch.Tensor) -> torch.Tensor:
    qn = torch.linalg.norm(qvec) + 1e-6
    mn = torch.linalg.norm(mats, dim=1) + 1e-6
    return (mats @ qvec) / (mn * qn)

def _bandit_init(dim: int, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    A_inv = torch.eye(dim) / max(1e-6, alpha)
    theta = torch.zeros(dim)
    return A_inv, theta

def _bandit_score_ucb(A_inv: torch.Tensor, theta: torch.Tensor, x: torch.Tensor, beta: float) -> float:
    a_hat = float(theta @ x)
    ucb = float(beta * torch.sqrt(torch.clamp(x @ (A_inv @ x), min=0.0)))
    return a_hat + ucb

def _bandit_update(A_inv: torch.Tensor, theta: torch.Tensor, x: torch.Tensor, y: float) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.view(-1)
    denom = float(1.0 + (x @ (A_inv @ x)))
    A_inv = A_inv - (A_inv @ torch.outer(x, x) @ A_inv) / max(denom, 1e-6)
    err = y - float(theta @ x)
    theta = theta + (A_inv @ x) * err
    return A_inv, theta

def _make_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def _lexical_overlap(query_text: str, doc_text: str, n: int = 2) -> float:
    q = (query_text or "").lower().split()
    d = (doc_text or "").lower().split()
    qn = _make_ngrams(q, n)
    dn = _make_ngrams(d, n)
    if not qn or not dn:
        return 0.0
    inter = len(qn & dn)
    union = len(qn | dn)
    return float(inter) / float(max(union, 1))

def _value_centroid(entry: KVCacheEntry) -> Optional[torch.Tensor]:
    try:
        v = entry.values
        # values shape: [L, S, H, D] or on GPU same
        if v is None or v.numel() == 0:
            return None
        # mean over layers, sequence and heads -> [D]
        while v.dim() > 1:
            v = v.mean(dim=0)
        return v  # [D]
    except Exception:
        return None


# ------------------------------ Transfers & materialization ------------------------------

def _transfer_gpu_to_cpu(kv: KVCacheManager, chunk_id: str) -> float:
    start = time.time()
    entry = kv.gpu_cache[chunk_id]
    entry.keys = entry.keys.cpu()
    entry.values = entry.values.cpu()
    entry.metadata.is_on_gpu = False
    kv.cpu_cache[chunk_id] = entry
    kv.gpu_memory_used -= entry.metadata.size_bytes
    kv.cpu_memory_used += entry.metadata.size_bytes
    del kv.gpu_cache[chunk_id]
    return time.time() - start

def _transfer_cpu_to_gpu(kv: KVCacheManager, chunk_id: str, device: str) -> float:
    start = time.time()
    entry = kv.cpu_cache[chunk_id]
    entry.keys = entry.keys.to(device)
    entry.values = entry.values.to(device)
    entry.metadata.is_on_gpu = True
    kv.gpu_cache[chunk_id] = entry
    kv.gpu_memory_used += entry.metadata.size_bytes
    kv.cpu_memory_used -= entry.metadata.size_bytes
    del kv.cpu_cache[chunk_id]
    return time.time() - start

@torch.inference_mode()
def _materialize_placeholder_on_lowprio(
    kv: KVCacheManager,
    tokenizer,
    model,
    device: torch.device,
    sample_id: str,
    idx: int,
    text: str,
    lowprio_stream: Optional[torch.cuda.Stream],
) -> Tuple[str, float]:
    cid = f"{sample_id}_chunk{idx}"
    start = time.time()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        with torch.cuda.stream(lowprio_stream) if lowprio_stream is not None else torch.cuda.stream(torch.cuda.current_stream()):
            inputs = _tokenize_chunk(tokenizer, text, device)
            outputs = _prefill_get_past(model, inputs)
        torch.cuda.synchronize(device)
    else:
        inputs = _tokenize_chunk(tokenizer, text, device)
        outputs = _prefill_get_past(model, inputs)

    entry = kv.create_kv_cache_entry(
        chunk_id=cid,
        text=text,
        tokens=tokenizer.encode(text, add_special_tokens=False),
        relevance_score=0.0,
        model_outputs=outputs,
    )
    kv.store_chunk(cid, entry, priority="cpu")  # CPU_READY
    entry.metadata.is_on_gpu = False
    entry.metadata.prefill_time_s = time.time() - start
    return cid, entry.metadata.prefill_time_s


# ------------------------------ Scoring & eviction ------------------------------

def _score_chunk(
    cid: str,
    kv: KVCacheManager,
    sim: float,
    now: float,
    bw_gbps: float,
    max_cost_normalizer_s: float = 0.2,
    weights: Dict[str, float] = None,
) -> float:
    if weights is None:
        weights = dict(w_sim=.45, w_temp=.20, w_rec=.15, w_aft=.10, w_pin=.20, w_cost=.35)

    e = kv.gpu_cache.get(cid) or kv.cpu_cache.get(cid)
    if e is None:
        return -1e9

    sim = max(0.0, min(1.0, float(sim)))
    temp = float(getattr(e.metadata, "temporal_locality", 0.0))
    rec = 1.0 if now - float(getattr(e.metadata, "last_access_time", 0.0)) < 0.5 else 0.0
    aft = min(1.0, float(getattr(e.metadata, "access_count", 0.0)) / 5.0)
    pin = 1.0 if bool(getattr(e.metadata, "pinned", False)) else 0.0

    if e.metadata.is_on_gpu:
        cost_s = 0.0
    else:
        has_kv = (e.keys is not None) and (e.keys.numel() > 0)
        size_gb = max(1e-9, e.metadata.size_bytes / (1024 ** 3))
        xfer = size_gb / max(1e-6, bw_gbps)
        mat = 0.0 if has_kv else float(getattr(e.metadata, "prefill_time_s", 0.05))
        cost_s = mat + xfer

    cost = min(1.0, cost_s / max_cost_normalizer_s)

    return (weights["w_sim"] * sim
            + weights["w_temp"] * temp
            + weights["w_rec"] * rec
            + weights["w_aft"] * aft
            + weights["w_pin"] * pin
            - weights["w_cost"] * cost)

def _pick_evictions(
    gpu_set: Set[str],
    target_set: Set[str],
    kv: KVCacheManager,
    scores: Dict[str, float],
    now: float,
    need_free: int,
) -> List[str]:
    if need_free <= 0:
        return list(gpu_set - target_set)

    candidates = list(gpu_set - target_set) or list(gpu_set)

    def evict_score(c):
        s = scores.get(c, -1e9)
        entry = (kv.gpu_cache.get(c) or kv.cpu_cache.get(c))
        pin = 1.0 if bool(getattr(entry.metadata, "pinned", False)) else 0.0
        cool_until = float(getattr(entry.metadata, "cooldown_until", 0.0))
        on_cooldown = 1.0 if now < cool_until else 0.0
        return s - 0.8 * pin - 0.6 * on_cooldown

    return sorted(candidates, key=evict_score)[:need_free]


# ------------------------------ Scheduler (pipeline API) ------------------------------

class TangoScheduler:
    """
    Wraps the original TANGO driver for direct use inside pipeline.py.
    """

    def run(
        self,
        pred_payload: Dict[str, Any],
        retr_samples: List[Dict[str, Any]],
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "auto",                 # "auto" | "bf16" | "fp16" | "fp32"
        max_gpu: int = 5,
        step_duration_ms: int = 50,
        safety_margin_ms: int = 30,
        max_samples: int = 1,
        load_cache_dir: str = "",
        cache_filter_prefix: str = "auto",
        load_initial_to_gpu: bool = False,
        out_path: Optional[str] = None,
        enable_progress: bool = False,
        provided_tokenizer: Optional[Any] = None,
        provided_model: Optional[Any] = None,
        steps: int = 16,
        promote_per_step: int = 2,
    ) -> Dict[str, Any]:
        """
        Execute scheduling and return {"trace": [...]}.
        """
        device_t = torch.device(device)
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "fp32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16 if (device_t.type == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability(device_t)[0] >= 8) else None

        tokenizer = provided_tokenizer if provided_tokenizer is not None else AutoTokenizer.from_pretrained(model_id)
        model = provided_model if provided_model is not None else AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
        model = model.to(device_t)
        model.eval()

        pred_samples = pred_payload.get("results", []) if isinstance(pred_payload, dict) else []
        pred_trace = pred_payload.get("trace", []) if isinstance(pred_payload, dict) else []

        if not isinstance(retr_samples, list) or not retr_samples:
            raise RuntimeError("Retrieval samples are empty or invalid")
        if max_samples > 0:
            pred_samples = pred_samples[:max_samples]

        # Group external predictions if present (else we will compute per-step from cosine sims)
        per_sample_steps: Dict[int, Dict[int, List[int]]] = {}
        if isinstance(pred_trace, list) and pred_trace:
            for row in pred_trace:
                si = int(row.get("sample_index", 0))
                st = int(row.get("step", 0))
                per_sample_steps.setdefault(si, {})[st] = list(map(int, row.get("promoted_indices", [])))

        # EMA bandwidth estimate (GB/s)
        bw_gbps = 5.0
        ema_alpha = 0.2

        out_trace: List[Dict[str, Any]] = []

        # low-priority stream (if CUDA)
        if device_t.type == "cuda" and torch.cuda.is_available():
            try:
                low_pri, high_pri = torch.cuda.get_stream_priority_range()
                lowprio_stream = torch.cuda.Stream(priority=high_pri)
            except AttributeError:
                lowprio_stream = torch.cuda.Stream()
        else:
            lowprio_stream = None

        samp_iter = enumerate(pred_samples)
        if enable_progress and tqdm is not None:
            samp_iter = enumerate(tqdm(pred_samples, desc="Samples", total=len(pred_samples)))

        for si, ps in samp_iter:
            sample = retr_samples[si]
            sample_id = str(sample.get("id", f"sample{si}"))
            texts_idx: List[Tuple[int, str]] = extract_texts(sample)

            kv = KVCacheManager(
                model_config={
                    "hidden_size": getattr(model.config, "hidden_size", 4096),
                    "num_layers": getattr(model.config, "num_hidden_layers", 32),
                    "num_attention_heads": getattr(model.config, "num_attention_heads", 32),
                    "head_dim": getattr(model.config, "hidden_size", 4096) // max(1, getattr(model.config, "num_attention_heads", 32)),
                    "vocab_size": getattr(model.config, "vocab_size", 32000),
                },
                gpu_memory_limit_gb=40.0,
                cpu_memory_limit_gb=100.0,
                max_gpu_chunks=max_gpu,
                max_cpu_chunks=10_000,
                device=str(device_t),
                require_kernels=True,
            )

            # Preload on-disk KV (CPU_READY)
            if load_cache_dir:
                prefix = f"{sample_id}_chunk" if cache_filter_prefix == "auto" else cache_filter_prefix
                _load_kv_cache_dir(
                    cache_dir=load_cache_dir,
                    kv=kv,
                    filter_prefix=prefix,
                    load_to_gpu=False,
                    device=str(device_t),
                )

            # Initial GPU placement
            initial_gpu_indices = list(map(int, ps.get("initial_gpu", [])))[:max_gpu]
            for idx in initial_gpu_indices:
                cid = f"{sample_id}_chunk{idx}"
                if cid in kv.cpu_cache:
                    if load_initial_to_gpu:
                        dt = _transfer_cpu_to_gpu(kv, cid, str(device_t))
                        entry = kv.gpu_cache[cid]
                        entry.metadata.pinned = (idx == initial_gpu_indices[0])
                        entry.metadata.cooldown_until = time.time() + 0.5
                        bytes_moved = max(1, entry.metadata.size_bytes)
                        if dt > 0:
                            inst = (bytes_moved / (1024 ** 3)) / dt
                            bw_gbps = ema_alpha * inst + (1 - ema_alpha) * bw_gbps
                    else:
                        # keep as CPU_READY; only mark metadata flags on existing placement
                        entry = kv.cpu_cache[cid]
                        entry.metadata.pinned = (idx == initial_gpu_indices[0])
                        entry.metadata.cooldown_until = time.time() + 0.5
                elif cid in kv.gpu_cache:
                    entry = kv.gpu_cache[cid]
                    entry.metadata.pinned = (idx == initial_gpu_indices[0])
                    entry.metadata.cooldown_until = time.time() + 0.5
                else:
                    text = texts_idx[idx][1] if 0 <= idx < len(texts_idx) else ""
                    inputs = _tokenize_chunk(tokenizer, text, device_t)
                    outputs = _prefill_get_past(model, inputs)
                    entry = kv.create_kv_cache_entry(
                        chunk_id=cid,
                        text=text,
                        tokens=tokenizer.encode(text, add_special_tokens=False),
                        relevance_score=1.0,
                        model_outputs=outputs,
                    )
                    kv.store_chunk(cid, entry, priority="gpu")
                    entry.metadata.pinned = (idx == initial_gpu_indices[0])
                    entry.metadata.cooldown_until = time.time() + 0.5

            # Initialize placeholders for remaining chunks
            seen_ids = set(kv.gpu_cache.keys()) | set(kv.cpu_cache.keys())
            for idx, text in texts_idx:
                cid = f"{sample_id}_chunk{idx}"
                if cid in seen_ids:
                    continue
                placeholder = kv.create_placeholder_entry(
                    chunk_id=cid,
                    text=text,
                    tokens=tokenizer.encode(text, add_special_tokens=False),
                    relevance_score=0.0,
                )
                kv.store_chunk(cid, placeholder, priority="cpu")

            print(f"[Scheduler] Bandit: preparing features for sample {si}")
            # ---------------- Bandit feature preparation (simple, lightweight) ----------------
            texts_only = [t for _, t in texts_idx]
            qvec = _compute_query_vec_local(tokenizer, model, device_t, sample.get("question", ""))
            centroids = _compute_chunk_centroids_local(tokenizer, model, device_t, texts_only)  # [N,H]
            align_cos = _cosine_scores(qvec, centroids)  # [N]

            # Retrieval scores if available
            idx_to_retr_score: Dict[int, float] = {}
            retr_scores = sample.get("retrieved_indices_score") or sample.get("retrieved_scores") or []
            retr_indices = sample.get("retrieved_indices") or []
            if isinstance(retr_indices, list) and isinstance(retr_scores, list):
                for i, s in zip(retr_indices, retr_scores):
                    try:
                        idx_to_retr_score[int(i)] = float(s)
                    except Exception:
                        continue

            # Token lengths
            tok_lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts_only]
            max_len = max(1, max(tok_lens) if tok_lens else 1)

            # Initialize bandit state
            feat_dim = 6  # [retrieval, log_len, residency, access_freq, align, recency]; keep small & simple
            A_inv, theta = _bandit_init(feat_dim, alpha=1.0)
            usefulness_ema: Dict[str, float] = {}
            ema_alpha_local = 0.3

            def _chunk_features(idx: int, now_ts: float) -> torch.Tensor:
                cid = f"{sample_id}_chunk{idx}"
                # Static
                retr = float(idx_to_retr_score.get(idx, 0.0))
                retr_norm = retr  # assume already roughly scaled
                log_len = float(torch.log(torch.tensor(tok_lens[idx] + 1.0))) / float(torch.log(torch.tensor(max_len + 1.0)))
                # Dynamic
                on_gpu = 1.0 if cid in kv.gpu_cache else 0.0
                entry = (kv.gpu_cache.get(cid) or kv.cpu_cache.get(cid))
                acc = float(getattr(entry.metadata, "access_count", 0.0)) if entry is not None else 0.0
                acc_norm = min(1.0, acc / 5.0)
                align = float(align_cos[idx]) if 0 <= idx < len(align_cos) else 0.0
                last_t = float(getattr(entry.metadata, "last_access_time", 0.0)) if entry is not None else 0.0
                recency = 1.0 if (now_ts - last_t) < 0.5 else 0.0
                # Local lexical match (bigram Jaccard)
                lex = _lexical_overlap(sample.get("question", ""), texts_only[idx], n=2)
                # Features: keep dimension modest; drift will be applied in scoring via 1 - align
                x = torch.tensor([retr_norm, log_len, on_gpu, acc_norm, align, recency], dtype=torch.float32)
                # We fold lex and drift via post-scoring penalties rather than expanding dim
                return x

            def _cost_seconds_for_idx(idx: int) -> float:
                cid = f"{sample_id}_chunk{idx}"
                e = (kv.gpu_cache.get(cid) or kv.cpu_cache.get(cid))
                if e is None:
                    return 0.1
                if e.metadata.is_on_gpu:
                    return 0.0
                has_kv = (e.keys is not None) and (e.keys.numel() > 0)
                size_gb = max(1e-9, e.metadata.size_bytes / (1024 ** 3))
                xfer = size_gb / max(1e-6, bw_gbps)
                mat = 0.0 if has_kv else float(getattr(e.metadata, "prefill_time_s", 0.05))
                return float(mat + xfer)

            # Per-step scheduling (treat each step as one token-time slot)
            external_steps = per_sample_steps.get(si, {})
            total_steps = max(steps, max(external_steps.keys()) + 1 if external_steps else steps)
            token_step = 0
            start_time = time.time()
            step_dur = step_duration_ms / 1000.0
            safety = safety_margin_ms / 1000.0

            step_iter = range(0, total_steps)
            if enable_progress and tqdm is not None:
                step_iter = tqdm(step_iter, desc=f"Steps (sample {si})", leave=False)

            for step in step_iter:
                now = start_time + step * step_dur
                print(f"[Scheduler] Step {step}: token_step={token_step}")
                # Decide candidates for this step: external predictions or top by bandit UCB value-per-cost
                preds = external_steps.get(step, [])
                if not preds:
                    now_ts = time.time()
                    current_gpu_idx = {int(cid.split("_chunk")[1]) for cid in kv.gpu_cache.keys() if "_chunk" in cid}
                    cpu_candidates = [i for i, _ in texts_idx if i not in current_gpu_idx]
                    scored: List[Tuple[int, float, float]] = []  # (idx, v_i, v_per_cost)
                    beta = 0.3
                    for i in cpu_candidates:
                        x = _chunk_features(i, now_ts)
                        v_base = _bandit_score_ucb(A_inv, theta, x, beta=beta)
                        # small lexical bonus, drift penalty via 1 - max(0, align)
                        lex = _lexical_overlap(sample.get("question", ""), texts_only[i], n=2)
                        align_i = float(align_cos[i]) if 0 <= i < len(align_cos) else 0.0
                        drift = 1.0 - max(0.0, align_i)
                        v_base = v_base + 0.1 * float(lex) - 0.2 * float(drift)
                        c_sec = _cost_seconds_for_idx(i)
                        v_ratio = float(v_base / max(c_sec, 1e-3))
                        scored.append((i, v_base, v_ratio))
                    scored.sort(key=lambda t: t[2], reverse=True)
                    preds = [i for i, _, _ in scored[:promote_per_step]]
                    print(f"[Scheduler] Predicted promotions (bandit): {preds}")

                sim_map: Dict[str, float] = {}
                expected_time: Dict[str, float] = {}
                for idx in preds:
                    cid = f"{sample_id}_chunk{idx}"
                    # Map bandit value to [0,1] via simple min-max over preds
                    sim_map[cid] = 1.0
                    expected_time[cid] = now + step_dur

                # score all chunks
                scores: Dict[str, float] = {}
                for cid in list(kv.gpu_cache.keys()) + list(kv.cpu_cache.keys()):
                    sim = sim_map.get(cid, 0.0)
                    scores[cid] = _score_chunk(cid, kv, sim, now, bw_gbps)

                all_ids = list(set(list(kv.gpu_cache.keys()) + list(kv.cpu_cache.keys())))

                # Select top-`max_gpu` target set
                target = set(sorted(all_ids, key=lambda c: scores.get(c, -1e9), reverse=True)[:max_gpu])

                gpu_set = set(kv.gpu_cache.keys())

                # Evict if needed
                need_free = max(0, (len(gpu_set | target) - max_gpu))
                evict = _pick_evictions(gpu_set, target, kv, scores, now, need_free)
                evicted_do: List[str] = []
                for cid in evict:
                    if cid in kv.gpu_cache:
                        dt = _transfer_gpu_to_cpu(kv, cid)
                        entry = kv.cpu_cache[cid]
                        bytes_moved = max(1, entry.metadata.size_bytes)
                        if dt > 0:
                            inst = (bytes_moved / (1024 ** 3)) / dt
                            bw_gbps = 0.2 * inst + 0.8 * bw_gbps
                        evicted_do.append(cid)

                # Promotions required
                gpu_set = set(kv.gpu_cache.keys())
                need_promote = list(target - gpu_set)

                to_promote: List[str] = []
                to_materialize: List[int] = []

                for cid in need_promote:
                    entry = kv.cpu_cache.get(cid)
                    if entry is None:
                        continue
                    has_kv = (entry.keys is not None) and (entry.keys.numel() > 0)
                    if has_kv:
                        to_promote.append(cid)
                    else:
                        try:
                            idx = int(cid.split("_chunk")[1])
                        except Exception:
                            idx = -1
                        if idx >= 0:
                            to_materialize.append(idx)

                # Materialize placeholders on low-priority stream if time allows
                mat_done: List[str] = []
                for idx in to_materialize:
                    cid = f"{sample_id}_chunk{idx}"
                    ddl = expected_time.get(cid, now + 2 * step_dur)
                    entry = kv.cpu_cache[cid]
                    mat_est = float(getattr(entry.metadata, "prefill_time_s", 0.05))
                    if (ddl - now - mat_est - safety) > 0:
                        text = texts_idx[idx][1]
                        _, mat_time = _materialize_placeholder_on_lowprio(
                            kv, tokenizer, model, device_t, sample_id, idx, text, lowprio_stream
                        )
                        entry2 = kv.cpu_cache[cid]
                        bytes_moved = max(1, entry2.metadata.size_bytes)
                        if mat_time > 0:
                            inst = (bytes_moved / (1024 ** 3)) / mat_time
                            bw_gbps = 0.2 * inst + 0.8 * bw_gbps
                        mat_done.append(cid)

                # Deadline-aware promotion decisions
                promote_now: List[str] = []
                for cid in list(set(to_promote) | set(mat_done)):
                    if cid not in kv.cpu_cache:
                        continue
                    ddl = expected_time.get(cid, now + 2 * step_dur)
                    entry = kv.cpu_cache[cid]
                    size_gb = max(1e-9, entry.metadata.size_bytes / (1024 ** 3))
                    xfer_est = size_gb / max(1e-6, bw_gbps)
                    if (ddl - now - xfer_est - safety) > 0:
                        promote_now.append(cid)

                # Ensure space, then promote
                gpu_free_slots = max_gpu - len(kv.gpu_cache)
                if len(promote_now) > gpu_free_slots:
                    extra = len(promote_now) - gpu_free_slots
                    current_gpu = set(kv.gpu_cache.keys())
                    victims = _pick_evictions(current_gpu, target, kv, scores, now, extra)
                    for cid in victims:
                        if cid in kv.gpu_cache:
                            dt = _transfer_gpu_to_cpu(kv, cid)
                            entry = kv.cpu_cache[cid]
                            bytes_moved = max(1, entry.metadata.size_bytes)
                            if dt > 0:
                                inst = (bytes_moved / (1024 ** 3)) / dt
                                bw_gbps = 0.2 * inst + 0.8 * bw_gbps

                promoted: List[str] = []
                for cid in promote_now:
                    if cid in kv.cpu_cache and len(kv.gpu_cache) < max_gpu:
                        dt = _transfer_cpu_to_gpu(kv, cid, str(device_t))
                        entry = kv.gpu_cache[cid]
                        entry.metadata.cooldown_until = time.time() + 0.5
                        bytes_moved = max(1, entry.metadata.size_bytes)
                        if dt > 0:
                            inst = (bytes_moved / (1024 ** 3)) / dt
                            bw_gbps = 0.2 * inst + 0.8 * bw_gbps
                        promoted.append(cid)
                if promoted:
                    print(f"[Scheduler] Promoted to GPU: {promoted}")

                out_trace.append({
                    "sample_index": si,
                    "step": step,
                    "predicted_indices": [f"{sample_id}_chunk{i}" for i in preds],
                    "materialized": mat_done,
                    "promoted": promoted,
                    "evicted": evicted_do,
                    "gpu_chunks": len(kv.gpu_cache),
                    "cpu_chunks": len(kv.cpu_cache),
                    "bw_gbps_ema": bw_gbps,
                    "token_step": token_step,
                })
                token_step += 1

                # Online bandit update from outcomes for promoted chunks
                now_ts2 = time.time()
                for i in preds:
                    cid = f"{sample_id}_chunk{i}"
                    x = _chunk_features(i, now_ts2)
                    # Proxy reward: 1.0 if promoted_now, else 0.0
                    y = 1.0 if any(cid == p for p in promoted) else 0.0
                    A_inv, theta = _bandit_update(A_inv, theta, x, y)
                    # Update usefulness EMA
                    prev = usefulness_ema.get(cid, 0.0)
                    usefulness_ema[cid] = ema_alpha_local * y + (1 - ema_alpha_local) * prev

        payload = {"trace": out_trace}

        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)

        return payload
