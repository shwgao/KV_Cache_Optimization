from __future__ import annotations
import os
import json
import time
import threading
from typing import Any, Dict, List, Tuple, Set, Optional
from queue import Queue, Empty
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

# Project helpers
from kv_cache_manager import KVCacheManager, KVCacheEntry, ChunkMetadata
from build_kv_cache import extract_texts, _tokenize_chunk, _prefill_get_past


# ------------------------------ Staging System for Per-Step Decoding ------------------------------

@dataclass
class PredictionEntry:
    """Single chunk prediction with metadata"""
    chunk_id: str
    predicted_for_token: int
    priority: float
    timestamp: float

class ChunkStaging:
    """Manages chunk predictions between scheduler runs"""
    
    def __init__(self):
        self.predictions: Dict[str, PredictionEntry] = {}
        self.preparing: Set[str] = set()
        self.cpu_ready: Set[str] = set()
        self.lock = threading.Lock()
        
    def add_prediction(self, chunk_id: str, predicted_for_token: int, priority: float):
        """Add a new chunk prediction"""
        with self.lock:
            if chunk_id in self.cpu_ready:
                return
            
            entry = PredictionEntry(
                chunk_id=chunk_id,
                predicted_for_token=predicted_for_token,
                priority=priority,
                timestamp=time.time()
            )
            
            if chunk_id not in self.predictions or entry.priority > self.predictions[chunk_id].priority:
                self.predictions[chunk_id] = entry
    
    def get_next_to_materialize(self) -> Optional[str]:
        """Get next chunk for background materialization"""
        with self.lock:
            candidates = []
            for chunk_id, entry in self.predictions.items():
                if chunk_id not in self.preparing and chunk_id not in self.cpu_ready:
                    candidates.append((chunk_id, entry.priority, entry.predicted_for_token))
            
            if not candidates:
                return None
            
            candidates.sort(key=lambda x: (-x[1], x[2]))
            return candidates[0][0]
    
    def mark_preparing(self, chunk_id: str):
        with self.lock:
            self.preparing.add(chunk_id)
    
    def mark_ready(self, chunk_id: str):
        with self.lock:
            self.preparing.discard(chunk_id)
            self.cpu_ready.add(chunk_id)
    
    def get_ready_chunks(self) -> List[Tuple[str, PredictionEntry]]:
        """Get chunks ready for GPU promotion"""
        with self.lock:
            ready_with_meta = []
            for chunk_id in self.cpu_ready:
                if chunk_id in self.predictions:
                    ready_with_meta.append((chunk_id, self.predictions[chunk_id]))
            return ready_with_meta
    
    def cleanup_old_predictions(self, current_token: int, max_age: int = 10):
        """Remove old predictions"""
        with self.lock:
            to_remove = []
            for chunk_id, entry in self.predictions.items():
                if current_token - entry.predicted_for_token > max_age:
                    to_remove.append(chunk_id)
            
            for chunk_id in to_remove:
                del self.predictions[chunk_id]
                self.preparing.discard(chunk_id)
                self.cpu_ready.discard(chunk_id)

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
            sample_id = str(sample.get("id", f"{si}"))
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

    def run_per_step_decode(
        self,
        retr_samples: List[Dict[str, Any]],
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "auto",
        max_gpu: int = 5,
        max_samples: int = 1,
        max_new_tokens: int = 10,
        scheduler_interval: int = 5,  # Run heavy scheduler every N tokens
        provided_tokenizer: Optional[Any] = None,
        provided_model: Optional[Any] = None,
        promote_per_step: int = 2,
        initial_gpu_indices: List[int] = None,  # Initial GPU chunk placement
    ) -> Dict[str, Any]:
        """
        New per-step decoding with lightweight prediction and periodic heavy scheduling.
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

        tokenizer = provided_tokenizer
        model = provided_model
        model = model.to(device_t)
        model.eval()

        if max_samples > 0:
            retr_samples = retr_samples[:max_samples]


        results = []
        
        for si, sample in enumerate(retr_samples):
            sample_id = f"{si}"
            texts_idx: List[Tuple[int, str]] = extract_texts(sample)
            
            # Initialize KV cache manager
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
            
            # Initialize staging
            staging = ChunkStaging()
            
            # Background materialization queue
            materialization_queue = Queue()
            
            
            print(f"[PerStepDecode] Sample {si}: Using initial GPU chunks {initial_gpu_indices}")
            
            # Create initial GPU chunks with CORRECT RoPE positions
            cumulative_position = 0
            for idx in initial_gpu_indices:
                if 0 <= idx < len(texts_idx):
                    cid = f"{sample_id}_chunk{idx}"
                    text = texts_idx[idx][1]
                    
                    # Apply correct RoPE positions
                    inputs = _tokenize_chunk(tokenizer, text, device_t)
                    seq_len = inputs["input_ids"].shape[1]

                    # Create position_ids starting from cumulative position
                    position_ids = torch.arange(
                        cumulative_position, 
                        cumulative_position + seq_len, 
                        device=device_t
                    ).unsqueeze(0)
                    inputs["position_ids"] = position_ids
                    
                    print(f"[RoPE] Chunk {idx}: positions [{cumulative_position}:{cumulative_position + seq_len}]")
                    
                    outputs = _prefill_get_past(model, inputs)
                    entry = kv.create_kv_cache_entry(
                        chunk_id=cid,
                        text=text,
                        tokens=tokenizer.encode(text, add_special_tokens=False),
                        relevance_score=1.0,
                        model_outputs=outputs,
                    )

                    kv.store_chunk(cid, entry, priority="gpu")
                    cumulative_position += seq_len
            
            # Create placeholders for remaining chunks
            for idx, text in texts_idx:
                cid = f"{sample_id}_chunk{idx}"
                if cid not in kv.gpu_cache:
                    placeholder = kv.create_placeholder_entry(
                        chunk_id=cid,
                        text=text,
                        tokens=tokenizer.encode(text, add_special_tokens=False),
                        relevance_score=0.0,
                    )
                    kv.store_chunk(cid, placeholder, priority="cpu")
            
            # Prepare bandit features
            texts_only = [t for _, t in texts_idx]
            qvec = _compute_query_vec_local(tokenizer, model, device_t, sample.get("question", ""))
            centroids = _compute_chunk_centroids_local(tokenizer, model, device_t, texts_only)
            align_cos = _cosine_scores(qvec, centroids)
            
            # Token length features
            tok_lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts_only]
            max_len = max(1, max(tok_lens) if tok_lens else 1)
            
            # Initialize bandit
            feat_dim = 6
            A_inv, theta = _bandit_init(feat_dim, alpha=1.0)
            
            # Per-step decoding
            result = self._decode_per_step(
                sample=sample,
                sample_id=sample_id,
                texts_idx=texts_idx,
                kv=kv,
                staging=staging,
                materialization_queue=materialization_queue,
                model=model,
                tokenizer=tokenizer,
                device_t=device_t,
                max_new_tokens=max_new_tokens,
                scheduler_interval=scheduler_interval,
                max_gpu=max_gpu,
                promote_per_step=promote_per_step,
                # Bandit state
                A_inv=A_inv,
                theta=theta,
                align_cos=align_cos,
                tok_lens=tok_lens,
                max_len=max_len
            )
            
            results.append({
                "sample_index": si,
                "sample_id": sample_id,
                **result
            })
        
        return {"results": results}
    
    def _decode_per_step(
        self,
        sample: Dict[str, Any],
        sample_id: str,
        texts_idx: List[Tuple[int, str]],
        kv: KVCacheManager,
        staging: ChunkStaging,
        materialization_queue: Queue,
        model,
        tokenizer,
        device_t: torch.device,
        max_new_tokens: int,
        scheduler_interval: int,
        max_gpu: int,
        promote_per_step: int,
        A_inv: torch.Tensor,
        theta: torch.Tensor,
        align_cos: torch.Tensor,
        tok_lens: List[int],
        max_len: int
    ) -> Dict[str, Any]:
        """Core per-step decoding logic"""
        
        # Prepare input: include context via stored KV if available; otherwise prepend text context
        question_text = sample.get("question", "")
        # Use only the question (no "Answer:" prefix) to avoid injecting stylistic tokens
        suffix = f"Question: {question_text}"
        
        # Extract and display the question
        question = suffix.split('Question: ')[1].strip() if 'Question: ' in suffix else suffix
        print(f"[Question] {question}")
        
        # Attempt to build past_key_values from currently available GPU chunks for first token
        current_gpu_chunks = set(kv.gpu_cache.keys())
        built_past = self._build_past_key_values_from_kv(kv, current_gpu_chunks)
        cache_for_model = None
        cached_len_for_pos = 0
        if built_past is not None:
            try:
                cache_for_model = self._convert_to_cache_format(built_past, model)
                if isinstance(cache_for_model, tuple) and len(cache_for_model) > 0:
                    # Infer cached length from first layer's K: shape [B, H, S, D]
                    cached_len_for_pos = cache_for_model[0][0].shape[2]
            except Exception as e:
                print(f"[FirstToken] Failed to convert built KV to model cache: {e}")
                cache_for_model = None
                cached_len_for_pos = 0
        else:
            print("No cache for model")
        
        # Always include the question tokens as input_ids for the first token decoding
        enc = tokenizer([suffix], return_tensors="pt", padding=True, truncation=False)
        input_ids = enc["input_ids"].to(device_t)
        
        generated_tokens = []
        trace = []
        
        start_time = time.time()
        first_token_time = None
        bw_gbps = 5.0  # EMA bandwidth estimate
        
        # Background materialization worker
        def background_worker():
            while True:
                try:
                    chunk_id = materialization_queue.get(timeout=0.1)
                    if chunk_id is None:  # Shutdown signal
                        break
                    self._materialize_chunk_background(chunk_id, kv, staging, tokenizer, model, device_t, sample_id, texts_idx)
                    materialization_queue.task_done()
                except Empty:
                    continue
                except Exception as e:
                    print(f"[Background] Error materializing: {e}")
        
        worker_thread = threading.Thread(target=background_worker, daemon=True)
        worker_thread.start()
        
        try:
            for token_step in range(max_new_tokens):
                
                # 1. Generate one token using CacheBlend's paged attention
                current_gpu_chunks = set(kv.gpu_cache.keys())
                
                with torch.no_grad():
                    if token_step == 0:
                        # For first token, use model forward with stored KV if available
                        model_inputs = {
                            "input_ids": input_ids,
                            "attention_mask": torch.ones_like(input_ids),
                            "use_cache": True,
                            "return_dict": True
                        }
                        if cache_for_model is not None:
                            model_inputs["past_key_values"] = cache_for_model
                            # Ensure correct positions after cached context
                            if cached_len_for_pos > 0:
                                position_ids = torch.arange(cached_len_for_pos, cached_len_for_pos + input_ids.shape[1], device=device_t).unsqueeze(0)
                                model_inputs["position_ids"] = position_ids
                                model_inputs["cache_position"] = torch.arange(cached_len_for_pos, cached_len_for_pos + input_ids.shape[1], device=device_t)
                        outputs = model(**model_inputs)
                        next_token_logits = outputs.logits[0, -1, :]
                    else:
                        # For subsequent tokens, use CacheBlend's efficient paged attention
                        last_token = torch.tensor([[generated_tokens[-1]]], device=device_t)
                        
                        # Use CacheBlend's built-in paged attention with proper KV cache format
                        next_token_logits = self._use_cacheblend_paged_attention(
                            model, tokenizer, last_token, kv, current_gpu_chunks, device_t
                        )
                        print(f"[CacheBlend] Successfully used paged attention with {len(current_gpu_chunks)} GPU chunks")
                    
                    # Greedy decoding for all steps (deterministic)
                    next_token = torch.argmax(next_token_logits).item()
                    
                    generated_tokens.append(next_token)
                    
                    # Show progress every few tokens
                    decoded_token = tokenizer.decode([next_token], skip_special_tokens=True)
                    print(f"[Token {token_step}] '{decoded_token}'", end=" " if token_step % 9 != 0 else "\n")


                token_end_time = time.time()
                if first_token_time is None:
                    first_token_time = token_end_time
                    
                # 2. Lightweight prediction using bandit (existing method)
                predictions = self._predict_chunks_bandit(
                    sample_id=sample_id,
                    texts_idx=texts_idx,
                    kv=kv,
                    A_inv=A_inv,
                    theta=theta,
                    align_cos=align_cos,
                    tok_lens=tok_lens,
                    max_len=max_len,
                    sample=sample,
                    promote_per_step=promote_per_step
                )
                
                # Show predictions only when they exist
                if predictions and token_step % 5 == 0:  # Every 5th token
                    pred_chunks = [p[0] if isinstance(p, tuple) else p for p in predictions]
                    print(f"[Predict T{token_step}] {pred_chunks}")
                
                # 3. Stage predictions
                for chunk_idx, priority in predictions:
                    chunk_id = f"{sample_id}_chunk{chunk_idx}"
                    staging.add_prediction(chunk_id, token_step + 1, priority)
                
                # 4. Request background materialization AND predictive transfer
                next_to_materialize = staging.get_next_to_materialize()
                if next_to_materialize:
                    materialization_queue.put(next_to_materialize)
                
                # 4b. Predictive GPU transfer on low-priority stream
                self._predictive_gpu_transfer(kv, staging, predictions, token_step, device_t, max_gpu)
                
                # 5. Heavy scheduler every N tokens
                if token_step % scheduler_interval == 0:
                    gpu_before = len(kv.gpu_cache)
                    ready_before = len(staging.get_ready_chunks())
                    self._run_heavy_scheduler_step(kv, staging, max_gpu, token_step, bw_gbps)
                    gpu_after = len(kv.gpu_cache)
                    ready_after = len(staging.get_ready_chunks())
                    print(f"[HeavyScheduler T{token_step}] GPU: {gpu_before}→{gpu_after}, Ready: {ready_before}→{ready_after}")
                
                # 6. Cleanup old predictions
                staging.cleanup_old_predictions(token_step)
                
                trace.append({
                    "token_step": token_step,
                    "token": next_token,
                    "predictions": predictions,
                    "gpu_chunks": len(kv.gpu_cache),
                    "cpu_chunks": len(kv.cpu_cache),
                    "staging_stats": {
                        "predictions": len(staging.predictions),
                        "preparing": len(staging.preparing),
                        "ready": len(staging.cpu_ready)
                    }
                })
                
                # 7. Check for EOS or repetitive patterns
                if next_token == tokenizer.eos_token_id:
                    print(f"\n[Stop] EOS at token {token_step}")
                    break
                
                # Check for repetitive patterns
                if len(generated_tokens) >= 3:
                    if generated_tokens[-1] == generated_tokens[-2] == generated_tokens[-3]:
                        print(f"\n[Stop] Repetitive pattern at token {token_step}")
                        break
                
                # Limit very long sequences
                if token_step >= max_new_tokens - 1:
                    print(f"\n[Stop] Max tokens ({max_new_tokens}) reached")
                    break
        
        finally:
            # Shutdown background worker
            materialization_queue.put(None)
            worker_thread.join(timeout=1.0)
        
        end_time = time.time()
        total_time = end_time - start_time
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Display final answer clearly
        print(f"\n[Answer] {generated_text.strip()}")
        print(f"[Summary] Generated {len(generated_tokens)} tokens in {total_time:.2f}s")
        
        # Calculate TPOT (Time Per Output Token)
        tpot = total_time / len(generated_tokens) if len(generated_tokens) > 0 else 0.0
        
        return {
            "answer": generated_text.strip(),
            "ttft": (first_token_time - start_time) if first_token_time else 0.0,
            "e2e_latency": total_time,
            "throughput": len(generated_tokens) / total_time if total_time > 0 else 0,
            "tpot": tpot,
            "decoded_tokens": len(generated_tokens),
            "trace": trace
        }
    
    def _predict_chunks_bandit(self, sample_id: str, texts_idx: List[Tuple[int, str]], kv: KVCacheManager, 
                              A_inv: torch.Tensor, theta: torch.Tensor, align_cos: torch.Tensor,
                              tok_lens: List[int], max_len: int, sample: Dict[str, Any], promote_per_step: int) -> List[Tuple[int, float]]:
        """Use existing bandit method for lightweight prediction"""
        
        def _chunk_features(idx: int, now_ts: float) -> torch.Tensor:
            cid = f"{sample_id}_chunk{idx}"
            # Use existing feature computation logic
            retr_score = 0.0  # Could get from sample if available
            log_len = float(torch.log(torch.tensor(tok_lens[idx] + 1.0))) / float(torch.log(torch.tensor(max_len + 1.0)))
            on_gpu = 1.0 if cid in kv.gpu_cache else 0.0
            entry = (kv.gpu_cache.get(cid) or kv.cpu_cache.get(cid))
            acc = float(getattr(entry.metadata, "access_count", 0.0)) if entry is not None else 0.0
            acc_norm = min(1.0, acc / 5.0)
            align = float(align_cos[idx]) if 0 <= idx < len(align_cos) else 0.0
            last_t = float(getattr(entry.metadata, "last_access_time", 0.0)) if entry is not None else 0.0
            recency = 1.0 if (now_ts - last_t) < 0.5 else 0.0
            
            return torch.tensor([retr_score, log_len, on_gpu, acc_norm, align, recency], dtype=torch.float32)
        
        now_ts = time.time()
        current_gpu_idx = {int(cid.split("_chunk")[1]) for cid in kv.gpu_cache.keys() if "_chunk" in cid}
        cpu_candidates = [i for i, _ in texts_idx if i not in current_gpu_idx]
        
        scored = []
        beta = 0.3
        
        for i in cpu_candidates:
            x = _chunk_features(i, now_ts)
            v_base = _bandit_score_ucb(A_inv, theta, x, beta=beta)
            scored.append((i, v_base))
        
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:promote_per_step]
    
    def _materialize_chunk_background(self, chunk_id: str, kv: KVCacheManager, staging: ChunkStaging, 
                                     tokenizer, model, device_t: torch.device, sample_id: str, texts_idx: List[Tuple[int, str]]):
        """Background chunk materialization"""
        staging.mark_preparing(chunk_id)
        
        try:
            if "_chunk" not in chunk_id:
                return
            
            idx = int(chunk_id.split("_chunk")[1])
            texts_dict = {i: text for i, text in texts_idx}
            
            if idx not in texts_dict:
                return
            
            text = texts_dict[idx]
            
            # Check if already materialized
            if chunk_id in kv.cpu_cache:
                entry = kv.cpu_cache[chunk_id]
                if entry.keys is not None and entry.keys.numel() > 0:
                    staging.mark_ready(chunk_id)
                    return
            
            # Materialize with correct RoPE positions
            with torch.inference_mode():
                inputs = _tokenize_chunk(tokenizer, text, device_t)
                seq_len = inputs["input_ids"].shape[1]
                
                # Calculate correct position for this chunk
                # Extract chunk index from chunk_id (e.g., "sample0_chunk3" -> 3)
                chunk_idx = int(chunk_id.split('_chunk')[-1])
                
                # Estimate position based on average chunk length and chunk index
                # This is approximate - ideally we'd track exact positions
                avg_chunk_length = 500  # Rough estimate, adjust based on your data
                estimated_start_position = chunk_idx * avg_chunk_length
                
                position_ids = torch.arange(
                    estimated_start_position, 
                    estimated_start_position + seq_len, 
                    device=device_t
                ).unsqueeze(0)
                inputs["position_ids"] = position_ids
                
                print(f"[RoPE] Background chunk {chunk_idx}: estimated positions [{estimated_start_position}:{estimated_start_position + seq_len}]")
                
                outputs = _prefill_get_past(model, inputs)
                
                entry = kv.create_kv_cache_entry(
                    chunk_id=chunk_id,
                    text=text,
                    tokens=tokenizer.encode(text, add_special_tokens=False),
                    relevance_score=0.0,
                    model_outputs=outputs,
                )
                
                kv.store_chunk(chunk_id, entry, priority="cpu")
                entry.metadata.is_on_gpu = False
                staging.mark_ready(chunk_id)
        
        except Exception as e:
            print(f"[Background] Failed to materialize {chunk_id}: {e}")
    
    def _run_heavy_scheduler_step(self, kv: KVCacheManager, staging: ChunkStaging, max_gpu: int, token_step: int, bw_gbps: float):
        """Run heavy scheduling decisions periodically"""
        ready_chunks = staging.get_ready_chunks()
        
        if not ready_chunks:
            print(f"[HeavyScheduler] No chunks ready for promotion")
            return
        
        # Sort ready chunks by priority (highest first)
        ready_chunks.sort(key=lambda x: x[1].priority, reverse=True)
        
        current_gpu_count = len(kv.gpu_cache)
        current_gpu_chunks = set(kv.gpu_cache.keys())
        
        print(f"[HeavyScheduler] {len(ready_chunks)} chunks ready, current GPU: {current_gpu_count}/{max_gpu}")
        
        # Determine which chunks should be on GPU (top priority ready + current)
        target_gpu_chunks = set()
        
        # Add current GPU chunks to consideration with default priority
        for chunk_id in current_gpu_chunks:
            if chunk_id in staging.predictions:
                priority = staging.predictions[chunk_id].priority
            else:
                # Give current GPU chunks a moderate priority to avoid immediate eviction
                priority = 0.5
            target_gpu_chunks.add((chunk_id, priority))
        
        # Add ready chunks to consideration
        for chunk_id, pred_entry in ready_chunks:
            target_gpu_chunks.add((chunk_id, pred_entry.priority))
        
        # Select top max_gpu chunks by priority
        sorted_targets = sorted(target_gpu_chunks, key=lambda x: x[1], reverse=True)[:max_gpu]
        target_chunk_ids = {chunk_id for chunk_id, _ in sorted_targets}
        
        print(f"[HeavyScheduler] Target GPU chunks: {sorted([cid.split('_chunk')[-1] for cid in target_chunk_ids])}")
        
        # Evict chunks not in target set
        evicted_chunks = []
        for chunk_id in list(current_gpu_chunks):
            if chunk_id not in target_chunk_ids:
                try:
                    _transfer_gpu_to_cpu(kv, chunk_id)
                    evicted_chunks.append(chunk_id)
                except Exception as e:
                    print(f"[HeavyScheduler] Failed to evict {chunk_id}: {e}")
        
        # Promote chunks in target set that aren't on GPU yet
        promoted_chunks = []
        for chunk_id in target_chunk_ids:
            if chunk_id not in kv.gpu_cache and chunk_id in kv.cpu_cache:
                if len(kv.gpu_cache) < max_gpu:
                    try:
                        _transfer_cpu_to_gpu(kv, chunk_id, str(kv.device))
                        promoted_chunks.append(chunk_id)
                    except Exception as e:
                        print(f"[HeavyScheduler] Failed to promote {chunk_id}: {e}")
        
        # Summary is now handled by the caller
    
    def _predictive_gpu_transfer(self, kv: KVCacheManager, staging: ChunkStaging, predictions: List[Tuple[int, float]], 
                                token_step: int, device_t: torch.device, max_gpu: int):
        """Predictively transfer high-priority chunks to GPU using low-priority CUDA stream"""
        if not predictions or device_t.type != "cuda":
            return
        
        # Get low-priority stream for background transfers
        if not hasattr(self, '_low_priority_stream'):
            try:
                low_pri, high_pri = torch.cuda.get_stream_priority_range()
                self._low_priority_stream = torch.cuda.Stream(priority=low_pri, device=device_t)
            except (AttributeError, RuntimeError):
                self._low_priority_stream = torch.cuda.Stream(device=device_t)
        
        # Check if we have GPU slots available for predictive transfers
        current_gpu_count = len(kv.gpu_cache)
        available_slots = max_gpu - current_gpu_count
        
        if available_slots <= 0:
            # GPU is full, but we can still prepare for future evictions
            print(f"[PredictiveTransfer] GPU full ({current_gpu_count}/{max_gpu}), preparing for future swaps")
            return
        
        # Get ready chunks sorted by priority
        ready_chunks = staging.get_ready_chunks()
        if not ready_chunks:
            return
        
        ready_chunks.sort(key=lambda x: x[1].priority, reverse=True)
        
        # Predictively transfer top chunks that are predicted for near-future tokens
        transferred = 0
        for chunk_id, pred_entry in ready_chunks[:available_slots]:
            # Only transfer chunks predicted for very soon (next 1-2 tokens)
            if pred_entry.predicted_for_token <= token_step + 2:
                if chunk_id in kv.cpu_cache and chunk_id not in kv.gpu_cache:
                    try:
                        # Transfer on low-priority stream
                        with torch.cuda.stream(self._low_priority_stream):
                            self._transfer_cpu_to_gpu_async(kv, chunk_id, device_t)
                            transferred += 1
                            print(f"[PredictiveTransfer] Started async transfer of {chunk_id} (priority: {pred_entry.priority:.3f}, for token: {pred_entry.predicted_for_token})")
                    except Exception as e:
                        print(f"[PredictiveTransfer] Failed to transfer {chunk_id}: {e}")
        
        if transferred > 0:
            # Synchronize the low-priority stream to ensure transfers complete before they're needed
            torch.cuda.current_stream().wait_stream(self._low_priority_stream)
            print(f"[PredictiveTransfer] Started {transferred} async transfers")
    
    def _transfer_cpu_to_gpu_async(self, kv: KVCacheManager, chunk_id: str, device_t: torch.device):
        """Asynchronously transfer chunk from CPU to GPU on current stream"""
        if chunk_id not in kv.cpu_cache:
            return
        
        entry = kv.cpu_cache[chunk_id]
        
        # Transfer tensors asynchronously (non-blocking)
        entry.keys = entry.keys.to(device_t, non_blocking=True)
        entry.values = entry.values.to(device_t, non_blocking=True)
        entry.metadata.is_on_gpu = True
        
        # Update cache management
        kv.gpu_cache[chunk_id] = entry
        kv.gpu_memory_used += entry.metadata.size_bytes
        kv.cpu_memory_used -= entry.metadata.size_bytes
        del kv.cpu_cache[chunk_id]
        
        print(f"[AsyncTransfer] {chunk_id} -> GPU (non-blocking)")
    
    def _use_cacheblend_paged_attention(
        self, 
        model, 
        tokenizer, 
        input_token: torch.Tensor, 
        kv: KVCacheManager, 
        gpu_chunk_ids: Set[str], 
        device: torch.device
    ) -> torch.Tensor:
        """
        Use CacheBlend's built-in paged attention implementation.
        This leverages the existing, tested paged attention from vllm_blend.
        """
        # Import CacheBlend's paged attention with correct path setup
        import sys
        import os
        
        # Add vllm_blend to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cacheblend_dir = os.path.dirname(current_dir)  # Go up from src/ to CacheBlend/
        vllm_blend_path = os.path.join(cacheblend_dir, 'vllm_blend')
        
        if vllm_blend_path not in sys.path:
            sys.path.insert(0, vllm_blend_path)
        
        try:
            from vllm.attention.ops.paged_attn import PagedAttention
            print(f"[CacheBlend] Successfully imported PagedAttention from {vllm_blend_path}")
        except ImportError as e:
            print(f"[CacheBlend] Failed to import PagedAttention: {e}")
            print(f"[CacheBlend] vllm_blend_path: {vllm_blend_path}")
            print(f"[CacheBlend] Path exists: {os.path.exists(vllm_blend_path)}")
            raise ImportError(f"Could not import CacheBlend's PagedAttention: {e}")
        
        print(f"[CacheBlend] Using built-in paged attention with {len(gpu_chunk_ids)} GPU chunks")
        
        # For now, fall back to standard attention with proper KV cache
        # TODO: Implement proper conversion to CacheBlend's paged format
        past_kv = self._build_past_key_values_from_kv(kv, gpu_chunk_ids)
        model_inputs = {
            "input_ids": input_token,
            "attention_mask": torch.ones_like(input_token),
            "use_cache": True,
            "return_dict": True
        }
        
        if past_kv is not None:
            cache_format = self._convert_to_cache_format(past_kv, model)
            if cache_format is not None:
                model_inputs["past_key_values"] = cache_format
                
                # Add proper position information
                if isinstance(cache_format, tuple) and len(cache_format) > 0:
                    cached_len = cache_format[0][0].shape[2] if len(cache_format[0]) > 0 else 0
                    if cached_len > 0:
                        position_ids = torch.arange(cached_len, cached_len + 1, device=device).unsqueeze(0)
                        model_inputs["position_ids"] = position_ids
                        cache_position = torch.arange(cached_len, cached_len + 1, device=device)
                        model_inputs["cache_position"] = cache_position
        
        outputs = model(**model_inputs)
        return outputs.logits[0, -1, :]
    
    def _build_past_key_values_from_kv(self, kv: KVCacheManager, gpu_chunk_ids: Set[str]):
        """
        Build past_key_values from GPU chunks with proper sequence ordering.
        
        Key insight: Each chunk was processed independently, so their KV caches
        represent separate sequence positions. We need to concatenate them in 
        the correct order to maintain sequence continuity.
        """
        if not gpu_chunk_ids:
            return None
        
        # Sort chunk IDs by their index to maintain proper sequence order
        sorted_chunk_ids = sorted(gpu_chunk_ids, key=lambda x: int(x.split('_chunk')[-1]) if '_chunk' in x else 0)
        
        entries = []
        for chunk_id in sorted_chunk_ids:
            if chunk_id in kv.gpu_cache:
                entries.append((chunk_id, kv.gpu_cache[chunk_id]))
        
        if not entries:
            return None
        
        # Get layer count from first entry
        first_entry = entries[0][1]
        num_layers = first_entry.keys.shape[0] if first_entry.keys.ndim >= 3 else 0
        if num_layers == 0:
            return None
        
        print(f"[KVCache] Building past_key_values from {len(entries)} chunks: {[cid for cid, _ in entries]}")
        
        past_key_values = []
        total_seq_len = 0
        
        for layer in range(num_layers):
            layer_keys = []
            layer_values = []
            
            for chunk_id, entry in entries:
                if entry.keys.ndim == 4:  # [L,S,H,D] -> [B,H,S,D]
                    k = entry.keys[layer].permute(1, 0, 2).unsqueeze(0)  # [1,H,S,D]
                    v = entry.values[layer].permute(1, 0, 2).unsqueeze(0)
                elif entry.keys.ndim == 5:  # [L,B,H,S,D] -> [B,H,S,D]
                    k = entry.keys[layer, 0]  # [H,S,D] -> need [B,H,S,D]
                    v = entry.values[layer, 0]
                    if k.ndim == 3:
                        k = k.unsqueeze(0)
                        v = v.unsqueeze(0)
                else:
                    print(f"[KVCache] Skipping chunk {chunk_id} with unexpected shape: {entry.keys.shape}")
                    continue
                
                layer_keys.append(k)
                layer_values.append(v)
                
                # Track total sequence length for debugging
                if layer == 0:  # Only count once per chunk
                    total_seq_len += k.shape[-2]
            
            if layer_keys:
                # Concatenate along sequence dimension (dim=-2)
                merged_k = torch.cat(layer_keys, dim=-2)  # [B,H,total_seq,D]
                merged_v = torch.cat(layer_values, dim=-2)
                past_key_values.append((merged_k, merged_v))
                
                if layer == 0:  # Debug print for first layer only
                    print(f"[KVCache] Layer {layer}: merged KV shape {merged_k.shape}, total_seq_len={total_seq_len}")
        
        if past_key_values:
            print(f"[KVCache] Built past_key_values with {len(past_key_values)} layers, total sequence length: {total_seq_len}")
            return tuple(past_key_values)
        else:
            print("[KVCache] Failed to build past_key_values - no valid entries")
            return None
    
    def _convert_to_cache_format(self, past_kv_tuple, model):
        """Convert tuple past_key_values to appropriate format for the model"""
        if past_kv_tuple is None:
            return None
        
        # Try different cache formats based on transformers version
        try:
            # Method 1: Try DynamicCache (newer transformers)
            from transformers import DynamicCache
            cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(past_kv_tuple):
                cache.update(k, v, layer_idx)
            return cache
        except ImportError:
            pass
        
        try:
            # Method 2: Try StaticCache
            from transformers import StaticCache
            # This might need different parameters
            return past_kv_tuple  # Fallback to tuple
        except ImportError:
            pass
        
        try:
            # Method 3: Check if model has specific cache class
            if hasattr(model.config, 'cache_implementation'):
                cache_class = getattr(model.config, 'cache_implementation', None)
                if cache_class == 'static':
                    return past_kv_tuple
            
            # Method 4: Try to create cache from model config
            if hasattr(model, 'get_cache'):
                cache = model.get_cache()
                for layer_idx, (k, v) in enumerate(past_kv_tuple):
                    if hasattr(cache, 'update'):
                        cache.update(k, v, layer_idx)
                return cache
        except Exception:
            pass
        
        # Method 5: Check model type specific handling
        model_type = getattr(model.config, 'model_type', 'unknown')
        if model_type == 'mistral':
            # For Mistral, we need to handle position_ids and cache_position
            return past_kv_tuple
        
        # Fallback: return tuple and hope it works
        return past_kv_tuple