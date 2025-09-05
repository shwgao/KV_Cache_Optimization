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

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device_t)
        model.eval()

        pred_samples = pred_payload.get("results", [])
        pred_trace = pred_payload.get("trace", [])

        if not isinstance(retr_samples, list) or not retr_samples:
            raise RuntimeError("Retrieval samples are empty or invalid")
        if max_samples > 0:
            pred_samples = pred_samples[:max_samples]

        # Group trace per (sample_index -> step -> promoted_indices)
        per_sample_steps: Dict[int, Dict[int, List[int]]] = {}
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

            # Per-step scheduling
            steps = per_sample_steps.get(si, {})
            if not steps:
                continue
            start_time = time.time()
            step_dur = step_duration_ms / 1000.0
            safety = safety_margin_ms / 1000.0

            step_iter = range(0, max(steps.keys()) + 1)
            if enable_progress and tqdm is not None:
                step_iter = tqdm(step_iter, desc=f"Steps (sample {si})", leave=False)

            for step in step_iter:
                now = start_time + step * step_dur
                preds = steps.get(step, [])

                sim_map: Dict[str, float] = {}
                expected_time: Dict[str, float] = {}
                for idx in preds:
                    cid = f"{sample_id}_chunk{idx}"
                    sim_map[cid] = 1.0  # unchanged: treat predicted ones as highest similarity
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
                })

        payload = {"trace": out_trace}

        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)

        return payload
