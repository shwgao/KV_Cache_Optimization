#!/usr/bin/env python3
"""
Speculative decode driver that:
- Rebuilds GPU-resident KV caches for top-K chunks (and placeholders for the rest)
- Runs a few decode steps while proactively promoting likely-needed CPU chunks
- Uses CacheBlend kernels for cache ops where available

Note: This focuses on cache preparation/promotion. It does not inject external KV into HF
generate; instead, it simulates per-step promotions using a lightweight relevance proxy.
"""

import os
import json
import argparse
import time
from typing import Any, Dict, List, Tuple
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from tqdm import tqdm

from kv_cache_manager import KVCacheManager, KVCacheEntry, ChunkMetadata
from build_kv_cache import extract_texts, _tokenize_chunk, _prefill_get_past


def _score_cpu_chunks(tokenizer, model, device, sample: Dict[str, Any], cpu_indices: List[int]) -> List[Tuple[int, float]]:
    """Innovative lightweight relevance proxy (no published method):
    - Compute a query embedding by averaging input embeddings of the question.
    - Approximate each chunk by averaging input embeddings of the first 32 tokens only.
    - Score = cosine similarity(query_emb, chunk_centroid).
    This is extremely cheap, model-agnostic, and works without full forward pass.
    """
    query = str(sample.get("question", ""))
    if not query or not cpu_indices:
        return []
    with torch.no_grad():
        # Query embedding via token embedding table
        ids = tokenizer.encode(query, add_special_tokens=False)[:64]
        ids_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        emb = model.get_input_embeddings()(ids_t)
        qvec = emb.mean(dim=1).squeeze(0)  # [hidden]
        qnorm = torch.linalg.norm(qvec) + 1e-6

    texts = extract_texts(sample)
    scores: List[Tuple[int, float]] = []
    for idx in cpu_indices:
        if idx < 0 or idx >= len(texts):
            continue
        text = texts[idx][1]
        ids = tokenizer.encode(text, add_special_tokens=False)[:32]
        if not ids:
            continue
        with torch.no_grad():
            t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            e = model.get_input_embeddings()(t)
            c = e.mean(dim=1).squeeze(0)
            cnorm = torch.linalg.norm(c) + 1e-6
            sim = float(torch.dot(qvec, c) / (qnorm * cnorm))
        scores.append((idx, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _load_cache_dir_into_manager(cache_dir: str, manager: KVCacheManager) -> None:
    if not cache_dir or not os.path.isdir(cache_dir):
        return
    for chunk_id in os.listdir(cache_dir):
        cdir = os.path.join(cache_dir, chunk_id)
        if not os.path.isdir(cdir):
            continue
        meta_path = os.path.join(cdir, "metadata.json")
        keys_path = os.path.join(cdir, "keys.pt")
        values_path = os.path.join(cdir, "values.pt")
        mask_path = os.path.join(cdir, "valid_mask.pt")
        if not (os.path.isfile(meta_path) and os.path.isfile(keys_path) and os.path.isfile(values_path) and os.path.isfile(mask_path)):
            continue
        with open(meta_path, "r") as f:
            meta = json.load(f)
        keys = torch.load(keys_path, map_location="cpu")
        values = torch.load(values_path, map_location="cpu")
        valid_mask = torch.load(mask_path, map_location="cpu")
        entry = KVCacheEntry(
            keys=keys,
            values=values,
            valid_mask=valid_mask,
            metadata=ChunkMetadata(
                chunk_id=meta.get("chunk_id", chunk_id),
                text=meta.get("text", ""),
                tokens=meta.get("tokens", []),
                relevance_score=meta.get("relevance_score", 0.0),
                access_count=meta.get("access_count", 0),
                last_access_time=meta.get("last_access_time", 0.0),
                size_bytes=meta.get("size_bytes", 0),
                layer_count=meta.get("layer_count", 0),
                is_on_gpu=bool(meta.get("is_on_gpu", False)),
            ),
        )
        priority = "gpu" if entry.metadata.is_on_gpu else "cpu"
        manager.store_chunk(entry.metadata.chunk_id, entry, priority=priority)


def _ensure_materialized(manager: KVCacheManager, tokenizer, model, device, sample: Dict[str, Any], idx: int) -> None:
    """If a CPU placeholder exists for chunk idx, compute real KV and replace it on GPU."""
    texts = extract_texts(sample)
    if idx < 0 or idx >= len(texts):
        return
    chunk_id = f"{sample.get('id', 'sample')}_chunk{idx}"
    # If already in GPU, nothing to do
    if chunk_id in manager.gpu_cache:
        return
    # If present on CPU with real KV, move to GPU without recompute
    if chunk_id in manager.cpu_cache:
        entry = manager.cpu_cache[chunk_id]
        has_real_kv = (entry.metadata.layer_count > 0) and (entry.keys.numel() > 0)
        if has_real_kv:
            try:
                del manager.cpu_cache[chunk_id]
                manager.cpu_memory_used -= entry.metadata.size_bytes
            except Exception:
                pass
            manager.store_chunk(chunk_id, entry, priority="gpu")
            return
    # Compute KV and store to GPU (replaces placeholder if present)
    text = texts[idx][1]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    inputs = _tokenize_chunk(tokenizer, text, device)
    outputs = _prefill_get_past(model, inputs)
    entry = manager.create_kv_cache_entry(
        chunk_id=chunk_id,
        text=text,
        tokens=tokens,
        relevance_score=1.0,
        model_outputs=outputs,
    )
    manager.store_chunk(chunk_id, entry, priority="gpu")


def main() -> None:
    p = argparse.ArgumentParser(description="Speculative decode with proactive CPU->GPU promotion using CacheBlend kernels")
    p.add_argument("--retrieval-json", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--steps", type=int, default=16, help="Number of speculative steps to simulate")
    p.add_argument("--promote-per-step", type=int, default=2, help="How many CPU chunks to promote each step")
    p.add_argument("--max-samples", type=int, default=1)
    p.add_argument("--gpu-mem-gb", type=float, default=40.0)
    p.add_argument("--cpu-mem-gb", type=float, default=100.0)
    p.add_argument("--max-gpu-chunks", type=int, default=0)
    p.add_argument("--max-cpu-chunks", type=int, default=1000)
    p.add_argument("--out", default="results/decoding/speculative_trace.json")
    p.add_argument("--load-cache-dir", default="", help="If set, load KV entries from this directory instead of rebuilding")
    p.add_argument("--async-prefetch", action="store_true", help="Run a background prefetcher to promote chunks in parallel")
    p.add_argument("--prefetch-interval-ms", type=int, default=50, help="Sleep interval between prefetch iterations")
    p.add_argument("--prefetch-batch-size", type=int, default=2, help="How many chunks to prefetch per iteration")
    p.add_argument("--gen-max-new-tokens", type=int, default=64, help="Max new tokens to decode for the answer")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    with open(args.retrieval_json, "r") as f:
        data = json.load(f)
    samples = data.get("results") if isinstance(data, dict) and "results" in data else data
    if not isinstance(samples, list) or len(samples) == 0:
        raise SystemExit("Retrieval JSON is empty or invalid")
    samples = samples[: args.max_samples]

    # Build or load caches
    # Mirror manager settings with strict top-k cap
    manager = KVCacheManager(
        model_config={
            "hidden_size": getattr(model.config, "hidden_size", 4096),
            "num_layers": getattr(model.config, "num_hidden_layers", 32),
            "num_attention_heads": getattr(model.config, "num_attention_heads", 32),
            "head_dim": getattr(model.config, "hidden_size", 4096)
            // max(1, getattr(model.config, "num_attention_heads", 32)),
            "vocab_size": getattr(model.config, "vocab_size", 32000),
        },
        gpu_memory_limit_gb=args.gpu_mem_gb,
        cpu_memory_limit_gb=args.cpu_mem_gb,
        max_gpu_chunks=(args.top_k if args.max_gpu_chunks == 0 else args.max_gpu_chunks),
        max_cpu_chunks=args.max_cpu_chunks,
        device=args.device,
        require_kernels=True,
    )

    trace = []
    # If a cache dir is provided, load cached entries once and skip initial rebuild
    if args.load_cache_dir:
        _load_cache_dir_into_manager(args.load_cache_dir, manager)

    # Simple lock to guard manager during async promotions
    mgr_lock = threading.Lock()

    def prefetch_loop(stop_event: threading.Event, sample: Dict[str, Any]):
        while not stop_event.is_set():
            try:
                # Determine CPU candidates
                texts = extract_texts(sample)
                cpu_indices = []
                with mgr_lock:
                    for i, _ in texts:
                        if f"{sample.get('id', 'sample')}_chunk{i}" in manager.cpu_cache:
                            cpu_indices.append(i)
                if not cpu_indices:
                    stop_event.wait(args.prefetch_interval_ms / 1000.0)
                    continue
                scored = _score_cpu_chunks(tokenizer, model, device, sample, cpu_indices)
                promote = [idx for idx, _ in scored[: args.prefetch_batch_size]]
                for idx in promote:
                    with mgr_lock:
                        _ensure_materialized(manager, tokenizer, model, device, sample, idx)
            except Exception:
                pass
            stop_event.wait(args.prefetch_interval_ms / 1000.0)

    prefetch_thread = None
    stop_event: threading.Event = threading.Event()

    results: List[Dict[str, Any]] = []
    for si, sample in enumerate(tqdm(samples, desc="Samples", total=len(samples))):
        texts = extract_texts(sample)
        if not texts:
            continue
        # If not loading from disk, build initial caches matching build_kv_cache logic
        if not args.load_cache_dir:
            top_idxs = set(int(i) for i in (sample.get("retrieved_indices") or [])[: args.top_k])
            for idx, chunk_text in texts:
                if idx in top_idxs:
                    inputs = _tokenize_chunk(tokenizer, chunk_text, device)
                    outputs = _prefill_get_past(model, inputs)
                    entry = manager.create_kv_cache_entry(
                        chunk_id=f"{sample.get('id', 'sample')}_chunk{idx}",
                        text=chunk_text,
                        tokens=tokenizer.encode(chunk_text, add_special_tokens=False),
                        relevance_score=1.0,
                        model_outputs=outputs,
                    )
                    with mgr_lock:
                        manager.store_chunk(entry.metadata.chunk_id, entry, priority="gpu")
                else:
                    ph = manager.create_placeholder_entry(
                        chunk_id=f"{sample.get('id', 'sample')}_chunk{idx}",
                        text=chunk_text,
                        tokens=tokenizer.encode(chunk_text, add_special_tokens=False),
                        relevance_score=0.0,
                    )
                    with mgr_lock:
                        manager.store_chunk(ph.metadata.chunk_id, ph, priority="cpu")

        # Start background prefetcher if requested
        if args.async_prefetch and prefetch_thread is None:
            prefetch_thread = threading.Thread(target=prefetch_loop, args=(stop_event, sample), daemon=True)
            prefetch_thread.start()

        # Speculative loop: score CPU chunks each step; promote top-M and log
        for step in range(args.steps):
            with mgr_lock:
                cpu_indices = [i for i, _ in texts if f"{sample.get('id', 'sample')}_chunk{i}" in manager.cpu_cache]
            scored = _score_cpu_chunks(tokenizer, model, device, sample, cpu_indices)
            promote = [idx for idx, _ in scored[: args.promote_per_step]]
            if not args.async_prefetch:
                for idx in promote:
                    with mgr_lock:
                        _ensure_materialized(manager, tokenizer, model, device, sample, idx)
            trace.append({
                "sample_index": si,
                "step": step,
                "promoted_indices": promote,
                "gpu_chunks": len(manager.gpu_cache),
                "cpu_chunks": len(manager.cpu_cache),
            })

        # Build prompt from current top-K (by relevance flag) among GPU cache or retrieved_indices
        top_idxs = list(int(i) for i in (sample.get("retrieved_indices") or [])[: args.top_k])
        docs = []
        for rank, idx in enumerate(top_idxs, 1):
            if idx < 0 or idx >= len(texts):
                continue
            docs.append(f"Document {rank}: {texts[idx][1]}")
        prompt = "\n\n".join(docs + [f"\nQuestion: {sample.get('question','')}", "\nAnswer:"])

        # Decode with TTFT measurement (time to first token)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=args.gen_max_new_tokens,
            do_sample=False,
            use_cache=True,
            streamer=streamer,
            return_dict_in_generate=False,
        )
        started = time.time()
        t_first: List[float] = []

        def _run_gen():
            with torch.inference_mode():
                _ = model.generate(**gen_kwargs)

        t = threading.Thread(target=_run_gen, daemon=True)
        t.start()
        answer_fragments: List[str] = []
        try:
            first_piece = next(streamer)
            t_first.append(time.time())
            answer_fragments.append(first_piece)
        except StopIteration:
            pass
        for piece in streamer:
            answer_fragments.append(piece)
        t.join()
        ttft_ms = (t_first[0] - started) * 1000.0 if t_first else None
        answer_text = "".join(answer_fragments)

        results.append({
            "sample_index": si,
            "ttft_ms": ttft_ms,
            "answer": answer_text,
        })

    # Stop background prefetcher
    if prefetch_thread is not None:
        stop_event.set()
        prefetch_thread.join(timeout=1.0)

    with open(args.out, "w") as f:
        json.dump({"trace": trace, "results": results}, f, indent=2)
    print(f"Saved speculative decode trace to: {args.out}")


if __name__ == "__main__":
    main()


