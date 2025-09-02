"""
Build KV caches from a retrieval JSON, storing top-K chunks on GPU and the rest on CPU,
using only CacheBlend kernels (no default fallbacks).

Inputs:
- retrieval JSON produced by rag_retrieval.py (contains per-sample retrieved_indices)
- model checkpoint to compute per-chunk KV (prefill only)

Output:
- A summary JSON with cache stats per sample
"""

import os
import json
import argparse
import time
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_cache_manager import KVCacheManager, KVCacheEntry


def _tokenize_chunk(tokenizer, text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    inputs = tokenizer([text], return_tensors="pt", padding=False, truncation=False)
    return {k: v.to(device) for k, v in inputs.items()}


def _prefill_get_past(model, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    with torch.inference_mode():
        out = model(
            **inputs,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    
    pkv = out.past_key_values
    return {"past_key_values": pkv}


def build_caches_for_sample(
    manager: KVCacheManager,
    model,
    tokenizer,
    sample: Dict[str, Any],
    k: int,
    device: torch.device,
    text_key_pairs: List[Tuple[int, str]],
):
    # text_key_pairs: list of (chunk_index, chunk_text) in sample order
    # read top-k indices from sample
    top_indices = sample.get("retrieved_indices")
    top_set = set(int(i) for i in top_indices[:k])

    for idx, chunk_text in text_key_pairs:
        tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
        chunk_id = f"{sample.get('id', 'sample')}_chunk{idx}"
        if idx in top_set:
            # prefill and create real KV entry only for top-k
            inputs = _tokenize_chunk(tokenizer, chunk_text, device)
            outputs = _prefill_get_past(model, inputs)
            entry = manager.create_kv_cache_entry(
                chunk_id=chunk_id,
                text=chunk_text,
                tokens=tokens,
                relevance_score=1.0,
                model_outputs=outputs,
            )
            manager.store_chunk(entry.metadata.chunk_id, entry, priority="gpu")
        else:
            # create placeholder CPU entry without prefill
            entry = manager.create_placeholder_entry(
                chunk_id=chunk_id,
                text=chunk_text,
                tokens=tokens,
                relevance_score=0.0,
            )
            manager.store_chunk(entry.metadata.chunk_id, entry, priority="cpu")


def extract_texts(sample: Dict[str, Any]) -> List[Tuple[int, str]]:
    pairs: List[Tuple[int, str]] = []
    if isinstance(sample.get("ctxs"), list):
        for i, ch in enumerate(sample["ctxs"]):
            title = (ch.get("title") or "").strip()
            text = (ch.get("text") or "").strip()
            full = f"{title}\n{text}".strip() if title else text
            if full:
                pairs.append((i, full))
    elif isinstance(sample.get("contents"), list):
        for i, it in enumerate(sample["contents"]):
            if isinstance(it, str):
                s = it.strip()
                if s:
                    pairs.append((i, s))
            elif isinstance(it, dict):
                s = (it.get("text") or it.get("content") or "").strip()
                if s:
                    pairs.append((i, s))
    return pairs


def _save_chunk(cache_dir: str, entry: KVCacheEntry) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    cdir = os.path.join(cache_dir, entry.metadata.chunk_id)
    os.makedirs(cdir, exist_ok=True)
   
    torch.save(entry.keys.cpu(), os.path.join(cdir, "keys.pt"))
    torch.save(entry.values.cpu(), os.path.join(cdir, "values.pt"))
    torch.save(entry.valid_mask.cpu(), os.path.join(cdir, "valid_mask.pt"))
    meta = {
        "chunk_id": entry.metadata.chunk_id,
        "text": entry.metadata.text,
        "tokens": entry.metadata.tokens,
        "relevance_score": entry.metadata.relevance_score,
        "access_count": entry.metadata.access_count,
        "last_access_time": entry.metadata.last_access_time,
        "size_bytes": entry.metadata.size_bytes,
        "layer_count": entry.metadata.layer_count,
        "is_on_gpu": entry.metadata.is_on_gpu,
    }
    with open(os.path.join(cdir, "metadata.json"), "w") as f:
        json.dump(meta, f)


def main() -> None:
    p = argparse.ArgumentParser(description="Build KV caches for retrieved chunks using CacheBlend kernels only")
    p.add_argument("--retrieval-json", required=True, help="Path to retrieval JSON output")
    p.add_argument("--model", required=True, help="HF model id/path to prefill")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top-k", type=int, required=True, help="Top-K chunks to keep on GPU per sample")
    p.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 = all)")
    p.add_argument("--max-gpu-chunks", type=int, default=0, help="Cap of chunks in GPU cache (0 uses top-k)")
    p.add_argument("--max-cpu-chunks", type=int, default=1000, help="Cap of chunks in CPU cache")
    p.add_argument("--gpu-mem-gb", type=float, default=40.0, help="GPU memory limit in GB for KV cache")
    p.add_argument("--cpu-mem-gb", type=float, default=100.0, help="CPU memory limit in GB for KV cache")
    p.add_argument("--dump-placements", action="store_true", help="Include per-chunk device placement in output")
    p.add_argument("--out_path", default="", help="Path to save the summary JSON")
    p.add_argument("--save-cache-dir", default="", help="If set, persist KV entries here (GPU entries only by default)")
    p.add_argument("--save-placeholders", action="store_true", help="Also save CPU placeholder entries (zero-sized)")
    args = p.parse_args()

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
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    # derive model config minimal fields for sizing
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", 4096)
    num_layers = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer", 32)
    num_heads = getattr(model.config, "num_attention_heads", None) or getattr(model.config, "n_head", 32)
    head_dim = hidden_size // num_heads
    model_config = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": getattr(model.config, "vocab_size", 32000),
    }

    manager = KVCacheManager(
        model_config=model_config,
        gpu_memory_limit_gb=args.gpu_mem_gb,
        cpu_memory_limit_gb=args.cpu_mem_gb,
        max_gpu_chunks=(args.top_k if args.max_gpu_chunks == 0 else args.max_gpu_chunks),
        max_cpu_chunks=args.max_cpu_chunks,
        device=args.device,
        require_kernels=True,
    )

    start = time.time()
    for i, sample in enumerate(samples):
        texts = extract_texts(sample)
        if not texts:
            continue
        build_caches_for_sample(manager, model, tokenizer, sample, args.top_k, device, texts)

    stats = manager.get_cache_stats()
    out_path = args.out_path
    payload = {
            "retrieval_json": args.retrieval_json,
            "model": args.model,
            "device": args.device,
            "top_k": args.top_k,
            "cache_stats": stats,
            "elapsed_sec": time.time() - start,
    }
    
    if args.save_cache_dir:
        for cid, entry in manager.gpu_cache.items():
            _save_chunk(args.save_cache_dir, entry)
        if args.save_placeholders:
            for cid, entry in manager.cpu_cache.items():
                _save_chunk(args.save_cache_dir, entry)
        payload["saved_cache_dir"] = args.save_cache_dir
    if args.dump_placements:
        payload["placements"] = {
            "gpu": [
                {
                    "chunk_id": cid,
                    "size_bytes": entry.metadata.size_bytes,
                }
                for cid, entry in manager.gpu_cache.items()
            ],
            "cpu": [
                {
                    "chunk_id": cid,
                    "size_bytes": entry.metadata.size_bytes,
                }
                for cid, entry in manager.cpu_cache.items()
            ],
        }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved cache summary to: {out_path}")



if __name__ == "__main__":
    main()


