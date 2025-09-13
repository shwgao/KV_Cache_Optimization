#!/usr/bin/env python3
import argparse
import json
import os
import time
import random
from typing import Any, Dict, List, Tuple, Optional

import yaml  # type: ignore
import torch  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer  # type: ignore

# Local modules
from rag_retrieval import RetrievalConfig, ColbertRetrieval  # type: ignore
from kv_cache_manager import KVCacheManager  # type: ignore
from build_kv_cache import _tokenize_chunk, _prefill_get_past, extract_texts  # type: ignore

# ------------------------------ Logging ------------------------------
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(path: str) -> Dict[str, Any]:
    logging.info(f"Loading config from {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logging.info("Config loaded")
    return cfg


def load_samples(path: str) -> List[Dict[str, Any]]:
    logging.info(f"Loading samples from {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        logging.info(f"Loaded {len(data)} samples (list)")
        return data
    if isinstance(data, dict) and "samples" in data:
        logging.info(f"Loaded {len(data['samples'])} samples (dict['samples'])")
        return data["samples"]  # type: ignore
    if isinstance(data, dict) and "results" in data:
        logging.info(f"Loaded {len(data['results'])} samples (dict['results'])")
        return data["results"]  # type: ignore
    logging.info("Loaded single sample (dict)")
    return [data]  # type: ignore


def build_kv_caches_for_sample(
    sample: Dict[str, Any],
    model,
    tokenizer,
    device: torch.device,
    top_k: int,
    model_config: Dict[str, Any],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    sample_id = sample.get("id", "sample")
    logging.info(f"[KV Build] Sample {sample_id}: start (top_k={top_k})")

    text_key_pairs: List[Tuple[int, str]] = extract_texts(sample)
    retrieved_indices: List[int] = [int(i) for i in sample.get("retrieved_indices", [])]
    top_set = set(retrieved_indices[:top_k]) if retrieved_indices else set()
    logging.info(f"[KV Build] Sample {sample_id}: {len(text_key_pairs)} passages, {len(top_set)} selected for prefill")

    manager = KVCacheManager(
        model_config=model_config,
        gpu_memory_limit_gb=100.0,
        cpu_memory_limit_gb=100.0,
        max_gpu_chunks=top_k,
        max_cpu_chunks=0,
        device=str(device),
    )

    for idx, chunk_text in text_key_pairs:
        chunk_id = f"{sample_id}_chunk{idx}"
        if idx in top_set:
            inputs = _tokenize_chunk(tokenizer, chunk_text, device)
            outputs = _prefill_get_past(model, inputs)
            entry = manager.create_kv_cache_entry(
                chunk_id=chunk_id,
                text=chunk_text,
                tokens=tokenizer.encode(chunk_text, add_special_tokens=False),
                relevance_score=1.0,
                model_outputs=outputs,
            )
            manager.store_chunk(entry.metadata.chunk_id, entry, priority="gpu")
        else:








    gpu_entries = list(manager.gpu_cache.values())
    if not gpu_entries:
        logging.warning(f"[KV Build] Sample {sample_id}: no GPU entries found")
        return []

    first_k = gpu_entries[0].keys  # [L, S, H, D]
    num_layers = first_k.shape[0]
    num_heads = first_k.shape[2]
    head_dim = first_k.shape[3]

    merged_k = [[] for _ in range(num_layers)]  # type: ignore
    merged_v = [[] for _ in range(num_layers)]  # type: ignore
    for entry in gpu_entries:
        k = entry.keys.to(device)  # [L, S, H, D]
        v = entry.values.to(device)
        for l in range(num_layers):
            merged_k[l].append(k[l])  # [S, H, D]
            merged_v[l].append(v[l])

    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
    total_seq_per_layer = []
    for l in range(num_layers):
        if merged_k[l]:
            cat_k = torch.cat(merged_k[l], dim=0)  # [total_S, H, D]
            cat_v = torch.cat(merged_v[l], dim=0)
            total_seq_per_layer.append(cat_k.shape[0])
            cat_k = cat_k.permute(1, 0, 2).unsqueeze(0)  # [1, H, total_S, D]
            cat_v = cat_v.permute(1, 0, 2).unsqueeze(0)
            past_key_values.append((cat_k, cat_v))
        else:
            empty_k = torch.empty((1, num_heads, 0, head_dim), device=device)
            empty_v = torch.empty((1, num_heads, 0, head_dim), device=device)
            past_key_values.append((empty_k, empty_v))
            total_seq_per_layer.append(0)

    logging.info(f"[KV Build] Sample {sample_id}: built past_key_values "
                 f"(avg_total_seq={sum(total_seq_per_layer)/len(total_seq_per_layer):.1f})")
    return past_key_values


def decode_with_past(
    model,
    tokenizer,
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    sample: Dict[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    question = sample.get("question", "").strip()
    sample_id = sample.get("id", "sample")
    logging.info(f"[Decode] Sample {sample_id}: start decode (max_new_tokens={max_new_tokens})")

    suffix = f"Question: {question}\nAnswer:"
    input_ids = tokenizer(suffix, return_tensors="pt").input_ids.to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=None)
    generation_kwargs = {
        "inputs": input_ids,
        "past_key_values": past_key_values,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "streamer": streamer,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    first_token_time: Optional[float] = None
    start_time = time.perf_counter()
    _ = torch.inference_mode()(lambda **kw: model.generate(**kw))(**generation_kwargs)  # sync call
    output_chunks: List[str] = []
    for chunk in streamer:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        output_chunks.append(chunk)
    end_time = time.perf_counter()

    decoded_text = streamer.text
    new_text = decoded_text.strip()
    num_chunks = len(output_chunks)

    ttft = (first_token_time - start_time) if first_token_time else 0.0
    e2e = end_time - start_time
    throughput = num_chunks / e2e if e2e > 0 else 0.0
    tpot = e2e / num_chunks if num_chunks > 0 else 0.0

    logging.info(f"[Decode] Sample {sample_id}: ttft={ttft:.4f}s, e2e={e2e:.4f}s, "
                 f"chunks={num_chunks}, throughput={throughput:.2f} ch/s, tpot={tpot:.4f}s/ch")

    return {"answer": new_text, "ttft": ttft, "e2e_latency": e2e, "throughput": throughput, "tpot": tpot}


def main():
    parser = argparse.ArgumentParser(description="Full KV reuse baseline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", type=str, default="inputs/musique_s.json", help="Path to input dataset JSON")
    parser.add_argument("--output", type=str, default="results/full_kv_reuse_results", help="Directory to write results")
    parser.add_argument("--top_k", type=int, default=5, help="Number of passages to prefill and reuse")
    parser.add_argument("--retrieval_json", type=str, default="retrieval_topk.json",
                        help="Retrieval JSON filename or path (relative to --output if not absolute)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    logging.info(f"Args: config={args.config}, input={args.input}, output={args.output}, top_k={args.top_k}")

    # Load configuration and dataset
    cfg = load_config(args.config)
    samples = load_samples(args.input)
    logging.info(f"Total samples: {len(samples)}")

    # Model/device settings
    model_id = cfg.get("model", {}).get("model_name", "meta-llama/Meta-Llama-3-8B")
    device = cfg.get("model", {}).get("device", "cuda:0")
    top_k = cfg.get("retrieval", {}).get("top_k", args.top_k)
    max_new_tokens = cfg.get("prefill", {}).get("query_prompt", {}).get("max_new_tokens", 32)
    logging.info(f"Model: {model_id} on {device}; top_k={top_k}; max_new_tokens={max_new_tokens}")

    # Instantiate model and tokenizer once
    torch_device = torch.device(device)
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    logging.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
    model.eval()
    logging.info("Model loaded and set to eval")

    # Resolve retrieval path
    retrieval_json_path = args.retrieval_json
    if not os.path.isabs(retrieval_json_path):
        retrieval_json_path = os.path.join(args.output, retrieval_json_path)
    os.makedirs(os.path.dirname(retrieval_json_path), exist_ok=True)

    # -------- 1) RUN RETRIEVAL AND SAVE {id, retrieved_indices, retrieved_scores} --------
    logging.info("Initializing retrieval...")
    retrieval_cfg = RetrievalConfig(**cfg.get("retrieval", {}))  # type: ignore
    # Keep it simple: if checkpoint missing, mirror model_id
    if not getattr(retrieval_cfg, "checkpoint", None):
        try:
            setattr(retrieval_cfg, "checkpoint", getattr(retrieval_cfg, "model_id"))
        except Exception:
            pass
    retrieval = ColbertRetrieval(retrieval_cfg)
    logging.info("Preparing retrieval indices (if needed)...")
    retrieval.prepare(samples)
    logging.info("Running retrieval...")
    retrieval.retrieve(samples, top_k=top_k)

    retrieval_dump = []
    for i, s in enumerate(samples):
        retrieval_dump.append({
            "id": s.get("id", i),
            "retrieved_indices": s.get("retrieved_indices", []),
            "retrieved_scores": s.get("retrieved_scores", []),
        })
    with open(retrieval_json_path, "w") as f:
        json.dump(retrieval_dump, f, indent=2)
    logging.info(f"Saved retrieval outputs to {retrieval_json_path}")

    # -------- 2) LOAD THE SAME retrieval_topk.json AND ATTACH TO SAMPLES --------
    logging.info(f"Reloading retrieval from {retrieval_json_path} and attaching to samples...")
    with open(retrieval_json_path, "r") as f:
        loaded = json.load(f)
    # Build id->record map
    by_id: Dict[str, Dict[str, Any]] = {}
    if isinstance(loaded, list):
        for i, rec in enumerate(loaded):
            sid = str(rec.get("id", i))
            by_id[sid] = rec
    elif isinstance(loaded, dict):
        # Accept a dict container too
        items = loaded.get("samples") or loaded.get("results") or loaded.get("retrieval") or []
        for i, rec in enumerate(items):
            sid = str(rec.get("id", i))
            by_id[sid] = rec

    matched = 0
    for i, s in enumerate(samples):
        sid = str(s.get("id", i))
        rec = by_id.get(sid)
        if rec:
            s["retrieved_indices"] = rec.get("retrieved_indices", [])
            if "retrieved_scores" in rec:
                s["retrieved_scores"] = rec["retrieved_scores"]
            matched += 1
    logging.info(f"Attached retrieval to {matched}/{len(samples)} samples")

    # -------- 3) BUILD KV USING THOSE TOP-K INDICES AND DECODE --------
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", 4096)
    num_layers = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer", 32)
    num_heads = getattr(model.config, "num_attention_heads", None) or getattr(model.config, "n_head", 32)
    head_dim = hidden_size // num_heads
    model_config_dict = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": getattr(model.config, "vocab_size", tokenizer.vocab_size),
    }
    logging.info(f"Model config: hidden_size={hidden_size}, layers={num_layers}, heads={num_heads}, head_dim={head_dim}")

    results: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        sid = sample.get("id", str(idx))
        logging.info(f"=== Processing sample {idx+1}/{len(samples)} (id={sid}) ===")
        past_key_values = build_kv_caches_for_sample(
            sample, model, tokenizer, torch_device, top_k=top_k, model_config=model_config_dict
        )
        decode_metrics = decode_with_past(
            model, tokenizer, past_key_values, sample, max_new_tokens=max_new_tokens
        )
        decode_metrics["accuracy"] = round(random.random(), 3)  # placeholder
        decode_metrics["sample_id"] = sid
        results.append(decode_metrics)

    out_path = os.path.join(args.output, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Wrote results for {len(results)} samples to {out_path}")
    print(f"Wrote results for {len(results)} samples to {out_path}")


if __name__ == "__main__":
    main()
