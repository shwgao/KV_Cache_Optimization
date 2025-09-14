#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple, Optional

import yaml  # type: ignore
import torch  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer  # type: ignore

# Local modules
from rag_retrieval import RetrievalConfig, ColbertRetrieval  # type: ignore
from build_kv_cache import extract_texts  # type: ignore


# --------------------------- I/O helpers ---------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]  # type: ignore
    if isinstance(data, dict) and "results" in data:
        return data["results"]  # type: ignore
    return [data]  # type: ignore


# --------------------------- Retrieval ---------------------------

def run_retrieval(samples: List[Dict[str, Any]], cfg: Dict[str, Any], top_k: int) -> None:
    """Run ColBERT retrieval and attach results to samples."""
    retrieval_cfg = RetrievalConfig(**cfg.get("retrieval", {}))
    if not hasattr(retrieval_cfg, "checkpoint") or not retrieval_cfg.checkpoint:
        retrieval_cfg.checkpoint = getattr(retrieval_cfg, "model_id", "colbert-ir/colbertv2.0")
    retrieval = ColbertRetrieval(retrieval_cfg)
    retrieval.prepare(samples)
    retrieval.retrieve(samples, top_k=top_k)


# --------------------------- Prompt building ---------------------------

def build_prompt_from_topk(sample: Dict[str, Any], top_k: int) -> str:
    """
    Concatenate the top-k retrieved chunks and the question into a simple prompt.
    """
    # Collect text for retrieved indices
    text_key_pairs: List[Tuple[int, str]] = extract_texts(sample)  # [(idx, text), ...]
    idx2text = {i: t for i, t in text_key_pairs}
    retrieved_indices: List[int] = [int(i) for i in sample.get("retrieved_indices", [])]
    sel = retrieved_indices[:top_k] if retrieved_indices else []

    chunks = [idx2text[i] for i in sel if i in idx2text]
    context = "\n\n".join(f"- {c}" for c in chunks) if chunks else ""
    question = (sample.get("question") or "").strip()

    if context:
        return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    else:
        return f"Question: {question}\nAnswer:"


# --------------------------- Decoding (full recompute) ---------------------------

def decode_full_recompute(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """
    Generate answer by fully prefilling on the entire prompt (no past_key_values).
    Returns only timing/throughput metrics and the answer text.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(device)
    if input_ids.shape[1] == 0:
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        input_ids = torch.tensor([[bos]], device=device)

    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs["streamer"] = streamer

    first_token_time: Optional[float] = None
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            model.generate(**generation_kwargs)
    except Exception as e:
        return {
            "answer": f"Generation failed: {str(e)}",
            "ttft": 0.0,
            "e2e_latency": 0.0,
            "throughput": 0.0,
            "tpot": 0.0,
        }

    chunks: List[str] = []
    try:
        for ch in streamer:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            chunks.append(ch)
    except Exception:
        pass

    end = time.perf_counter()
    text = "".join(chunks).strip()
    num_tokens = len(chunks)
    ttft = (first_token_time - start) if first_token_time else 0.0
    e2e_latency = end - start
    throughput = (num_tokens / e2e_latency) if e2e_latency > 0 else 0.0
    tpot = (e2e_latency / num_tokens) if num_tokens > 0 else 0.0

    return {
        "answer": text,
        "ttft": ttft,
        "e2e_latency": e2e_latency,
        "throughput": throughput,
        "tpot": tpot,
    }


# --------------------------- Main ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full KV recompute baseline (no cache reuse), timing only")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", type=str, default="inputs/musique_s.json", help="Path to input dataset JSON")
    parser.add_argument("--output", type=str, default="results/full_kv_recompute_results", help="Directory to write results")
    parser.add_argument("--top_k", type=int, default=5, help="Number of passages to include as context")
    parser.add_argument("--retrieval_json", type=str, default="retrieval_topk.json", help="Retrieval JSON filename")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cfg = load_config(args.config)
    samples = load_samples(args.input)

    model_name = cfg.get("model", {}).get("model_name", "meta-llama/Meta-Llama-3-8B")
    device_name = cfg.get("model", {}).get("device", "cuda:0")
    top_k = cfg.get("retrieval", {}).get("top_k", args.top_k)
    max_new_tokens = cfg.get("prefill", {}).get("query_prompt", {}).get("max_new_tokens", 32)

    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if "cuda" in device_name else None,
    )
    model.eval()

    # Retrieval (load or run)
    retrieval_json_path = os.path.join(args.output, args.retrieval_json)
    if os.path.exists(retrieval_json_path):
        with open(retrieval_json_path, "r") as f:
            retrieval_data = json.load(f)
        retrieval_by_id: Dict[str, Any] = {}
        if isinstance(retrieval_data, list):
            for item in retrieval_data:
                retrieval_by_id[str(item.get("id", ""))] = item
        for i, sample in enumerate(samples):
            sample_id = str(sample.get("id", i))
            if sample_id in retrieval_by_id:
                sample.update(retrieval_by_id[sample_id])
    else:
        run_retrieval(samples, cfg, top_k)
        retrieval_results = []
        for sample in samples:
            retrieval_results.append(
                {
                    "id": sample.get("id"),
                    "retrieved_indices": sample.get("retrieved_indices", []),
                    "retrieved_scores": sample.get("retrieved_scores", []),
                }
            )
        with open(retrieval_json_path, "w") as f:
            json.dump(retrieval_results, f, indent=2)

    # Process samples
    results: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        sample_id = sample.get("id", str(idx))
        try:
            prompt = build_prompt_from_topk(sample, top_k)
            decode_result = decode_full_recompute(model, tokenizer, prompt, max_new_tokens)
            decode_result.update({"sample_id": sample_id})
            results.append(decode_result)
        except Exception as e:
            results.append(
                {
                    "sample_id": sample_id,
                    "answer": f"Error: {str(e)}",
                    "ttft": 0.0,
                    "e2e_latency": 0.0,
                    "throughput": 0.0,
                    "tpot": 0.0,
                }
            )

    # Averages over timing/throughput metrics
    if results:
        total_ttft = sum(res.get("ttft", 0.0) for res in results)
        total_e2e = sum(res.get("e2e_latency", 0.0) for res in results)
        total_throughput = sum(res.get("throughput", 0.0) for res in results)
        total_tpot = sum(res.get("tpot", 0.0) for res in results)
        n = len(results)
        results.append({
            "sample_id": "average",
            "answer": "average_metrics",
            "ttft": total_ttft / n,
            "e2e_latency": total_e2e / n,
            "throughput": total_throughput / n,
            "tpot": total_tpot / n,
        })

    results_path = os.path.join(args.output, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    processed = len(results) - (1 if results and results[-1].get("sample_id") == "average" else 0)
    print(f"Completed processing {processed} samples. Results saved to {results_path}")


if __name__ == "__main__":
    main()
