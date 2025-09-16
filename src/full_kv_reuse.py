#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple, Optional

import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

# If you have your ColBERT wrapper, keep these; otherwise stub them out.
from rag_retrieval import RetrievalConfig, ColbertRetrieval  # type: ignore

# Optional: if you already have this helper; otherwise we fall back below.
try:
    from build_kv_cache import extract_texts  # type: ignore
except Exception:
    extract_texts = None  # we'll use a fallback extractor


# ---------------------------- utils ----------------------------

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


def fallback_extract_texts(sample: Dict[str, Any]) -> List[Tuple[int, str]]:
    """
    Try common fields if build_kv_cache.extract_texts is unavailable.
    Expected formats:
      - sample["retrieved"] = [{"idx": int, "text": str}, ...]
      - sample["retrieved_texts"] & sample["retrieved_indices"]
      - sample["passages"] = [{"id": int, "content": str}, ...]
    """
    out: List[Tuple[int, str]] = []
    if isinstance(sample.get("retrieved"), list):
        for it in sample["retrieved"]:
            idx = int(it.get("idx", len(out)))
            txt = str(it.get("text", "")).strip()
            if txt:
                out.append((idx, txt))
        return out
    if sample.get("retrieved_texts") and sample.get("retrieved_indices"):
        for idx, txt in zip(sample["retrieved_indices"], sample["retrieved_texts"]):
            if txt:
                out.append((int(idx), str(txt)))
        return out
    if isinstance(sample.get("passages"), list):
        for it in sample["passages"]:
            idx = int(it.get("id", len(out)))
            txt = str(it.get("content", "")).strip()
            if txt:
                out.append((idx, txt))
        return out
    # fallback: if the sample has a "context" string, treat it as a single chunk
    ctx = str(sample.get("context", "")).strip()
    if ctx:
        out.append((0, ctx))
    return out


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        # most causal LMs are fine using EOS as PAD
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ----------------------- retrieval wrapper ----------------------

def run_retrieval(samples: List[Dict[str, Any]], cfg: Dict[str, Any], top_k: int) -> None:
    """
    Run ColBERT retrieval and attach 'retrieved_indices' (and optionally texts)
    to each sample in-place.
    """
    rconf = RetrievalConfig(**cfg.get("retrieval", {}))
    if not getattr(rconf, "checkpoint", None):
        rconf.checkpoint = getattr(rconf, "model_id", "colbert-ir/colbertv2.0")
    retriever = ColbertRetrieval(rconf)
    retriever.prepare(samples)
    retriever.retrieve(samples, top_k=top_k)


# --------------------- KV prefill (one pass) --------------------

def prefill_topk_kv(
    model,
    tokenizer,
    device: torch.device,
    sample: Dict[str, Any],
    top_k: int,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]:
    """
    Build past_key_values for the *concatenation* of the top-k retrieved chunks
    in a single forward pass (correct RoPE). Returns (past_key_values, cached_len).

    cached_len is the total number of prompt tokens contributed by the chunks.
    """
    # 1) pick top-k indices (if present), and extract chunk texts (idx, text)
    retrieved_indices: List[int] = [int(i) for i in sample.get("retrieved_indices", [])]
    top_set = set(retrieved_indices[:top_k]) if retrieved_indices else None

    pairs: List[Tuple[int, str]]
    if extract_texts is not None:
        pairs = extract_texts(sample)  # type: ignore
    else:
        pairs = fallback_extract_texts(sample)

    if not pairs:
        return [], 0

    # 2) order as in retrieved_indices when available; otherwise keep given order
    ordered_chunks: List[str] = []
    if top_set is not None:
        # keep only those in top_set, in retrieved_indices order
        idx_to_text = {i: t for i, t in pairs}
        for i in retrieved_indices:
            if i in top_set and i in idx_to_text and idx_to_text[i]:
                ordered_chunks.append(idx_to_text[i])
        # If we didn't find any text (e.g., retrieval-only indices), bail out
        if not ordered_chunks:
            return [], 0
    else:
        # take the first top_k texts as they appear
        ordered_chunks = [t for _, t in pairs[:top_k] if t]

    # 3) tokenize concatenating with a separator to reduce accidental merges
    sep_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        sep_ids = [tokenizer.eos_token_id]
    elif hasattr(tokenizer, "encode"):
        sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)

    concat_ids: List[int] = []
    for j, text in enumerate(ordered_chunks):
        ids = tokenizer.encode(text, add_special_tokens=False)
        concat_ids.extend(ids)
        if j != len(ordered_chunks) - 1 and sep_ids:
            concat_ids.extend(sep_ids)

    if not concat_ids:
        return [], 0

    input_ids = torch.tensor([concat_ids], device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)

    # out.past_key_values: list of L tuples (k, v) with shape [b, h, s, d]
    pkv: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for (k, v) in out.past_key_values:  # type: ignore[attr-defined]
        pkv.append((k.contiguous(), v.contiguous()))

    cached_len = input_ids.shape[1]
    return pkv, cached_len


# ------------------- decode with streaming TTFT -----------------

def format_user_prompt(tokenizer, question: str) -> torch.Tensor:
    """Use chat template when available; otherwise a plain QA suffix."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": question}]
        ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        return ids
    # fallback
    suffix = f"Question: {question}\nAnswer:"
    ids = tokenizer(suffix, return_tensors="pt", add_special_tokens=True).input_ids
    return ids


def decode_with_past(
    model,
    tokenizer,
    device: torch.device,
    pkv: List[Tuple[torch.Tensor, torch.Tensor]],
    cached_len: int,
    sample: Dict[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    """
    Streamed generation with proper TTFT:
      - generation runs in a background thread,
      - we consume TextIteratorStreamer concurrently.
    """
    question = (sample.get("question") or "").strip()
    input_ids = format_user_prompt(tokenizer, question).to(device)

    # Create attention mask to avoid pad token warnings
    attention_mask = torch.ones_like(input_ids)
    
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # attach past_key_values + positions so we don't recompute prefill
    if pkv and any(k.shape[2] > 0 for (k, _) in pkv):
        # position ids start after cached_len
        pos = torch.arange(cached_len, cached_len + input_ids.shape[1], device=device).unsqueeze(0)
        generation_kwargs["past_key_values"] = tuple((k.contiguous(), v.contiguous()) for (k, v) in pkv)
        generation_kwargs["position_ids"] = pos
        
        # Extend attention mask to include cached tokens
        cached_attention_mask = torch.ones(input_ids.shape[0], cached_len, device=device, dtype=attention_mask.dtype)
        generation_kwargs["attention_mask"] = torch.cat([cached_attention_mask, attention_mask], dim=1)
        
        # some models accept cache_position; guard it and ensure it's not empty
        if "cache_position" in model.generate.__code__.co_varnames:
            cache_pos = torch.arange(cached_len, cached_len + input_ids.shape[1], device=device)
            if cache_pos.numel() > 0:  # Only set if not empty
                generation_kwargs["cache_position"] = cache_pos

    # streaming
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs["streamer"] = streamer

    first_tok_t: Optional[float] = None
    t0 = time.perf_counter()

    # background generation
    import threading
    th = threading.Thread(target=lambda: model.generate(**generation_kwargs), daemon=True)
    th.start()

    pieces: List[str] = []
    try:
        for text in streamer:
            if first_tok_t is None:
                first_tok_t = time.perf_counter()
            pieces.append(text)
    finally:
        th.join()

    t1 = time.perf_counter()
    gen_text = "".join(pieces).strip()

    # token count based on tokenizer (strings != tokens)
    try:
        gen_ids = tokenizer.encode(gen_text, add_special_tokens=False)
        num_toks = len(gen_ids)
    except Exception:
        num_toks = max(1, len(pieces))

    e2e = t1 - t0
    ttft = (first_tok_t - t0) if first_tok_t else 0.0
    throughput = (num_toks / e2e) if e2e > 0 else 0.0
    tpot = (e2e / num_toks) if num_toks > 0 else 0.0

    return {
        "answer": gen_text,
        "ttft": ttft,
        "e2e_latency": e2e,
        "throughput": throughput,
        "tpot": tpot,
    }


# ------------------------------ main ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser("Full KV Reuse (corrected)")
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--input", type=str, default="inputs/musique_s.json")
    ap.add_argument("--output", type=str, default="results/full_kv_reuse_results")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--retrieval_json", type=str, default="retrieval_topk.json")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cfg = load_config(args.config)
    samples = load_samples(args.input)

    model_name = cfg.get("model", {}).get("model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
    device_pref = cfg.get("model", {}).get("device", "cuda:0")
    top_k = cfg.get("retrieval", {}).get("top_k", args.top_k)
    max_new_tokens = cfg.get("generation", {}).get("max_new_tokens", 128)

    # model + tok
    tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(model_name))
    # Prefer a single explicit device unless you know you want sharding
    device = torch.device(device_pref) if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None,  # keep on one device for simpler PKV plumbing
    ).to(device).eval()

    # ------ retrieval cache or run ------
    retrieval_json_path = os.path.join(args.output, args.retrieval_json)
    if os.path.exists(retrieval_json_path):
        with open(retrieval_json_path, "r") as f:
            retrieval_data = json.load(f)
        by_id = {str(it.get("id", i)): it for i, it in enumerate(retrieval_data)}
        for i, smp in enumerate(samples):
            sid = str(smp.get("id", i))
            if sid in by_id:
                smp.update(by_id[sid])
    else:
        run_retrieval(samples, cfg, top_k)
        to_save = []
        for smp in samples:
            to_save.append({
                "id": smp.get("id"),
                "retrieved_indices": smp.get("retrieved_indices", []),
                "retrieved_scores": smp.get("retrieved_scores", []),
            })
        with open(retrieval_json_path, "w") as f:
            json.dump(to_save, f, indent=2)

    # ------ per-sample run ------
    results: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples):
        sid = sample.get("id", str(i))
        try:
            pkv, cached_len = prefill_topk_kv(model, tokenizer, device, sample, top_k)
            out = decode_with_past(model, tokenizer, device, pkv, cached_len, sample, max_new_tokens)
            out.update({"sample_id": sid})
        except Exception as e:
            out = {
                "sample_id": sid,
                "answer": f"Error: {e}",
                "ttft": 0.0,
                "e2e_latency": 0.0,
                "throughput": 0.0,
                "tpot": 0.0,
            }
        results.append(out)

    # ------ summary row ------
    if results:
        n = len(results)
        avg = lambda k: sum(r.get(k, 0.0) for r in results) / max(1, n)
        results.append({
            "sample_id": "average",
            "answer": "average_metrics",
            "ttft": avg("ttft"),
            "e2e_latency": avg("e2e_latency"),
            "throughput": avg("throughput"),
            "tpot": avg("tpot"),
        })

    out_path = os.path.join(args.output, "results_full_kv_reuse.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done: {len(results)-1} samples → {out_path}")


if __name__ == "__main__":
    main()
