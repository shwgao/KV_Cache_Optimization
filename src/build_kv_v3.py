#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------- Prompt Structure -----------------------

PREFIX_PROMPT = (
    "You will be asked a question after reading several passages. "
    "Please directly answer the question based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
)

QUERY_PROMPT = (
    "\n\nAnswer the question directly based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
)

# ----------------------- Helper Functions -----------------------

def extract_texts(sample: Dict[str, Any]) -> List[Tuple[int, str]]:
    """Extract text chunks from sample data."""
    pairs: List[Tuple[int, str]] = []
    ctxs = sample.get("ctxs")
    if isinstance(ctxs, list):
        for i, ch in enumerate(ctxs):
            title = (ch.get("title") or "").strip()
            text = (ch.get("text") or "").strip()
            full = f"{title}\n{text}".strip() if title else text
            if full:
                pairs.append((i, full))
    return pairs

def build_qa_prompt(sample: Dict[str, Any], query_prompt: str) -> Tuple[List[str], str]:
    """Prepare passages and the question prompt."""
    texts = extract_texts(sample)
    doc_prompts = [text for _, text in texts]
    question = sample.get("question", "")
    q_prompt = query_prompt + question
    return doc_prompts, q_prompt

def build_sequence(
    doc_prompts: List[str],
    q_prompt: str,
    tokenizer: Any,
    prefix_prompt: str = PREFIX_PROMPT
) -> List[int]:
    """
    Build input sequence: [prefix_prompt] + [doc1] + ... + [question]
    Notes:
      - We keep your custom Mistral token markers; components are encoded WITHOUT special tokens.
    """
    # Tokenize without special tokens to avoid extra BOS/EOS
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)

    doc_chunk_ids = [enc(doc) for doc in doc_prompts]
    q_ids = enc(q_prompt)

    # Mistral chat-ish markers (as in your code)
    s_start_full = [733, 16289, 28793] + enc(prefix_prompt)
    s_start = []  # none for doc chunks
    s_end = [733, 28748, 16289, 28793]

    # Assemble chunks
    chunks: List[List[int]] = [s_start_full] + [s_start + ids for ids in doc_chunk_ids] + [s_start + q_ids + s_end]

    # Concatenate
    input_ids: List[int] = []
    for i, ids in enumerate(chunks):
        if i == 0:
            input_ids += ids
        else:
            input_ids += ids  # no overlap trimming needed when add_special_tokens=False
    return input_ids

def build_chunk_sequence(
    chunk_text: str,
    tokenizer: Any,
    prefix_prompt: str = PREFIX_PROMPT
) -> List[int]:
    """
    Build sequence for individual chunk KV cache generation:
      [prefix_prompt] + [chunk_text]
    """
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    chunk_ids = enc(chunk_text)
    s_start_full = [733, 16289, 28793] + enc(prefix_prompt)
    sequence = s_start_full + chunk_ids
    return sequence

# ----------------------- KV Cache Builder -----------------------

def _ensure_legacy_kv(kv: Any) -> Any:
    """HF sometimes returns a DynamicCache; convert to legacy tuple[(K,V), ...] if available."""
    try:
        if hasattr(kv, "to_legacy_cache"):
            return kv.to_legacy_cache()
    except Exception:
        pass
    return kv

def build_chunk_kv_caches(
    samples: List[Dict[str, Any]],
    model_id: str,
    top_k: int = 5,
    device: str = "cuda:0",
    provided_tokenizer: Optional[Any] = None,
    provided_model: Optional[Any] = None
) -> Dict[str, Dict[int, Any]]:
    """
    Build per-chunk KV caches.
      - Top-K chunks go to GPU as KV
      - The rest are kept as raw text for CPU-side precompute by the scheduler
    """
    # Tokenizer
    tokenizer = provided_tokenizer or AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    if provided_model is not None:
        model = provided_model.eval()  # do NOT .to(device) if caller already device-mapped
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()

    gpu_chunks: Dict[int, Any] = {}
    cpu_chunks: Dict[int, str] = {}

    for sample in samples:
        texts = extract_texts(sample)
        if not texts:
            continue

        # Prefer explicit retrieved indices if present, else fall back to first top_k
        ret_idx = sample.get("retrieved_indices")
        if isinstance(ret_idx, list) and len(ret_idx) > 0:
            top_set = set(int(i) for i in ret_idx[:top_k])
        else:
            top_set = set(range(min(top_k, len(texts))))

        for idx, chunk_text in texts:
            place_in_gpu = (idx in top_set) or (not top_set and len(gpu_chunks) < top_k)

            if place_in_gpu and len(gpu_chunks) < top_k:
                input_ids = build_chunk_sequence(chunk_text, tokenizer)
                current_input = torch.tensor([input_ids], device=device if provided_model is None else model.device)
                with torch.inference_mode():
                    outputs = model(current_input, use_cache=True, return_dict=True)
                    kv = _ensure_legacy_kv(outputs.past_key_values)
                    gpu_chunks[idx] = kv
            else:
                cpu_chunks[idx] = chunk_text

    return {"gpu_chunks": gpu_chunks, "cpu_chunks": cpu_chunks}

def build_full_sequence_kv_cache(
    sample: Dict[str, Any],
    model_id: str,
    device: str = "cuda:0",
    provided_tokenizer: Optional[Any] = None,
    provided_model: Optional[Any] = None
) -> Tuple[List[int], Any]:
    """
    Build KV cache for the full concatenated sequence.
    Returns (input_ids, legacy_past_key_values)
    """
    tokenizer = provided_tokenizer or AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if provided_model is not None:
        model = provided_model.eval()
        model_device = model.device
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
        model_device = device

    doc_prompts, q_prompt = build_qa_prompt(sample, QUERY_PROMPT)
    input_ids = build_sequence(doc_prompts, q_prompt, tokenizer)
    current_input = torch.tensor([input_ids], device=model_device)

    with torch.inference_mode():
        outputs = model(current_input, use_cache=True, return_dict=True)
        past_key_values = _ensure_legacy_kv(outputs.past_key_values)

    return input_ids, past_key_values
