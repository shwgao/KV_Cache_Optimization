#!/usr/bin/env python3

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Any, Optional

# ----------------------- Prompt Structure -----------------------

PREFIX_PROMPT = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"

QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

# ----------------------- Helper Functions -----------------------

def extract_texts(sample: Dict[str, Any]) -> List[Tuple[int, str]]:
    """Extract text chunks from sample data (same logic as before)"""
    pairs = []
    
    if isinstance(sample.get("ctxs"), list):
        for i, ch in enumerate(sample["ctxs"]):
            title = (ch.get("title") or "").strip()
            text = (ch.get("text") or "").strip()
            full = f"{title}\n{text}".strip() if title else text
            if full:
                pairs.append((i, full))
    
    return pairs

def build_qa_prompt(sample: Dict[str, Any], query_prompt: str) -> Tuple[List[str], str]:
    """Build QA prompt structure similar to debug.py logic"""
    # Extract passages
    texts = extract_texts(sample)
    doc_prompts = [text for _, text in texts]
    
    # Build question prompt
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
    Build input sequence: [prefix_prompt] + [doc1] + [doc2] + ... + [question]
    """
    
    # Tokenize each component (remove BOS tokens [1:])
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]
    
    # Mistral chat format tokens
    s_start_full = [733, 16289, 28793] + tokenizer.encode(prefix_prompt)[1:]
    s_start = []  # Empty for document chunks
    s_end = [733, 28748, 16289, 28793]
    s_start_1_len = len(s_start) + 1
    
    # Build chunk structure  
    doc_chunk_ids = [s_start + chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start + q_ids + s_end]
    
    # Concatenate avoiding duplicate tokens  
    input_ids = []
    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = doc_chunk_ids[i]  # Full first chunk (prefix)
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]  # Remove overlapping tokens
        input_ids += temp_ids
    
    return input_ids

def build_chunk_sequence(
    chunk_text: str,
    tokenizer: Any,
    prefix_prompt: str = PREFIX_PROMPT
) -> List[int]:
    """
    Build sequence for individual chunk KV cache generation
    Format: [prefix_prompt] + [chunk_text]
    """
    
    # Tokenize components (remove BOS)
    chunk_ids = tokenizer.encode(chunk_text)[1:]
    
    # Mistral chat format
    s_start_full = [733, 16289, 28793] + tokenizer.encode(prefix_prompt)[1:]
    s_start = []
    
    # Build sequence: prefix + chunk
    sequence = s_start_full + s_start + chunk_ids
    
    return sequence

# ----------------------- KV Cache Builder -----------------------

def build_chunk_kv_caches(
    samples: List[Dict[str, Any]],
    model_id: str,
    top_k: int = 5,
    device: str = "cuda:0",
    provided_tokenizer: Optional[Any] = None,
    provided_model: Optional[Any] = None
) -> Dict[str, Dict[int, Any]]:
    """
    Build KV caches:
    - Same prompt structure (prefix + passages + query)
    - Same tokenization logic (remove BOS, handle Mistral tokens)
    - Same KV cache generation (prefill each chunk)
    """
    
    # Setup model and tokenizer  
    tokenizer = provided_tokenizer if provided_tokenizer else AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = provided_model if provided_model else AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(device).eval()
    
    # Storage
    gpu_chunks = {}
    cpu_chunks = {}
    
    # Process samples
    for sample in samples:
        texts = extract_texts(sample)
        if not texts:
            continue
            
        # Get top-k indices
        top_indices = sample.get("retrieved_indices", [])
        top_set = set(int(i) for i in top_indices[:top_k]) if top_indices else set()
        
        for idx, chunk_text in texts:
            if idx in top_set and len(gpu_chunks) < top_k:
                # Build sequence for this chunk
                input_ids = build_chunk_sequence(chunk_text, tokenizer)
                current_input = torch.tensor([input_ids], device=device)
                
                # Generate KV cache
                with torch.inference_mode():
                    outputs = model(
                        current_input,
                        use_cache=True,
                        return_dict=True
                    )
                    # Store past_key_value
                    gpu_chunks[idx] = outputs.past_key_values
            else:
                # Store on CPU: just the text
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
    Build complete sequence KV cache exactly like debug.py:
    Returns (input_ids, past_key_values) for the full context
    """
    
    # Setup
    tokenizer = provided_tokenizer if provided_tokenizer else AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = provided_model if provided_model else AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(device).eval()
    
    # Build prompt structure  
    doc_prompts, q_prompt = build_qa_prompt(sample, QUERY_PROMPT)
    
    # Build complete sequence
    input_ids = build_sequence(doc_prompts, q_prompt, tokenizer)
    
    # Generate KV cache for full sequence
    current_input = torch.tensor([input_ids], device=device)
    
    with torch.inference_mode():
        outputs = model(
            current_input,
            use_cache=True,
            return_dict=True
        )
        past_key_values = outputs.past_key_values
    
    return input_ids, past_key_values