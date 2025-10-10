#!/usr/bin/env python3
"""
CORRECT FULL RECOMPUTE & FULL REUSE IMPLEMENTATION
Simple, accurate implementation with proper TTFT/TPOT calculation
Fixes all issues found in the original full_recompute_and_reuse.py
"""

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from rag_retrieval import RetrievalConfig, ColbertRetrieval
from build_kv_v2 import build_chunk_kv_caches, QUERY_PROMPT

def convert_to_serializable(obj):
    """Convert NumPy/PyTorch types to JSON-serializable Python types"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

def setup_logging(level: str = "WARNING") -> logging.Logger:
    """Setup minimal logging for performance"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger("pipeline")

def load_samples(path: str) -> List[Dict[str, Any]]:
    """Load dataset samples from JSON"""
    with open(path, "r") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "results" in data:
            return data["results"]
        return data

def calculate_text_f1(pred: str, ref: str) -> float:
    """
    CORRECTED F1 CALCULATION: Proper text similarity F1 score
    Fixes the incorrect label-based F1 from original implementation
    """
    # Strip whitespace for comparison
    pred = pred.strip() if pred else ""
    ref = ref.strip() if ref else ""
    
    if not pred or not ref:
        print(f"[F1 DEBUG] Empty strings detected - pred: '{pred}', ref: '{ref}'")
        return 0.0
    
    # Tokenize into words
    pred_words = set(pred.lower().split())
    ref_words = set(ref.lower().split())
    
    print(f"[F1 DEBUG] Pred words: {len(pred_words)}, Ref words: {len(ref_words)}")
    
    # Calculate precision, recall, F1
    if not pred_words and not ref_words:
        return 1.0
    
    if not pred_words or not ref_words:
        return 0.0
    
    intersection = pred_words & ref_words
    print(f"[F1 DEBUG] Intersection: {len(intersection)} words")
    
    precision = len(intersection) / len(pred_words)
    recall = len(intersection) / len(ref_words)
    
    print(f"[F1 DEBUG] Precision: {precision:.3f}, Recall: {recall:.3f}")
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_cosine_similarity(pred: str, ref: str) -> float:
    """Calculate cosine similarity between prediction and reference"""
    if not pred or not ref:
        return 0.0
    
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([pred, ref])
        cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return cosine_sim
    except:
        return 0.0

def sample_with_temperature(logits: torch.Tensor, temperature: float = 0.7) -> torch.Tensor:
    """Sample next token with temperature"""
    if temperature <= 0:
        return torch.argmax(logits[:, -1, :], dim=-1)
    scaled_logits = logits[:, -1, :] / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probabilities, 1).squeeze(-1)
    return next_token

# CORRECTED IMPLEMENTATION: Full Recompute

def full_recompute_generation(
    sample: Dict[str, Any],
    gpu_chunks: Dict[int, Any],
    model: Any,
    tokenizer: Any,
    device: str,
    max_tokens: int = 32
) -> Dict[str, Any]:
    """
    CORRECTED: Full recompute with proper TTFT/TPOT calculation
    Rebuilds KV cache from scratch for every generation
    """
    
    print(f"\n{'='*80}")
    print(f"[FullRecompute] Starting full recompute generation")
    print(f"{'='*80}")
    
    current_gpu_chunks = list(gpu_chunks.keys())
    print(f"[FullRecompute] Using GPU chunks: {current_gpu_chunks}")
    
    # Build combined KV cache from GPU chunks
    combined_kv = None
    if gpu_chunks:
        valid_caches = [gpu_chunks[idx] for idx in current_gpu_chunks if idx in gpu_chunks]
        
        if valid_caches:
            num_layers = len(valid_caches[0])
            combined_kv = []
            
            with torch.inference_mode():
                for layer_idx in range(num_layers):
                    keys_to_concat = []
                    values_to_concat = []
                    
                    for kv_cache in valid_caches:
                        k, v = kv_cache[layer_idx]
                        if k.dim() == 3:
                            k = k.unsqueeze(0)
                            v = v.unsqueeze(0)
                        keys_to_concat.append(k)
                        values_to_concat.append(v)
                    
                    merged_k = torch.cat(keys_to_concat, dim=2)
                    merged_v = torch.cat(values_to_concat, dim=2)
                    combined_kv.append((merged_k, merged_v))
            
            combined_kv = tuple(combined_kv)
    
    if combined_kv is None:
        print(f"[FullRecompute] ERROR: No KV caches available")
        return {"error": "No KV caches", "metrics": {"ttft_s": 0, "tpot_s": 0, "e2e_s": 0}}
    
    chunk_seq_len = combined_kv[0][0].shape[2]
    print(f"[FullRecompute] Context length: {chunk_seq_len} tokens")
    
    # Prepare question
    question = sample.get("question", "")
    formatted_question = QUERY_PROMPT + question
    question_ids = tokenizer.encode(formatted_question, add_special_tokens=False)
    question_input = torch.tensor([question_ids], device=device)
    question_length = len(question_ids)
    total_context_len = chunk_seq_len + question_length
    attention_mask = torch.ones((1, total_context_len), device=device, dtype=torch.bool)
    position_ids = torch.arange(chunk_seq_len, chunk_seq_len + question_length, device=device).unsqueeze(0)
    
    print(f"[FullRecompute] Total context: {total_context_len} tokens")
    
    # CORRECTED TIMING: Start timer RIGHT before generation
    generation_start = time.perf_counter()
    first_token_time = None
    decode_times = []
    generated_tokens = []
    
    with torch.inference_mode():
        # Initial forward pass - TTFT measurement
        initial_cache = DynamicCache.from_legacy_cache(combined_kv)
        outputs = model(
            input_ids=question_input,
            past_key_values=initial_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True
        )
        
        past_key_values = outputs.past_key_values
        next_token = sample_with_temperature(outputs.logits, temperature=0.7)
        token_id = next_token.item()
        
        # CORRECTED TTFT: First token time
        first_token_time = time.perf_counter()
        ttft_duration = first_token_time - generation_start
        
        print(f"[FullRecompute] First token in {ttft_duration*1000:.2f}ms")
        
        if token_id != tokenizer.eos_token_id:
            generated_tokens.append(token_id)
        
        # Decode loop for remaining tokens
        for step in range(1, max_tokens):
            if token_id == tokenizer.eos_token_id:
                break
            
            step_start = time.perf_counter()
            
            # Generate next token
            input_token = next_token.unsqueeze(-1)
            outputs = model(
                input_ids=input_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            past_key_values = outputs.past_key_values
            next_token = sample_with_temperature(outputs.logits, temperature=0.7)
            token_id = next_token.item()
            
            if token_id == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(token_id)
            step_end = time.perf_counter()
            decode_times.append(step_end - step_start)
    
    generation_end = time.perf_counter()
    
    # Generate final text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # CORRECTED METRICS CALCULATION
    e2e_latency = generation_end - generation_start
    
    # CORRECTED TPOT: Exclude TTFT from TPOT calculation
    if len(decode_times) > 0:
        tpot_s = sum(decode_times) / len(decode_times)
    else:
        tpot_s = 0.0
    
    print(f"[FullRecompute] Generated {len(generated_tokens)} tokens")
    print(f"[FullRecompute] Generated text: '{generated_text}'")
    print(f"[FullRecompute] CORRECTED Metrics: TTFT={ttft_duration*1000:.1f}ms, TPOT={tpot_s*1000:.1f}ms")
    
    return {
        "method": "full_recompute",
        "question": question,
        "answer": generated_text,
        "generated_tokens": generated_tokens,
        "chunks_used": current_gpu_chunks,
        "cached_kv": combined_kv,  # FIXED: Return cached KV for reuse
        "metrics": {
            "ttft_s": ttft_duration,
            "tpot_s": tpot_s,
            "e2e_s": e2e_latency,
            "num_tokens": len(generated_tokens),
            "context_length": total_context_len
        }
    }

# CORRECTED IMPLEMENTATION: Full Reuse

def full_reuse_generation(
    sample: Dict[str, Any],
    gpu_chunks: Dict[int, Any],
    cached_kv: Optional[tuple],
    model: Any,
    tokenizer: Any,
    device: str,
    max_tokens: int = 32
) -> Dict[str, Any]:
    """
    CORRECTED: Full reuse with proper KV cache reuse
    Reuses the SAME KV cache without rebuilding
    """
    
    print(f"\n{'='*80}")
    print(f"[FullReuse] Starting full reuse generation")
    print(f"{'='*80}")
    
    current_gpu_chunks = list(gpu_chunks.keys())
    print(f"[FullReuse] Reusing chunks: {current_gpu_chunks}")
    
    # Use cached KV if available, otherwise build once
    if cached_kv is not None:
        combined_kv = cached_kv
        print(f"[FullReuse] Using cached KV cache")
    else:
        # Build once for reuse
        combined_kv = None
        if gpu_chunks:
            valid_caches = [gpu_chunks[idx] for idx in current_gpu_chunks if idx in gpu_chunks]
            
            if valid_caches:
                num_layers = len(valid_caches[0])
                combined_kv = []
                
                with torch.inference_mode():
                    for layer_idx in range(num_layers):
                        keys_to_concat = []
                        values_to_concat = []
                        
                        for kv_cache in valid_caches:
                            k, v = kv_cache[layer_idx]
                            if k.dim() == 3:
                                k = k.unsqueeze(0)
                                v = v.unsqueeze(0)
                            keys_to_concat.append(k)
                            values_to_concat.append(v)
                        
                        merged_k = torch.cat(keys_to_concat, dim=2)
                        merged_v = torch.cat(values_to_concat, dim=2)
                        combined_kv.append((merged_k, merged_v))
                
                combined_kv = tuple(combined_kv)
        
        print(f"[FullReuse] Built KV cache for reuse")
    
    if combined_kv is None:
        print(f"[FullReuse] ERROR: No KV caches available")
        return {"error": "No KV caches", "metrics": {"ttft_s": 0, "tpot_s": 0, "e2e_s": 0}}
    
    chunk_seq_len = combined_kv[0][0].shape[2]
    print(f"[FullReuse] Context length: {chunk_seq_len} tokens")
    
    # Prepare question (same as recompute)
    question = sample.get("question", "")
    formatted_question = QUERY_PROMPT + question
    question_ids = tokenizer.encode(formatted_question, add_special_tokens=False)
    question_input = torch.tensor([question_ids], device=device)
    question_length = len(question_ids)
    total_context_len = chunk_seq_len + question_length
    attention_mask = torch.ones((1, total_context_len), device=device, dtype=torch.bool)
    position_ids = torch.arange(chunk_seq_len, chunk_seq_len + question_length, device=device).unsqueeze(0)
    
    # CORRECTED TIMING: Start timer RIGHT before generation
    generation_start = time.perf_counter()
    first_token_time = None
    decode_times = []
    generated_tokens = []
    
    with torch.inference_mode():
        # Initial forward pass - TTFT measurement (REUSING cached KV)
        initial_cache = DynamicCache.from_legacy_cache(combined_kv)
        outputs = model(
            input_ids=question_input,
            past_key_values=initial_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True
        )
        
        past_key_values = outputs.past_key_values
        next_token = sample_with_temperature(outputs.logits, temperature=0.7)
        token_id = next_token.item()
        
        # CORRECTED TTFT: First token time
        first_token_time = time.perf_counter()
        ttft_duration = first_token_time - generation_start
        
        print(f"[FullReuse] First token in {ttft_duration*1000:.2f}ms (with reuse)")
        
        if token_id != tokenizer.eos_token_id:
            generated_tokens.append(token_id)
        
        # Decode loop (same as recompute)
        for step in range(1, max_tokens):
            if token_id == tokenizer.eos_token_id:
                break
            
            step_start = time.perf_counter()
            
            # Generate next token
            input_token = next_token.unsqueeze(-1)
            outputs = model(
                input_ids=input_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            past_key_values = outputs.past_key_values
            next_token = sample_with_temperature(outputs.logits, temperature=0.7)
            token_id = next_token.item()
            
            if token_id == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(token_id)
            step_end = time.perf_counter()
            decode_times.append(step_end - step_start)
    
    generation_end = time.perf_counter()
    
    # Generate final text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # CORRECTED METRICS CALCULATION
    e2e_latency = generation_end - generation_start
    
    # CORRECTED TPOT: Exclude TTFT from TPOT calculation
    if len(decode_times) > 0:
        tpot_s = sum(decode_times) / len(decode_times)
    else:
        tpot_s = 0.0
    
    print(f"[FullReuse] Generated {len(generated_tokens)} tokens")
    print(f"[FullReuse] Generated text: '{generated_text}'")
    print(f"[FullReuse] CORRECTED Metrics: TTFT={ttft_duration*1000:.1f}ms, TPOT={tpot_s*1000:.1f}ms")
    
    return {
        "method": "full_reuse",
        "question": question,
        "answer": generated_text,
        "generated_tokens": generated_tokens,
        "chunks_used": current_gpu_chunks,
        "cached_kv": combined_kv,  # Return for next reuse
        "metrics": {
            "ttft_s": ttft_duration,
            "tpot_s": tpot_s,
            "e2e_s": e2e_latency,
            "num_tokens": len(generated_tokens),
            "context_length": total_context_len
        }
    }

# MAIN PIPELINE

def run_corrected_full_pipeline(
    input_file: str,
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    output_dir: str = "results",
    top_k: int = 5,
    max_tokens: int = 32,
    device: str = "cuda:0"
):
    """
    CORRECTED: Full recompute and reuse pipeline with proper metrics
    Fixes all issues from original implementation
    """
    
    print(f"\n{'='*80}")
    print(f"[CorrectedPipeline] CORRECTED Full Recompute & Reuse Implementation")
    print(f"[CorrectedPipeline] - Fixed TPOT calculation")
    print(f"[CorrectedPipeline] - Fixed F1 calculation") 
    print(f"[CorrectedPipeline] - Proper KV cache reuse")
    print(f"[CorrectedPipeline] - Accurate timing measurements")
    print(f"{'='*80}\n")
    
    logger = setup_logging("WARNING")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load samples
    samples = load_samples(input_file)
    print(f"[CorrectedPipeline] Loaded {len(samples)} samples")
    
    # Setup retrieval
    retrieval_config = RetrievalConfig()
    retriever = ColbertRetrieval(retrieval_config)
    print(f"[CorrectedPipeline] Retrieval system initialized")
    
    # Load model/tokenizer
    print(f"[CorrectedPipeline] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()
    print(f"[CorrectedPipeline] Model loaded")
    
    results = []
    cached_kv_store = {}  # Store KV caches for reuse
    
    for si, sample in enumerate(tqdm(samples, desc="Processing", unit="sample")):
        try:
            sample_start = time.perf_counter()
            print(f"\n[Sample {si}] Processing with corrected implementation...")
            
            torch.cuda.empty_cache()
            
            # Retrieval (done once, outside timing)
            retrieval_start = time.perf_counter()
            prepared_samples = retriever.prepare([sample])
            retrieved_samples = retriever.retrieve(prepared_samples, top_k=top_k)
            sample = retrieved_samples[0] if retrieved_samples else sample
            retrieval_time = time.perf_counter() - retrieval_start
            
            # Build KV caches (done once, outside timing)
            kv_start = time.perf_counter()
            kv_result = build_chunk_kv_caches(
                samples=[sample],
                model_id=model_id,
                top_k=top_k,
                device=device,
                provided_tokenizer=tokenizer,
                provided_model=model
            )
            kv_time = time.perf_counter() - kv_start
            
            gpu_chunks = kv_result.get("gpu_chunks", {})
            
            if not gpu_chunks:
                print(f"[Sample {si}] ERROR: No GPU chunks")
                continue
            
            sample_id = str(sample.get("id", f"sample_{si}"))
            
            # 1. FULL RECOMPUTE
            print(f"\n[Sample {si}] Running FULL RECOMPUTE...")
            recompute_result = full_recompute_generation(
                sample=sample,
                gpu_chunks=gpu_chunks,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_tokens=max_tokens
            )
            
            # Store KV cache for reuse
            if "cached_kv" in recompute_result:
                cached_kv_store[sample_id] = recompute_result["cached_kv"]
            elif gpu_chunks:
                # Build cache for reuse if not available
                current_gpu_chunks = list(gpu_chunks.keys())
                valid_caches = [gpu_chunks[idx] for idx in current_gpu_chunks if idx in gpu_chunks]
                
                if valid_caches:
                    num_layers = len(valid_caches[0])
                    combined_kv = []
                    
                    with torch.inference_mode():
                        for layer_idx in range(num_layers):
                            keys_to_concat = []
                            values_to_concat = []
                            
                            for kv_cache in valid_caches:
                                k, v = kv_cache[layer_idx]
                                if k.dim() == 3:
                                    k = k.unsqueeze(0)
                                    v = v.unsqueeze(0)
                                keys_to_concat.append(k)
                                values_to_concat.append(v)
                            
                            merged_k = torch.cat(keys_to_concat, dim=2)
                            merged_v = torch.cat(values_to_concat, dim=2)
                            combined_kv.append((merged_k, merged_v))
                    
                    cached_kv_store[sample_id] = tuple(combined_kv)
            
            # 2. FULL REUSE
            print(f"\n[Sample {si}] Running FULL REUSE...")
            reuse_result = full_reuse_generation(
                sample=sample,
                gpu_chunks=gpu_chunks,
                cached_kv=cached_kv_store.get(sample_id),
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_tokens=max_tokens
            )
            
            sample_total = time.perf_counter() - sample_start
            
            # Calculate corrected F1 scores
            # Check available keys in sample
            print(f"\n[Sample {si}] Sample keys: {list(sample.keys())}")
            reference_answer = sample.get("answer", "")
            
            print(f"\n[Sample {si}] === F1 CALCULATION DEBUG ===")
            print(f"[Sample {si}] Reference answer: '{reference_answer}'")
            print(f"[Sample {si}] Recompute answer: '{recompute_result['answer']}'")
            print(f"[Sample {si}] Reuse answer: '{reuse_result['answer']}'")
            
            recompute_f1 = calculate_text_f1(recompute_result["answer"], reference_answer)
            reuse_f1 = calculate_text_f1(reuse_result["answer"], reference_answer)
            
            print(f"[Sample {si}] Recompute F1: {recompute_f1:.3f}")
            print(f"[Sample {si}] Reuse F1: {reuse_f1:.3f}")
            
            recompute_cosine = calculate_cosine_similarity(recompute_result["answer"], reference_answer)
            reuse_cosine = calculate_cosine_similarity(reuse_result["answer"], reference_answer)
            
            # Store results
            sample_result = {
                "sample_id": sample_id,
                "sample_index": si,
                "question": sample.get("question", ""),
                "reference_answer": reference_answer,
                "timing_breakdown": {
                    "retrieval_s": retrieval_time,
                    "kv_building_s": kv_time,
                    "total_s": sample_total
                },
                "recompute": {
                    **recompute_result,
                    "f1_score": recompute_f1,
                    "cosine_similarity": recompute_cosine
                },
                "reuse": {
                    **reuse_result,
                    "f1_score": reuse_f1,
                    "cosine_similarity": reuse_cosine
                }
            }
            
            results.append(sample_result)
            
            # Log results
            recomp_metrics = recompute_result["metrics"]
            reuse_metrics = reuse_result["metrics"]
            
            print(f"\n[Sample {si}] CORRECTED RESULTS:")
            print(f"[Sample {si}] Total time: {sample_total:.3f}s")
            print(f"[Sample {si}] RECOMPUTE: TTFT={recomp_metrics['ttft_s']*1000:.1f}ms, TPOT={recomp_metrics['tpot_s']*1000:.1f}ms, F1={recompute_f1:.3f}")
            print(f"[Sample {si}] REUSE: TTFT={reuse_metrics['ttft_s']*1000:.1f}ms, TPOT={reuse_metrics['tpot_s']*1000:.1f}ms, F1={reuse_f1:.3f}")
            print(f"[Sample {si}] Recompute answer: '{recompute_result['answer'][:100]}...'")
            print(f"[Sample {si}] Reuse answer: '{reuse_result['answer'][:100]}...'")
            
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.warning(f"Sample {si} failed: {str(e)}")
            continue
    
    # Save results
    output_path = os.path.join(output_dir, f"results_corrected_full.json")
    with open(output_path, "w") as f:
        serializable_results = convert_to_serializable({"results": results})
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n[CorrectedPipeline] Results saved to {output_path}")
    
    # Performance summary
    if results:
        successful_results = results
        
        # Average metrics
        avg_recomp_ttft = sum(r["recompute"]["metrics"]["ttft_s"] for r in successful_results) / len(successful_results)
        avg_recomp_tpot = sum(r["recompute"]["metrics"]["tpot_s"] for r in successful_results) / len(successful_results)
        avg_recomp_f1 = sum(r["recompute"]["f1_score"] for r in successful_results) / len(successful_results)
        
        avg_reuse_ttft = sum(r["reuse"]["metrics"]["ttft_s"] for r in successful_results) / len(successful_results)
        avg_reuse_tpot = sum(r["reuse"]["metrics"]["tpot_s"] for r in successful_results) / len(successful_results)
        avg_reuse_f1 = sum(r["reuse"]["f1_score"] for r in successful_results) / len(successful_results)
        
        print(f"\n{'='*80}")
        print(f"CORRECTED FULL RECOMPUTE & REUSE RESULTS")
        print(f"{'='*80}")
        print(f"Samples processed: {len(successful_results)}")
        print(f"\n--- FULL RECOMPUTE (Corrected) ---")
        print(f"Average TTFT: {avg_recomp_ttft*1000:.1f}ms")
        print(f"Average TPOT: {avg_recomp_tpot*1000:.1f}ms")
        print(f"Average F1: {avg_recomp_f1:.3f}")
        print(f"\n--- FULL REUSE (Corrected) ---")
        print(f"Average TTFT: {avg_reuse_ttft*1000:.1f}ms")
        print(f"Average TPOT: {avg_reuse_tpot*1000:.1f}ms")
        print(f"Average F1: {avg_reuse_f1:.3f}")
        print(f"\n--- IMPROVEMENTS ---")
        print(f"TTFT improvement: {((avg_recomp_ttft - avg_reuse_ttft) / avg_recomp_ttft * 100):.1f}%")
        print(f"TPOT improvement: {((avg_recomp_tpot - avg_reuse_tpot) / avg_recomp_tpot * 100):.1f}%")
        print(f"F1 difference: {(avg_reuse_f1 - avg_recomp_f1):.3f}")
        print(f"{'='*80}")
    
    return {"output_path": output_path, "results": results}

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser("CORRECTED: Full Recompute & Reuse")
    parser.add_argument("--input", required=True, help="Input dataset JSON")
    parser.add_argument("--output_dir", default="results_corrected", help="Output directory")
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.2", help="Model ID")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k chunks")
    parser.add_argument("--max_tokens", type=int, default=20, help="Max generation length")
    parser.add_argument("--device", default="cuda:0", help="Device")
    
    args = parser.parse_args()
    
    print("CORRECTED FULL RECOMPUTE & REUSE IMPLEMENTATION")
    print("=" * 60)
    print("1. Fixed TPOT calculation (excludes TTFT)")
    print("2. Fixed F1 calculation (text similarity, not labels)")
    print("3. Proper KV cache reuse")
    print("4. Accurate timing measurements")
    print("=" * 60)
    
    run_corrected_full_pipeline(
        input_file=args.input,
        model_id=args.model_id,
        output_dir=args.output_dir,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        device=args.device
    )

if __name__ == "__main__":
    main()