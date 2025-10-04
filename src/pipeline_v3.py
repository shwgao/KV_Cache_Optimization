#!/usr/bin/env python3

"""
HIGH-PERFORMANCE OPTIMIZED Pipeline for Minimal Latency
Addresses all critical performance bottlenecks for paper-worthy results
"""

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import numpy as np

# Local imports
from rag_retrieval import RetrievalConfig, ColbertRetrieval
from build_kv_v2 import build_chunk_kv_caches, QUERY_PROMPT
from scheduler_v3 import HighPerformanceBanditScheduler, FastKVCacheManager

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
    elif hasattr(obj, 'item'):  # PyTorch scalars
        return obj.item()
    else:
        return obj

def setup_logging(level: str = "WARNING") -> logging.Logger:  # REDUCED logging
    """Setup logging and return logger"""
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

def compute_chunk_boundaries_from_kv_lengths(
    gpu_chunks: Dict[int, Any], selected_chunks: List[int]
) -> List[tuple]:
    """
    Compute (start, end) token boundaries for each selected chunk when their
    per-chunk KV caches are concatenated along the sequence dimension.
    """
    boundaries: List[tuple] = []
    current = 0
    for chunk_idx in selected_chunks:
        if chunk_idx not in gpu_chunks:
            continue
        kv_cache = gpu_chunks[chunk_idx]
        try:
            key_tensor = kv_cache[0][0]
            chunk_len = int(key_tensor.shape[2])
        except Exception:
            chunk_len = 0
        start = current
        end = current + chunk_len
        boundaries.append((start, end))
        current = end
    return boundaries

def create_sparse_kv_cache_with_ratio(
    full_kv_cache, 
    chunk_boundaries: List[tuple[int, int]], 
    sparsity_ratio: float = 0.6,
    priority_chunks: List[int] = None) -> Any:
    """Create sparse KV cache by keeping only sparsity_ratio of tokens"""
    if not chunk_boundaries or sparsity_ratio >= 1.0:
        return full_kv_cache
    
    # Calculate total sequence length
    total_tokens = sum(end - start for start, end in chunk_boundaries)
    tokens_to_keep = int(total_tokens * sparsity_ratio)
    
    if tokens_to_keep >= total_tokens:
        return full_kv_cache
    
    # Prioritize chunks
    priority_set = set(priority_chunks) if priority_chunks else set()
    keep_positions = []
    tokens_allocated = 0
    
    # Allocate tokens to priority chunks first
    for i, (start, end) in enumerate(chunk_boundaries):
        chunk_tokens = end - start
        if i in priority_set and tokens_allocated < tokens_to_keep:
            tokens_for_chunk = min(chunk_tokens, tokens_to_keep - tokens_allocated)
            keep_positions.extend(range(start, start + tokens_for_chunk))
            tokens_allocated += tokens_for_chunk
    
    # Allocate remaining tokens
    remaining_tokens = tokens_to_keep - tokens_allocated
    non_priority_chunks = [(i, start, end) for i, (start, end) in enumerate(chunk_boundaries) if i not in priority_set]
    
    if remaining_tokens > 0 and non_priority_chunks:
        tokens_per_chunk = remaining_tokens // len(non_priority_chunks)
        
        for i, start, end in non_priority_chunks:
            if tokens_allocated >= tokens_to_keep:
                break
            chunk_tokens = end - start
            tokens_for_chunk = min(tokens_per_chunk, chunk_tokens, tokens_to_keep - tokens_allocated)
            keep_positions.extend(range(start, start + tokens_for_chunk))
            tokens_allocated += tokens_for_chunk
    
    # Sort positions to maintain order
    keep_positions = sorted(set(keep_positions))
    
    if not keep_positions:
        return full_kv_cache
    
    # Apply sparse selection
    sparse_kv_cache = []
    for keys, values in full_kv_cache:
        max_seq_len = values.shape[2]  # [batch, heads, seq_len, head_dim]
        valid_positions = [pos for pos in keep_positions if pos < max_seq_len]
        
        if valid_positions and len(valid_positions) < max_seq_len:
            sparse_keys = keys[:, :, valid_positions, :]
            sparse_values = values[:, :, valid_positions, :]
            sparse_kv_cache.append((sparse_keys, sparse_values))
        else:
            sparse_kv_cache.append((keys, values))
    
    return tuple(sparse_kv_cache)

# FIXED sampling with proper temperature:
def sample_with_temperature(logits: torch.Tensor, temperature: float = 0.7) -> torch.Tensor:
    """Proper temperature sampling"""
    if temperature <= 0:
        return torch.argmax(logits[:, -1, :], dim=-1)
    
    # Apply temperature
    scaled_logits = logits[:, -1, :] / temperature
    
    # Sample with temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probabilities, 1).squeeze(-1)
    
    return next_token

def optimized_generate_with_kv(
    sample: Dict[str, Any],
    gpu_chunks: Dict[int, Any],
    cpu_chunks: Dict[int, str],
    scheduler: HighPerformanceBanditScheduler,
    model: Any,
    tokenizer: Any,
    device: str,
    max_tokens: int = 20) -> Dict[str, Any]:
    """FIXED: Complete generation with all 4 problems resolved"""
    
    start_time = time.perf_counter()
    first_token_time = None
    decode_times = []
    
    # Track cumulative transfers
    total_promotions = 0
    total_demotions = 0
    
    # Initialize scheduler
    scheduler.initialize(
        sample=sample,
        gpu_chunks=gpu_chunks,
        cpu_chunks=cpu_chunks,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_gpu=len(gpu_chunks)
    )
    
    current_gpu_chunks = list(gpu_chunks.keys())
    
    # Build initial combined KV cache
    combined_kv = FastKVCacheManager.fast_concatenate_chunks(gpu_chunks, current_gpu_chunks)
    if combined_kv is None:
        return {"error": "No KV caches", "metrics": {"ttft_s": 0, "tpot_s": 0, "e2e_s": 0}}
    
    chunk_seq_len = combined_kv[0][0].shape[2]  # Get seq_len from first layer's keys
    
    # Prepare question
    question = sample.get("question", "")
    formatted_question = QUERY_PROMPT + question
    question_ids = tokenizer.encode(formatted_question, add_special_tokens=False)
    question_input = torch.tensor([question_ids], device=device)
    question_length = len(question_ids)
    
    total_context_len = chunk_seq_len + question_length
    attention_mask = torch.ones((1, total_context_len), device=device, dtype=torch.bool)
    position_ids = torch.arange(chunk_seq_len, chunk_seq_len + question_length, device=device).unsqueeze(0)
    
    generated_tokens = []
    trace = []
    
    with torch.inference_mode():
        # Initial forward pass
        initial_cache = DynamicCache.from_legacy_cache(combined_kv)
        initial_start = time.perf_counter()
        outputs = model(
            input_ids=question_input,
            past_key_values=initial_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True
        )
        past_key_values = outputs.past_key_values
        
        # FIXED: Proper temperature sampling
        next_token = sample_with_temperature(outputs.logits, temperature=0.7)
        token_id = next_token.item()
        first_token_time = time.perf_counter()
        initial_decode_time = first_token_time - initial_start
        decode_times.append(initial_decode_time)  # Add initial token time
        
        if token_id != tokenizer.eos_token_id:
            generated_tokens.append(token_id)
            trace.append({"step": 0, "token": token_id, "chunks_used": current_gpu_chunks.copy()})
        
        # Generation loop
        for step in range(1, max_tokens):
            if token_id == tokenizer.eos_token_id:
                print(f"[Pipeline] EOS at step {step}, stopping generation")
                break
            
            step_start = time.perf_counter()
            
            # Scheduling logic - run more frequently for short generations
            should_reschedule = (step % 10 == 0) and (step > 0)
            # should_reschedule = True
            
            if should_reschedule:
                print(f"\n[Pipeline] ===== SCHEDULING AT STEP {step} =====")
                # FIXED: 5-step prediction
                predicted_chunks = scheduler.predict(step + 5, generated_tokens)
                print(f"[Pipeline] step {step}: predicted for step {step + 5}: {predicted_chunks}")


                old_gpu_order = list(current_gpu_chunks)
                new_gpu_order = scheduler.schedule_to_gpu()
                print(f"[Pipeline] step {step}: schedule_to_gpu -> {new_gpu_order}")

                # Track transfers
                promotions_this_step = len(scheduler.last_promoted) if hasattr(scheduler, 'last_promoted') else 0
                demotions_this_step = len(scheduler.last_demoted) if hasattr(scheduler, 'last_demoted') else 0
                total_promotions += promotions_this_step
                total_demotions += demotions_this_step
                
                if promotions_this_step > 0 or demotions_this_step > 0:
                    print(f"[Pipeline] step {step}: +{promotions_this_step} promotions, -{demotions_this_step} demotions (total: {total_promotions + total_demotions})")
                
                # Handle GPU order changes
                if set(new_gpu_order) != set(current_gpu_chunks):
                    updated_gpu_chunks = scheduler.get_gpu_chunks()
                    new_combined_kv = FastKVCacheManager.fast_concatenate_chunks(
                        updated_gpu_chunks, new_gpu_order
                    )
                    
                    if new_combined_kv is not None:
                        full_len = new_combined_kv[0][0].shape[2]  # Get seq_len from first layer's keys
                        
                        try:
                            boundaries = compute_chunk_boundaries_from_kv_lengths(updated_gpu_chunks, new_gpu_order)
                            
                            # FIXED: Apply actual sparsity
                            if scheduler.enable_sparsity and scheduler.sparsity_ratio < 1.0:
                                priority_chunks = predicted_chunks if predicted_chunks else []
                                pruned_kv = create_sparse_kv_cache_with_ratio(
                                    new_combined_kv,
                                    boundaries,
                                    sparsity_ratio=scheduler.sparsity_ratio,
                                    priority_chunks=priority_chunks
                                )
                                combined_kv = pruned_kv
                                pruned_len = combined_kv[0][0].shape[2]
                                print(f"[Pipeline] step {step}: applied sparsity {scheduler.sparsity_ratio:.2f}, pruned_len={pruned_len}, full_len={full_len}")
                            else:
                                combined_kv = new_combined_kv
                                pruned_len = combined_kv[0][0].shape[2]
                                print(f"[Pipeline] step {step}: no sparsity, len={pruned_len}")
                        except Exception as e:
                            combined_kv = new_combined_kv
                            pruned_len = combined_kv[0][0].shape[2]
                            print(f"[Pipeline] step {step}: sparsity failed {e}, using full KV")
                        
                        current_gpu_chunks = new_gpu_order
                        
                        # FIXED: Incremental context update instead of full replay
                        new_total_context = pruned_len + question_length
                        attention_mask = torch.ones((1, new_total_context), device=device, dtype=torch.bool)
                        
                        rebuilt_cache = DynamicCache.from_legacy_cache(combined_kv)
                        context_outputs = model(
                            input_ids=question_input,
                            past_key_values=rebuilt_cache,
                            attention_mask=attention_mask,
                            use_cache=True,
                            return_dict=True
                        )
                        
                        # FIXED: Only replay recent tokens (last 3) instead of all
                        if generated_tokens:
                            recent_tokens = generated_tokens[-3:]  # Only last 3 tokens
                            prev_tokens = torch.tensor([recent_tokens], device=device)
                            replay_outputs = model(
                                input_ids=prev_tokens,
                                past_key_values=context_outputs.past_key_values,
                                use_cache=True,
                                return_dict=True
                            )
                            past_key_values = replay_outputs.past_key_values
                            print(f"[Pipeline] step {step}: replayed {len(recent_tokens)} recent tokens")
                        else:
                            past_key_values = context_outputs.past_key_values
                            print(f"[Pipeline] step {step}: using new context directly")
                else:
                    print(f"[Pipeline] step {step}: order-only change detected; skipping KV rebuild")
            
            # Generate next token
            input_token = next_token.unsqueeze(-1)
            outputs = model(
                input_ids=input_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            past_key_values = outputs.past_key_values
            
            # FIXED: Consistent temperature sampling
            next_token = sample_with_temperature(outputs.logits, temperature=0.7)
            token_id = next_token.item()
            
            if token_id == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(token_id)
            
            step_end = time.perf_counter()
            decode_times.append(step_end - step_start)
            
            trace.append({"step": step, "token": token_id, "chunks_used": current_gpu_chunks.copy()})
            
            # Update rewards
            if should_reschedule:
                scheduler.update_rewards(current_gpu_chunks, 1.0)
    
    # Generate final text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    end_time = time.perf_counter()
    
    print(f"[Pipeline] Generation complete: {len(generated_tokens)} tokens generated")
    
    # Calculate metrics
    ttft_s = first_token_time - start_time if first_token_time else 0.0
    tpot_s = sum(decode_times) / len(decode_times) if decode_times else 0.0
    e2e_s = end_time - start_time
    throughput_tps = len(generated_tokens) / sum(decode_times) if decode_times else 0.0
    
    return {
        "question": question,
        "answer": generated_text,
        "generated_tokens": generated_tokens,
        "trace": trace,
        "final_gpu_chunks": current_gpu_chunks,
        "metrics": {
            "ttft_s": ttft_s,
            "tpot_s": tpot_s,
            "e2e_s": e2e_s,
            "throughput_tps": throughput_tps,
            "num_tokens": len(generated_tokens),
            "transfers_promotions": total_promotions,
            "transfers_demotions": total_demotions,
            "transfers_total": total_promotions + total_demotions
        }
    }


def run_optimized_pipeline(
    input_file: str,
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    output_dir: str = "results",
    top_k: int = 5,
    max_tokens: int = 32,
    device: str = "cuda:0"
):
    logger = setup_logging("WARNING")  # Minimal logging
    os.makedirs(output_dir, exist_ok=True)
    
    # OPTIMIZED: Load and prepare data efficiently
    samples = load_samples(input_file)
    
    # OPTIMIZED: Setup retrieval once
    retrieval_config = RetrievalConfig()
    retriever = ColbertRetrieval(retrieval_config)
    
    # OPTIMIZED: Initialize model/tokenizer once with optimal settings
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # OPTIMIZED: Load model with performance optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use FP16 for speed
        device_map="auto"
    ).eval()
    
    # OPTIMIZED: Use high-performance scheduler
    scheduler = HighPerformanceBanditScheduler(
        scheduler_interval=15,  # Reduced scheduling frequency
        promote_per_step=2,     # Allow 2 swaps per scheduling step
        exploration_c=1.5,      # Higher exploration bonus
        max_candidates=15,      # Smaller search space
        sparsity_ratio=0.6,
        enable_sparsity=True,
        sparsity_strategy="priority",
        epsilon=0.3  # 30% chance of forced exploration
    )
    
    results = []
    
    # OPTIMIZED: Process samples with minimal overhead
    for si, sample in enumerate(tqdm(samples, desc="Processing", unit="sample")):
        
        try:
            # OPTIMIZED: Fast retrieval
            prepared_samples = retriever.prepare([sample])
            retrieved_samples = retriever.retrieve(prepared_samples, top_k=top_k)
            sample = retrieved_samples[0] if retrieved_samples else sample
            
            # OPTIMIZED: Build KV caches with provided model
            kv_result = build_chunk_kv_caches(
                samples=[sample],
                model_id=model_id,
                top_k=top_k,
                device=device,
                provided_tokenizer=tokenizer,
                provided_model=model
            )
            
            gpu_chunks = kv_result.get("gpu_chunks", {})
            cpu_chunks = kv_result.get("cpu_chunks", {})
            
            print(f"\n[Sample {si}] Initial split: {len(gpu_chunks)} GPU chunks, {len(cpu_chunks)} CPU chunks")
            print(f"[Sample {si}] GPU chunk IDs: {list(gpu_chunks.keys())}")
            print(f"[Sample {si}] CPU chunk IDs: {list(cpu_chunks.keys())}")
            
            if not gpu_chunks:
                print(f"[Sample {si}] ERROR: No GPU chunks available, skipping")
                continue
            
            # OPTIMIZED: Generate with high-performance implementation
            result = optimized_generate_with_kv(
                sample=sample,
                gpu_chunks=gpu_chunks,
                cpu_chunks=cpu_chunks,
                scheduler=scheduler,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_tokens=max_tokens
            )
            
            result["sample_id"] = str(sample.get("id", f"sample_{si}"))
            result["sample_index"] = si
            results.append(result)

            # Per-sample metrics printout
            metrics = result.get("metrics", {})
            ttft_ms = metrics.get("ttft_ms", metrics.get("ttft_s", 0.0) * 1000.0)
            tpot_ms = metrics.get("tpot_ms", metrics.get("tpot_s", 0.0) * 1000.0)
            e2e_s = metrics.get("e2e_s", 0.0)
            throughput = metrics.get("throughput_tps", 0.0)
            transfers_total = metrics.get("transfers_total", 0)
            answer_text = result.get("answer", "")
            print(f"[Sample {si}] TTFT: {ttft_ms:.1f} ms | TPOT: {tpot_ms:.1f} ms | Latency: {e2e_s:.3f} s | Throughput: {throughput:.2f} tok/s | Transfers: {transfers_total}")
            print(f"[Sample {si}] Final answer: {answer_text}")
            
        except Exception as e:
            logger.warning(f"Sample {si} failed: {str(e)}")
            continue
    
    # OPTIMIZED: Save results efficiently
    output_path = os.path.join(output_dir, "optimized_results.json")
    with open(output_path, "w") as f:
        serializable_results = convert_to_serializable({"results": results})
        json.dump(serializable_results, f, indent=2)
    
    # OPTIMIZED: Calculate and display performance metrics
    if results:
        successful_results = [r for r in results if "metrics" in r]
        if successful_results:
            avg_ttft = sum(r["metrics"]["ttft_s"] for r in successful_results) / len(successful_results)
            avg_tpot = sum(r["metrics"]["tpot_s"] for r in successful_results) / len(successful_results)
            avg_e2e = sum(r["metrics"]["e2e_s"] for r in successful_results) / len(successful_results)
            avg_throughput = sum(r["metrics"]["throughput_tps"] for r in successful_results) / len(successful_results)
            avg_transfers = sum(r["metrics"].get("transfers_total", 0) for r in successful_results) / len(successful_results)
            
            print(f"\n=== OPTIMIZED PERFORMANCE METRICS ===")
            print(f"Samples processed: {len(successful_results)}")
            print(f"Average TTFT: {avg_ttft*1000:.1f} ms")
            print(f"Average TPOT: {avg_tpot*1000:.1f} ms")
            print(f"Average E2E: {avg_e2e:.3f} s")
            print(f"Average throughput: {avg_throughput:.1f} tok/s")
            print(f"Average transfers per sample: {avg_transfers:.2f}")
            print(f"Results saved to: {output_path}")
    
    return {"output_path": output_path, "results": results}

def main():
    import argparse
    
    parser = argparse.ArgumentParser("Optimized RAG Pipeline for Minimal Latency")
    parser.add_argument("--input", required=True, help="Input dataset JSON")
    parser.add_argument("--output_dir", default="results_optimized", help="Output directory")
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.2", help="Model ID")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k chunks")
    parser.add_argument("--max_tokens", type=int, default=20, help="Max generation length")
    parser.add_argument("--device", default="cuda:0", help="Device")
    
    args = parser.parse_args()
    
    run_optimized_pipeline(
        input_file=args.input,
        model_id=args.model_id,
        output_dir=args.output_dir,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        device=args.device
    )

if __name__ == "__main__":
    main()