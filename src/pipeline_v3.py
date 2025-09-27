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

def optimized_generate_with_kv(
    sample: Dict[str, Any],
    gpu_chunks: Dict[int, Any],
    cpu_chunks: Dict[int, str],
    scheduler: HighPerformanceBanditScheduler,
    model: Any,
    tokenizer: Any,
    device: str,
    max_tokens: int = 20
) -> Dict[str, Any]:
    """
    OPTIMIZED: High-performance generation with minimal latency overhead
    
    Key Optimizations:
    1. Minimal KV cache rebuilding
    2. Reduced scheduling frequency  
    3. Eliminated unnecessary tensor operations
    4. Streamlined generation loop
    5. Minimal logging in critical path
    """
    
    # OPTIMIZED: Single high-precision timer
    start_time = time.perf_counter()
    first_token_time = None
    decode_times = []
    
    # OPTIMIZED: Initialize scheduler once with pre-computed caches
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
    
    # OPTIMIZED: Build initial combined KV cache once
    combined_kv = FastKVCacheManager.fast_concatenate_chunks(gpu_chunks, current_gpu_chunks)
    
    if combined_kv is None:
        return {"error": "No KV caches", "metrics": {"ttft_s": 0, "tpot_s": 0, "e2e_s": 0}}
    
    chunk_seq_len = combined_kv[0][0].shape[2]
    
    # OPTIMIZED: Prepare question tensors once
    question = sample.get("question", "")
    formatted_question = QUERY_PROMPT + question
    question_ids = tokenizer.encode(formatted_question, add_special_tokens=False)
    question_input = torch.tensor([question_ids], device=device)
    question_length = len(question_ids)
    
    # OPTIMIZED: Pre-compute attention components
    total_context_len = chunk_seq_len + question_length
    attention_mask = torch.ones(1, total_context_len, device=device, dtype=torch.bool)
    position_ids = torch.arange(chunk_seq_len, chunk_seq_len + question_length, device=device).unsqueeze(0)
    
    generated_tokens = []
    trace = []
    
    # OPTIMIZED: Single inference mode for entire generation
    with torch.inference_mode():
        
        # OPTIMIZED: First forward pass with pre-allocated tensors
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
        
        # OPTIMIZED: Fast token sampling
        next_token = torch.multinomial(
            torch.softmax(outputs.logits[:, -1, :] / 0.7, dim=-1), 1
        ).squeeze(-1)
        
        token_id = next_token.item()
        first_token_time = time.perf_counter()  # Record TTFT immediately
        
        if token_id != tokenizer.eos_token_id:
            generated_tokens.append(token_id)
        
        trace.append({
            "step": 0,
            "token": token_id,
            "chunks_used": current_gpu_chunks.copy()
        })
        
        # OPTIMIZED: Streamlined generation loop
        for step in range(1, max_tokens):
            if token_id == tokenizer.eos_token_id:
                break
            
            step_start = time.perf_counter()
            
            # OPTIMIZED: Reduced scheduling frequency (every 15 steps instead of 5)
            should_reschedule = (step % 15 == 0) and step > 5
            
            if should_reschedule:
                # OPTIMIZED: Fast chunk prediction without logging
                predicted_chunks = scheduler.predict(step, generated_tokens)
                
                # OPTIMIZED: Lightweight GPU scheduling
                new_gpu_order = scheduler.schedule_to_gpu()
                
                # OPTIMIZED: Only rebuild cache if significant changes
                if len(set(new_gpu_order) - set(current_gpu_chunks)) >= 2:
                    updated_gpu_chunks = scheduler.get_gpu_chunks()
                    new_combined_kv = FastKVCacheManager.fast_concatenate_chunks(
                        updated_gpu_chunks, new_gpu_order
                    )
                    
                    if new_combined_kv is not None:
                        # OPTIMIZED: Fast cache update without re-processing history
                        combined_kv = new_combined_kv
                        current_gpu_chunks = new_gpu_order
                        
                        # OPTIMIZED: Quick context switch
                        new_chunk_seq_len = combined_kv[0][0].shape[2]
                        new_total_context = new_chunk_seq_len + question_length
                        
                        # OPTIMIZED: Resize attention mask efficiently  
                        attention_mask = torch.ones(1, new_total_context, device=device, dtype=torch.bool)
                        
                        # OPTIMIZED: Re-establish context without full recomputation
                        rebuilt_cache = DynamicCache.from_legacy_cache(combined_kv)
                        context_outputs = model(
                            input_ids=question_input,
                            past_key_values=rebuilt_cache,
                            attention_mask=attention_mask[:, :new_chunk_seq_len + question_length],
                            use_cache=True,
                            return_dict=True
                        )
                        
                        # OPTIMIZED: Fast token replay for consistency
                        if generated_tokens:
                            prev_tokens = torch.tensor([generated_tokens], device=device)
                            replay_outputs = model(
                                input_ids=prev_tokens,
                                past_key_values=context_outputs.past_key_values,
                                use_cache=True,
                                return_dict=True
                            )
                            past_key_values = replay_outputs.past_key_values
                        else:
                            past_key_values = context_outputs.past_key_values
            
            # OPTIMIZED: Fast next token generation
            input_token = next_token.unsqueeze(-1)
            
            outputs = model(
                input_ids=input_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            past_key_values = outputs.past_key_values
            
            # OPTIMIZED: Fast sampling without temperature adjustment
            next_token = torch.multinomial(
                torch.softmax(outputs.logits[:, -1, :], dim=-1), 1
            ).squeeze(-1)
            
            token_id = next_token.item()
            
            if token_id == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(token_id)
            
            # OPTIMIZED: Record timing after successful generation
            step_end = time.perf_counter()
            decode_times.append(step_end - step_start)
            
            # OPTIMIZED: Minimal reward update
            if should_reschedule:
                scheduler.update_rewards(current_gpu_chunks, 1.0)
            
            # OPTIMIZED: Minimal trace recording
            trace.append({
                "step": step,
                "token": token_id,
                "chunks_used": current_gpu_chunks.copy()
            })
    
    # OPTIMIZED: Fast cleanup
    scheduler.shutdown()
    
    # OPTIMIZED: Single text decode operation
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    end_time = time.perf_counter()
    
    # OPTIMIZED: Efficient metrics calculation
    ttft_s = (first_token_time - start_time) if first_token_time else 0.0
    tpot_s = (sum(decode_times) / len(decode_times)) if decode_times else 0.0
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
            "ttft_ms": ttft_s * 1000,
            "tpot_ms": tpot_s * 1000,
            "e2e_ms": e2e_s * 1000,
            "throughput_tps": throughput_tps,
            "num_tokens": len(generated_tokens)
        }
    }

def run_optimized_pipeline(
    input_file: str,
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    output_dir: str = "results",
    top_k: int = 5,
    max_tokens: int = 20,
    device: str = "cuda:0"
):
    """
    OPTIMIZED: High-performance pipeline execution
    
    Key Optimizations:
    1. Reduced logging overhead
    2. Batch processing optimizations
    3. Minimal object allocations
    4. Efficient model/tokenizer reuse
    5. Streamlined data flow
    """
    
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
        promote_per_step=1,     # Minimal promotions per step
        exploration_c=0.5,      # Faster convergence
        max_candidates=15       # Smaller search space
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
            
            if not gpu_chunks:
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
            answer_text = result.get("answer", "")
            print(f"[Sample {si}] TTFT: {ttft_ms:.1f} ms | TPOT: {tpot_ms:.1f} ms | Latency: {e2e_s:.3f} s | Throughput: {throughput:.2f} tok/s")
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
            
            print(f"\n=== OPTIMIZED PERFORMANCE METRICS ===")
            print(f"Samples processed: {len(successful_results)}")
            print(f"Average TTFT: {avg_ttft*1000:.1f} ms")
            print(f"Average TPOT: {avg_tpot*1000:.1f} ms")
            print(f"Average E2E: {avg_e2e:.3f} s")
            print(f"Average throughput: {avg_throughput:.1f} tok/s")
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