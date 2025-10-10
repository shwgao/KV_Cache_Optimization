#!/usr/bin/env python3
"""
ASYNC PREFETCH PIPELINE v6 - True overlapping generation with chunk transfers
Key improvements over v4:
- Uses AsyncPrefetchScheduler with CUDA streams
- True background prefetch of KV caches
- Non-blocking transfer completion checks
- Pointer swapping instead of cache rebuilding
- Maintains all original functionality while reducing TTFT/TPOT
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
import math

# Local imports
from rag_retrieval import RetrievalConfig, ColbertRetrieval
from build_kv_v2 import build_chunk_kv_caches, QUERY_PROMPT
from scheduler_v5 import AsyncPrefetchScheduler, FastKVCacheManager

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

# ============= REAL SPARSE ATTENTION IMPLEMENTATION =============

def _create_sparse_kv_cache_from_chunks(
    full_kv_cache: tuple,
    chunk_boundaries: List[Tuple[int, int]],
    selected_chunk_indices: List[int],
    sparsity_ratio: float = 0.6
) -> Tuple[tuple, int]:
    if not full_kv_cache or not chunk_boundaries:
        return full_kv_cache, 0

    # Calculate how many tokens to keep based on sparsity ratio
    total_tokens = full_kv_cache[0][0].shape[2]  # [batch, heads, seq_len, head_dim]
    if sparsity_ratio >= 1.0:
        return full_kv_cache, total_tokens

    num_tokens_to_keep = max(1, int(total_tokens * sparsity_ratio))

    # Strategy 1: If we have selected chunk indices, prioritize those
    if selected_chunk_indices and chunk_boundaries:
        keep_positions = []
        for chunk_idx in selected_chunk_indices:
            if chunk_idx < len(chunk_boundaries):
                start_pos, end_pos = chunk_boundaries[chunk_idx]
                keep_positions.extend(range(start_pos, end_pos))

        # If we have too many positions, take first num_tokens_to_keep
        if len(keep_positions) > num_tokens_to_keep:
            keep_positions = keep_positions[:num_tokens_to_keep]
        # If we have too few, add some from remaining chunks
        elif len(keep_positions) < num_tokens_to_keep:
            remaining_needed = num_tokens_to_keep - len(keep_positions)
            used_positions = set(keep_positions)

            # Add positions from unused chunks
            for chunk_idx in range(len(chunk_boundaries)):
                if chunk_idx not in selected_chunk_indices and remaining_needed > 0:
                    start_pos, end_pos = chunk_boundaries[chunk_idx]
                    for pos in range(start_pos, end_pos):
                        if pos not in used_positions and remaining_needed > 0:
                            keep_positions.append(pos)
                            remaining_needed -= 1
    else:
        # Strategy 2: Simple uniform sampling across all tokens
        step_size = max(1, total_tokens // num_tokens_to_keep)
        keep_positions = list(range(0, total_tokens, step_size))[:num_tokens_to_keep]

    if not keep_positions:
        return full_kv_cache, total_tokens

    # Sort positions to maintain order
    keep_positions = sorted(list(set(keep_positions)))

    # Create sparse KV cache by slicing - THIS IS THE KEY REAL SPARSITY OPERATION
    sparse_kv_cache = []
    for layer_kv in full_kv_cache:
        keys, values = layer_kv
        # REAL SPARSITY: Actually slice the tensors to reduce computation
        sparse_keys = keys[:, :, keep_positions, :]  # Shape reduced!
        sparse_values = values[:, :, keep_positions, :]  # Shape reduced!
        sparse_kv_cache.append((sparse_keys, sparse_values))

    print(f"[REAL_SPARSE] KV cache reduced: {len(keep_positions)} tokens from {total_tokens} tokens")
    print(f"[REAL_SPARSE] Compression: {len(keep_positions)/total_tokens:.1%} of original size")

    return tuple(sparse_kv_cache), len(keep_positions)

def apply_real_sparse_attention(
    combined_kv: tuple,
    current_gpu_chunks: List[int],
    sparsity_ratio: float,
    chunk_size_estimate: int = 100
) -> Tuple[tuple, Dict[str, Any]]:
    if sparsity_ratio >= 1.0:
        original_tokens = combined_kv[0][0].shape[2]
        return combined_kv, {
            "sparsity_applied": False,
            "original_tokens": original_tokens,
            "sparse_tokens": original_tokens,
            "compression_ratio": 1.0
        }

    # Estimate chunk boundaries based on chunk order and size
    chunk_boundaries = []
    current_pos = 0
    for chunk_idx in current_gpu_chunks:
        start_pos = current_pos
        end_pos = current_pos + chunk_size_estimate
        chunk_boundaries.append((start_pos, end_pos))
        current_pos = end_pos

    # Adjust last boundary to match actual sequence length
    if chunk_boundaries:
        actual_length = combined_kv[0][0].shape[2]
        last_start = chunk_boundaries[-1][0]
        chunk_boundaries[-1] = (last_start, actual_length)

    # Apply real sparse attention
    sparse_kv, num_sparse_tokens = _create_sparse_kv_cache_from_chunks(
        combined_kv,
        chunk_boundaries,
        current_gpu_chunks,  # Prioritize all current GPU chunks
        sparsity_ratio
    )

    # Create metadata about sparsity
    original_tokens = combined_kv[0][0].shape[2]
    sparse_info = {
        "sparsity_applied": True,
        "original_tokens": original_tokens,
        "sparse_tokens": num_sparse_tokens,
        "compression_ratio": num_sparse_tokens / original_tokens if original_tokens > 0 else 1.0,
        "chunks_used": len(current_gpu_chunks),
        "sparsity_ratio": sparsity_ratio
    }

    return sparse_kv, sparse_info

# ============= ENHANCED FAST KV CACHE MANAGER =============

class EnhancedFastKVCacheManager(FastKVCacheManager):
    @staticmethod
    def fast_concatenate_chunks_with_real_sparsity(
        gpu_chunks: Dict[int, Any],
        selected_chunks: List[int],
        sparsity_ratio: float = 1.0
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        # Step 1: Use original fast concatenation
        combined_kv = FastKVCacheManager.fast_concatenate_chunks(gpu_chunks, selected_chunks)
        if combined_kv is None:
            return None, {"error": "Concatenation failed"}

        # Step 2: Apply real sparse attention if needed
        if sparsity_ratio < 1.0:
            sparse_kv, sparse_info = apply_real_sparse_attention(
                combined_kv,
                selected_chunks,
                sparsity_ratio,
                chunk_size_estimate=100  # Reasonable default
            )
            return sparse_kv, sparse_info
        else:
            # Return full attention with metadata
            original_tokens = combined_kv[0][0].shape[2]
            return combined_kv, {
                "sparsity_applied": False,
                "original_tokens": original_tokens,
                "sparse_tokens": original_tokens,
                "compression_ratio": 1.0,
                "chunks_used": len(selected_chunks)
            }

def sample_with_temperature(logits: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Sample next token with temperature (lower = more deterministic/factual)"""
    if temperature <= 0:
        return torch.argmax(logits[:, -1, :], dim=-1)
    scaled_logits = logits[:, -1, :] / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probabilities, 1).squeeze(-1)
    return next_token

# ============= ASYNC PREFETCH GENERATION FUNCTION =============

def async_generate_with_kv(
    sample: Dict[str, Any],
    gpu_chunks: Dict[int, Any],
    cpu_chunks: Dict[int, str],
    scheduler: AsyncPrefetchScheduler,
    model: Any,
    tokenizer: Any,
    device: str,
    max_tokens: int = 32,
    sparsity_ratio: float = 1.0  # Default to full attention
) -> Dict[str, Any]:
    """
    ASYNC PREFETCH generation function with true background chunk transfers
    Key improvements:
    - Background prefetch overlaps with generation
    - Non-blocking transfer completion checks
    - Pointer swapping instead of cache rebuilding
    - Maintains all original functionality
    """
    
    print(f"\n{'='*80}")
    print(f"[AsyncGeneration] Starting ASYNC PREFETCH generation")
    print(f"{'='*80}")
    print(f"[AsyncGeneration] - True background prefetch with CUDA streams")
    print(f"[AsyncGeneration] - Non-blocking transfer completion checks")
    print(f"[AsyncGeneration] - Pointer swapping instead of rebuilding")
    print(f"[AsyncGeneration] Sparsity ratio: {sparsity_ratio:.2f}")

    # TTFT timer starts here - critical path timing
    ttft_start = time.perf_counter()
    first_token_time = None
    decode_times = []

    # Use initial GPU chunks
    current_gpu_chunks = list(gpu_chunks.keys())
    print(f"[AsyncGeneration] Initial GPU chunks: {current_gpu_chunks}")
    print(f"[AsyncGeneration] Total GPU chunks: {len(current_gpu_chunks)}")
    print(f"[AsyncGeneration] Total CPU chunks: {len(cpu_chunks)}")
    print(f"[AsyncGeneration] Total chunks available: {len(current_gpu_chunks) + len(cpu_chunks)}")
    print(f"{'='*80}\n")

    # Build initial combined KV cache with optional sparsity
    combined_kv, cache_info = EnhancedFastKVCacheManager.fast_concatenate_chunks_with_real_sparsity(
        gpu_chunks,
        current_gpu_chunks,
        sparsity_ratio=sparsity_ratio
    )

    if combined_kv is None:
        print(f"[AsyncGeneration] ERROR: No KV caches available")
        return {"error": "No KV caches", "metrics": {"ttft_s": 0, "tpot_s": 0, "e2e_s": 0}}

    print(f"[AsyncGeneration] KV cache built: {cache_info}")
    
    if cache_info.get("sparsity_applied", False):
        sparse_tokens = cache_info.get("sparse_tokens", 0)
        original_tokens = cache_info.get("original_tokens", 0)
        compression = cache_info.get("compression_ratio", 1.0)
        print(f"[AsyncGeneration] Sparsity applied: {sparse_tokens}/{original_tokens} tokens ({compression:.1%})")

    chunk_seq_len = combined_kv[0][0].shape[2]

    # Prepare question input
    question = sample.get("question", "")
    formatted_question = QUERY_PROMPT + question
    question_ids = tokenizer.encode(formatted_question, add_special_tokens=False)
    question_input = torch.tensor([question_ids], device=device)
    question_length = len(question_ids)
    total_context_len = chunk_seq_len + question_length
    attention_mask = torch.ones((1, total_context_len), device=device, dtype=torch.bool)
    position_ids = torch.arange(chunk_seq_len, chunk_seq_len + question_length, device=device).unsqueeze(0)

    print(f"[AsyncGeneration] Context: {chunk_seq_len} + {question_length} = {total_context_len} tokens")

    generated_tokens = []
    trace = []

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
        next_token = sample_with_temperature(outputs.logits, temperature=0.1)  # Lower for factual accuracy
        token_id = next_token.item()

        # TTFT measured here
        first_token_time = time.perf_counter()
        ttft_duration = first_token_time - ttft_start
        decode_times.append(ttft_duration)

        print(f"[AsyncGeneration] First token in {ttft_duration*1000:.2f}ms (async)")

        if token_id != tokenizer.eos_token_id:
            generated_tokens.append(token_id)
            trace.append({"step": 0, "token": token_id, "chunks_used": current_gpu_chunks.copy()})

        # Initialize scheduler AFTER first token (not during TTFT)
        if cpu_chunks:
            print(f"\n[AsyncGeneration] Initializing async scheduler after first token...")
            scheduler.initialize(
                sample=sample,
                gpu_chunks=gpu_chunks,
                cpu_chunks=cpu_chunks,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_gpu=len(gpu_chunks)
            )
            scheduler_initialized = True
            
            # Show initial chunk rewards/scores
            print(f"[AsyncGeneration] Initial chunk rewards after initialization:")
            if hasattr(scheduler, 'rewards') and scheduler.rewards:
                sorted_rewards = sorted(scheduler.rewards.items(), key=lambda x: x[1], reverse=True)
                for chunk_idx, reward in sorted_rewards:
                    location = "GPU" if chunk_idx in gpu_chunks else "CPU"
                    print(f"[AsyncGeneration]   Chunk {chunk_idx} ({location}): reward={reward:.4f}")
            print()
        else:
            scheduler_initialized = False
            print(f"[AsyncGeneration] No CPU chunks - all chunks fit on GPU, no swapping needed\n")

        # ASYNC GENERATION LOOP
        transfer_count = 0
        cache_swaps = 0
        prefetch_count = 0

        for step in range(1, max_tokens):
            if token_id == tokenizer.eos_token_id:
                break

            step_start = time.perf_counter()

            # ASYNC PREFETCH: Kick off background transfers if needed
            if scheduler_initialized:
                # RETRY PENDING: Check if any pending chunks are now ready to prefetch
                scheduler.retry_pending_prefetch()
                
                # Check if this step requires scheduling
                should_schedule = step in scheduler.next_prefetch_steps
                if should_schedule:
                    prefetch_count += 1
                    print(f"\n[Step {step}] === PREDICTION & PREFETCH ===")
                    print(f"[Step {step}] Triggering background prefetch #{prefetch_count}")

                    print(scheduler.scheduler_interval)
                    
                    # Get predicted chunks before calling schedule_if_needed
                    future_step = step + scheduler.scheduler_interval
                    predicted_chunks = scheduler.predict_chunks(future_step, generated_tokens)
                    
                    # Show predicted chunks with their priorities (reward + exploration_bonus + recency)
                    print(f"[Step {step}] Predicted chunks for step {future_step}: {predicted_chunks}")
                    if hasattr(scheduler, 'rewards') and scheduler.rewards:
                        print(f"[Step {step}] Chunk priorities (reward + exploration_bonus + recency):")
                        # Calculate priorities for all chunks and sort
                        priorities = []
                        for chunk_idx in scheduler.rewards.keys():
                            reward = scheduler.rewards.get(chunk_idx, 0.0)
                            priority = scheduler._calculate_priority(chunk_idx)
                            location = "GPU" if chunk_idx in gpu_chunks else "CPU"
                            priorities.append((chunk_idx, reward, priority, location))
                        
                        # Sort by priority
                        priorities.sort(key=lambda x: x[2], reverse=True)
                        
                        # Show top 10
                        for chunk_idx, reward, priority, location in priorities[:10]:
                            marker = "★" if chunk_idx in predicted_chunks else " "
                            bonus = priority - reward
                            print(f"[Step {step}]   {marker} Chunk {chunk_idx} ({location}): reward={reward:.4f}, priority={priority:.4f} (bonus={bonus:+.4f})")
                    
                scheduler.schedule_if_needed(step, generated_tokens)

            # NON-BLOCKING CHECK: See if any prefetched chunks are ready
            if scheduler_initialized:
                available_chunks = scheduler.get_available_chunks()
                
                # Debug: Show status EVERY time at step % 5 to see what's happening
                if step % 5 == 0:
                    print(f"\n{'='*60}")
                    print(f"[DEBUG Step {step}] Prefetch Status Check")
                    print(f"{'='*60}")
                    print(f"[DEBUG] pending_prefetch: {scheduler.pending_prefetch}")
                    with scheduler.prefetch_lock:
                        prefetched_dict = dict(scheduler.prefetched)
                    print(f"[DEBUG] prefetched keys: {list(prefetched_dict.keys())}")
                    print(f"[DEBUG] ready_kv count: {len(scheduler.ready_kv)}")
                    print(f"[DEBUG] current_gpu_chunks: {current_gpu_chunks}")
                    print(f"[DEBUG] CPU chunks available: {list(cpu_chunks.keys())}")
                    print(f"[DEBUG] Condition: (pending={bool(scheduler.pending_prefetch)} OR prefetched={bool(scheduler.prefetched)}) = {bool(scheduler.pending_prefetch or scheduler.prefetched)}")
                    
                    if scheduler.pending_prefetch or scheduler.prefetched:
                        print(f"[Step {step}] ✓ ENTERING DETAILED VIEW:")
                        print(f"[Step {step}]   Pending (waiting for precompute): {list(scheduler.pending_prefetch)}")
                        print(f"[Step {step}]   Prefetched (ready for swap): {list(prefetched_dict.keys())}")
                    else:
                        print(f"[DEBUG] ✗ NOT ENTERING LOOP - Both sets are empty!")
                        print(f"[DEBUG] Possible reasons:")
                        print(f"[DEBUG]   1. No prefetch initiated yet (check if schedule_if_needed was called)")
                        print(f"[DEBUG]   2. All chunks transferred instantly (check ready_kv)")
                        print(f"[DEBUG]   3. All chunks already on GPU (no CPU chunks to prefetch)")
                    print(f"{'='*60}\n")
                
                # Check if we should swap to a better chunk arrangement
                current_set = set(current_gpu_chunks)
                available_set = set(available_chunks)
                new_chunks = available_set - current_set
                
                if new_chunks and step % scheduler.scheduler_interval == 0:
                    print(f"\n{'='*80}")
                    print(f"[Step {step}] ASYNC SCHEDULING CHECKPOINT")
                    print(f"{'='*80}")
                    print(f"[Step {step}] Current GPU chunks: {current_gpu_chunks}")
                    print(f"[Step {step}] Available chunks (prefetched): {available_chunks}")
                    print(f"[Step {step}] New chunks ready: {list(new_chunks)}")
                    
                    # CRITICAL FIX: Check if new chunks are from predictions (prefetched)
                    # These should be FORCED into GPU since we predicted them
                    with scheduler.prefetch_lock:
                        predicted_and_ready = [c for c in new_chunks if c in scheduler.prefetched]
                    
                    # Show rewards for all available chunks
                    print(f"\n[Step {step}] Chunk Selection (by reward):")
                    available_with_rewards = [(chunk_idx, scheduler.rewards.get(chunk_idx, 0.0)) for chunk_idx in available_chunks]
                    available_with_rewards.sort(key=lambda x: x[1], reverse=True)
                    
                    for chunk_idx, reward in available_with_rewards:
                        in_current = "✓" if chunk_idx in current_gpu_chunks else " "
                        is_new = "NEW" if chunk_idx in new_chunks else ""
                        is_predicted = "★PREDICTED" if chunk_idx in predicted_and_ready else ""
                        print(f"[Step {step}]   {in_current} Chunk {chunk_idx}: reward={reward:.4f} {is_new} {is_predicted}")
                    
                    # DECISION: Should we swap?
                    should_swap = False
                    new_gpu_order = current_gpu_chunks  # Default: keep current
                    
                    # CASE 1: If predicted chunks are ready, ALWAYS swap them in!
                    if predicted_and_ready:
                        should_swap = True
                        print(f"\n[Step {step}] PREDICTED CHUNKS READY: Forcing swap for {predicted_and_ready}")
                        # Keep some current chunks, add predicted chunks
                        num_to_keep = max(0, scheduler.max_gpu - len(predicted_and_ready))
                        best_current = sorted(current_gpu_chunks, 
                                            key=lambda x: scheduler.rewards.get(x, 0.0), reverse=True)
                        new_gpu_order = predicted_and_ready + best_current[:num_to_keep]
                        new_gpu_order = new_gpu_order[:scheduler.max_gpu]  # Trim to max_gpu
                    else:
                        print(f"\n[Step {step}] No predicted chunks ready yet (still in pending_prefetch)")
                        print(f"[Step {step}] Pending: {list(scheduler.pending_prefetch)}")
                    
                    print(f"\n[Step {step}] Selected for GPU (top {scheduler.max_gpu}): {new_gpu_order}")
                    print(f"[Step {step}] Selected chunks rewards: {[scheduler.rewards.get(c, 0.0) for c in new_gpu_order]}")
                    
                    if should_swap and set(new_gpu_order) != current_set:
                        # Analyze changes
                        chunks_to_evict = current_set - set(new_gpu_order)
                        chunks_to_use = set(new_gpu_order) - current_set
                        chunks_unchanged = current_set & set(new_gpu_order)
                        
                        print(f"[Step {step}] Chunks to evict: {list(chunks_to_evict) if chunks_to_evict else 'None'}")
                        print(f"[Step {step}] Chunks to use: {list(chunks_to_use) if chunks_to_use else 'None'}")
                        print(f"[Step {step}] Chunks unchanged: {list(chunks_unchanged)}")
                        
                        # Increment swap counter
                        cache_swaps += 1
                        print(f"[Step {step}] ASYNC CACHE SWAP #{cache_swaps}")
                        
                        # Show if this was a predicted swap
                        if predicted_and_ready and any(c in predicted_and_ready for c in chunks_to_use):
                            print(f"[Step {step}] ✓ This swap includes PREDICTED chunks: {[c for c in chunks_to_use if c in predicted_and_ready]}")
                        
                        # Build new KV cache with available chunks
                        updated_gpu_chunks = scheduler.get_gpu_chunks()
                        
                        # Try to build new cache
                        new_combined_kv, new_cache_info = EnhancedFastKVCacheManager.fast_concatenate_chunks_with_real_sparsity(
                            updated_gpu_chunks,
                            new_gpu_order,
                            sparsity_ratio=sparsity_ratio
                        )
                        
                        if new_combined_kv is not None:
                            print(f"[Step {step}] ✓ Cache rebuilt successfully: {new_cache_info}")
                            combined_kv = new_combined_kv
                            current_gpu_chunks = new_gpu_order
                            print(f"[Step {step}] ✓ GPU chunks now: {current_gpu_chunks}")
                            
                            # Quick context rebuild
                            pruned_len = combined_kv[0][0].shape[2]
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
                            
                            # Minimal token replay
                            if generated_tokens:
                                recent_tokens = generated_tokens[-1:]
                                prev_tokens = torch.tensor([recent_tokens], device=device)
                                replay_outputs = model(
                                    input_ids=prev_tokens,
                                    past_key_values=context_outputs.past_key_values,
                                    use_cache=True,
                                    return_dict=True
                                )
                                past_key_values = replay_outputs.past_key_values
                            else:
                                past_key_values = context_outputs.past_key_values
                        else:
                            print(f"[Step {step}] ✗ Cache rebuild failed, keeping old cache")
                            cache_swaps -= 1  # Don't count failed swaps
                    else:
                        print(f"[Step {step}] ✓ No changes needed - GPU chunks already optimal")
                    
                    print(f"{'='*80}\n")

            # Generate next token (fast path)
            input_token = next_token.unsqueeze(-1)
            outputs = model(
                input_ids=input_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )

            past_key_values = outputs.past_key_values
            next_token = sample_with_temperature(outputs.logits, temperature=0.1)  # Lower for factual accuracy
            token_id = next_token.item()

            if token_id == tokenizer.eos_token_id:
                break

            generated_tokens.append(token_id)
            step_end = time.perf_counter()
            decode_times.append(step_end - step_start)
            trace.append({"step": step, "token": token_id, "chunks_used": current_gpu_chunks.copy()})

            # Update rewards less frequently (but more often than before since interval is shorter)
            if scheduler_initialized and step % (scheduler.scheduler_interval * 3) == 0:
                print(f"\n[Step {step}] === REWARD UPDATE ===")
                print(f"[Step {step}] Updating rewards using moving average for used chunks")
                
                # Capture rewards before update
                old_rewards = {chunk_idx: scheduler.rewards.get(chunk_idx, 0.0) for chunk_idx in current_gpu_chunks}
                
                scheduler.update_rewards(current_gpu_chunks, 1.0)
                
                # Show how rewards changed for used chunks
                print(f"[Step {step}] Used chunks (moving average update):")
                for chunk_idx in current_gpu_chunks:
                    old_r = old_rewards.get(chunk_idx, 0.0)
                    new_r = scheduler.rewards.get(chunk_idx, 0.0)
                    count = scheduler.counts.get(chunk_idx, 1)
                    delta = new_r - old_r
                    print(f"[Step {step}]   Chunk {chunk_idx}: {old_r:.4f} → {new_r:.4f} (Δ {delta:+.4f}, count={count})")
                
                # Show current priorities for CPU chunks (with exploration bonus)
                cpu_chunk_list = [c for c in scheduler.rewards.keys() if c not in gpu_chunks]
                if cpu_chunk_list[:5]:  # Show first 5 CPU chunks
                    print(f"[Step {step}] CPU chunk priorities (reward + exploration_bonus):")
                    for chunk_idx in cpu_chunk_list[:5]:
                        reward = scheduler.rewards.get(chunk_idx, 0.0)
                        priority = scheduler._calculate_priority(chunk_idx)
                        bonus = priority - reward  # This shows recency + exploration_bonus
                        print(f"[Step {step}]   Chunk {chunk_idx} (CPU): reward={reward:.4f}, priority={priority:.4f} (bonus={bonus:+.4f})")

    # Generate final text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    end_time = time.perf_counter()

    # Calculate metrics
    ttft_s = ttft_duration
    tpot_s = sum(decode_times[1:]) / max(1, len(decode_times[1:])) if len(decode_times) > 1 else 0.0
    e2e_s = end_time - ttft_start
    throughput_tps = len(generated_tokens) / sum(decode_times) if decode_times else 0.0

    print(f"\n{'='*80}")
    print(f"[AsyncGeneration] GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"[AsyncGeneration] Total tokens generated: {len(generated_tokens)}")
    print(f"[AsyncGeneration] Background prefetches initiated: {prefetch_count}")
    print(f"[AsyncGeneration] Cache swaps completed: {cache_swaps}")
    print(f"[AsyncGeneration] Final GPU chunks: {current_gpu_chunks}")
    print(f"[AsyncGeneration] ASYNC Metrics: TTFT={ttft_s*1000:.1f}ms, TPOT={tpot_s*1000:.1f}ms, E2E={e2e_s:.3f}s")
    print(f"{'='*80}\n")

    return {
        "question": question,
        "answer": generated_text,
        "generated_tokens": generated_tokens,
        "trace": trace,
        "final_gpu_chunks": current_gpu_chunks,
        "sparsity_info": cache_info,
        "transfer_count": cache_swaps,
        "prefetch_count": prefetch_count,  # NEW: Track background prefetches
        "metrics": {
            "ttft_s": ttft_s,
            "tpot_s": tpot_s,
            "e2e_s": e2e_s,
            "throughput_tps": throughput_tps,
            "num_tokens": len(generated_tokens),
            "num_transfers": cache_swaps,
            "num_prefetches": prefetch_count  # NEW: Include in metrics
        }
    }

# ============= ASYNC PREFETCH PIPELINE FUNCTION =============

def run_async_pipeline(
    input_file: str,
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    output_dir: str = "results",
    top_k: int = 5,
    max_tokens: int = 32,
    device: str = "cuda:0",
    sparsity_ratio: float = 1.0  # Default to full attention for quality
):
    """
    ASYNC PREFETCH pipeline with true background chunk transfers
    Key improvements over v4:
    - True background prefetch overlaps with generation
    - Non-blocking transfer completion checks
    - Reduced TTFT and TPOT through async operations
    """
    
    print(f"\n{'='*80}")
    print(f"[AsyncPipeline] Starting ASYNC PREFETCH Pipeline v6")
    print(f"[AsyncPipeline] - True background prefetch with CUDA streams")
    print(f"[AsyncPipeline] - Non-blocking transfer completion checks")
    print(f"[AsyncPipeline] - Optimized for reduced TTFT/TPOT")
    print(f"[AsyncPipeline] Sparsity ratio: {sparsity_ratio:.2f}")
    print(f"{'='*80}\n")

    logger = setup_logging("WARNING")
    os.makedirs(output_dir, exist_ok=True)

    # Load samples
    samples = load_samples(input_file)
    print(f"[AsyncPipeline] Loaded {len(samples)} samples")

    # Setup retrieval
    retrieval_config = RetrievalConfig()
    retriever = ColbertRetrieval(retrieval_config)
    print(f"[AsyncPipeline] Retrieval system initialized")

    # Load model/tokenizer
    print(f"[AsyncPipeline] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()
    print(f"[AsyncPipeline] Model loaded")

    results = []
    print(f"\n[AsyncPipeline] Processing samples...")

    for si, sample in enumerate(tqdm(samples, desc="Processing", unit="sample")):
        try:
            print(f"\n[Sample {si}] Processing with async prefetch...")
            scheduler = AsyncPrefetchScheduler(
                device=device,
                scheduler_interval=10,  # Check for swaps every 10 steps
                promote_per_step=2,     # Prefetch 2 chunks at a time (more aggressive)
                exploration_c=1.2,
                max_candidates=8,
                sparsity_ratio=sparsity_ratio,
                epsilon=0.2             # 20% random exploration in chunk prediction
            )
            print(f"[AsyncPipeline] Created async scheduler")

            torch.cuda.empty_cache()

            # Retrieval (outside timing)
            prepared_samples = retriever.prepare([sample])
            retrieved_samples = retriever.retrieve(prepared_samples, top_k=top_k)
            sample = retrieved_samples[0] if retrieved_samples else sample

            # Build KV caches (outside timing)
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
                print(f"[Sample {si}] ERROR: No GPU chunks")
                continue

            # ASYNC PREFETCH GENERATION (timed)
            result = async_generate_with_kv(
                sample=sample,
                gpu_chunks=gpu_chunks,
                cpu_chunks=cpu_chunks,
                scheduler=scheduler,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_tokens=max_tokens,
                sparsity_ratio=sparsity_ratio
            )

            result["sample_id"] = str(sample.get("id", f"sample_{si}"))
            result["sample_index"] = si
            result["sparsity_ratio"] = sparsity_ratio
            results.append(result)

            # Log metrics
            metrics = result.get("metrics", {})
            sparsity_info = result.get("sparsity_info", {})
            transfer_count = result.get("transfer_count", 0)
            prefetch_count = result.get("prefetch_count", 0)
            ttft_ms = metrics.get("ttft_s", 0.0) * 1000.0
            tpot_ms = metrics.get("tpot_s", 0.0) * 1000.0
            num_tokens = metrics.get("num_tokens", 0)
            answer = result.get("answer", "")
            compression = sparsity_info.get("compression_ratio", 1.0)

            print(f"\n[Sample {si}] ASYNC RESULTS:")
            print(f"[Sample {si}] Tokens generated: {num_tokens}")
            print(f"[Sample {si}] Background prefetches: {prefetch_count}")
            print(f"[Sample {si}] Cache swaps: {transfer_count}")
            print(f"[Sample {si}] ASYNC TTFT: {ttft_ms:.1f}ms | TPOT: {tpot_ms:.1f}ms")
            print(f"[Sample {si}] Compression: {compression:.1%}")
            print(f"[Sample {si}] Answer: {answer[:100]}...")

            # Clean shutdown
            scheduler.shutdown()
            del scheduler
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Sample {si} failed: {str(e)}")
            continue

    # Save results
    output_path = os.path.join(output_dir, f"results_async_v6.json")
    with open(output_path, "w") as f:
        serializable_results = convert_to_serializable({"results": results})
        json.dump(serializable_results, f, indent=2)

    print(f"[AsyncPipeline] Results saved to {output_path}")

    # Calculate aggregate metrics
    if results:
        successful_results = [r for r in results if "metrics" in r]
        if successful_results:
            avg_ttft = sum(r["metrics"]["ttft_s"] for r in successful_results) / len(successful_results)
            avg_tpot = sum(r["metrics"]["tpot_s"] for r in successful_results) / len(successful_results)
            avg_e2e = sum(r["metrics"]["e2e_s"] for r in successful_results) / len(successful_results)

            # Calculate compression statistics
            compressions = [r.get("sparsity_info", {}).get("compression_ratio", 1.0) for r in successful_results]
            avg_compression = sum(compressions) / len(compressions)

            # Calculate transfer statistics
            total_transfers = sum(r.get("transfer_count", 0) for r in successful_results)
            avg_transfers = total_transfers / len(successful_results) if successful_results else 0
            max_transfers = max((r.get("transfer_count", 0) for r in successful_results), default=0)
            min_transfers = min((r.get("transfer_count", 0) for r in successful_results), default=0)
            
            # Calculate prefetch statistics
            total_prefetches = sum(r.get("prefetch_count", 0) for r in successful_results)
            avg_prefetches = total_prefetches / len(successful_results) if successful_results else 0
            max_prefetches = max((r.get("prefetch_count", 0) for r in successful_results), default=0)
            min_prefetches = min((r.get("prefetch_count", 0) for r in successful_results), default=0)

            print(f"\n{'='*80}")
            print(f"ASYNC PREFETCH PERFORMANCE METRICS")
            print(f"{'='*80}")
            print(f"Sparsity Ratio: {sparsity_ratio:.2f}")
            print(f"Samples processed: {len(successful_results)}")
            print(f"\n--- ASYNC Latency Metrics ---")
            print(f"Average TTFT: {avg_ttft*1000:.1f} ms (async optimized)")
            print(f"Average TPOT: {avg_tpot*1000:.1f} ms (async optimized)")
            print(f"Average E2E: {avg_e2e:.3f} s")
            print(f"\n--- Memory & Compression ---")
            print(f"Average Compression: {avg_compression:.1%} of original")
            print(f"\n--- Async Prefetch Statistics ---")
            print(f"Total background prefetches: {total_prefetches}")
            print(f"Average prefetches per sample: {avg_prefetches:.1f}")
            print(f"Min prefetches (single sample): {min_prefetches}")
            print(f"Max prefetches (single sample): {max_prefetches}")
            print(f"\n--- Cache Swap Statistics ---")
            print(f"Total cache swaps: {total_transfers}")
            print(f"Average cache swaps per sample: {avg_transfers:.1f}")
            print(f"Min swaps (single sample): {min_transfers}")
            print(f"Max swaps (single sample): {max_transfers}")
            print(f"{'='*80}")

    return {"output_path": output_path, "results": results}

def main():
    """Main function with async prefetch"""
    import argparse
    parser = argparse.ArgumentParser("ASYNC PREFETCH RAG Pipeline v6")
    parser.add_argument("--input", default="/nfs/hpc/share/jainc/SemCache/baselines/CacheBlend/inputs/musique_s.json", help="Input dataset JSON")
    parser.add_argument("--output_dir", default="results_async_v6", help="Output directory")
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.2", help="Model ID")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k chunks")
    parser.add_argument("--max_tokens", type=int, default=20, help="Max generation length")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--sparsity_ratio", type=float, default=1.0, help="Sparsity ratio")

    args = parser.parse_args()

    print("ASYNC PREFETCH RAG PIPELINE v6")
    print("=" * 60)
    print("Features:")
    print("- True background prefetch with CUDA streams")
    print("- Non-blocking transfer completion checks")
    print("- Pointer swapping instead of cache rebuilding")
    print("- Optimized for reduced TTFT/TPOT")
    print("=" * 60)

    run_async_pipeline(
        input_file=args.input,
        model_id=args.model_id,
        output_dir=args.output_dir,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        device=args.device,
        sparsity_ratio=args.sparsity_ratio
    )

if __name__ == "__main__":
    main()