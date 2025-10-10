#!/usr/bin/env python3
"""
ASYNC PREFETCH SCHEDULER v6 - True background prefetch with CUDA streams
This scheduler implements true overlapping of chunk transfers with generation
Key improvements:
- Background prefetch of full KV caches using CUDA streams
- Non-blocking transfer completion checks
- Pointer swapping instead of cache rebuilding
"""

import time
import math
import threading
from queue import Queue, Empty
from typing import Dict, List, Optional, Any, Tuple
import yaml
import torch
import numpy as np
from build_kv_v2 import build_chunk_sequence

class AsyncPrefetchScheduler:
    def __init__(
        self,
        device: str,
        config_path: str = "configs/config.yaml",
        promote_per_step: Optional[int] = None,
        scheduler_interval: Optional[int] = None,
        exploration_c: float = 1.0,
        max_candidates: int = 10,
        sparsity_ratio: float = 1.0,
        epsilon: float = 0.15,
    ):
        self.device = torch.device(device)
        self.config_path = config_path
        self.promote_per_step = promote_per_step or 1
        self.scheduler_interval = scheduler_interval or 5  # Less frequent to reduce overhead
        self.exploration_c = exploration_c
        self.max_candidates = max_candidates
        self.sparsity_ratio = sparsity_ratio
        self.epsilon = epsilon

        # Core state
        self.sample: Optional[Dict[str, Any]] = None
        self.model = None
        self.tokenizer = None

        # Chunk management
        self.gpu_chunks: Dict[int, Any] = {}
        self.cpu_chunks: Dict[int, str] = {}
        self.ready_kv: Dict[int, Any] = {}

        # Bandit state - simplified
        self.rewards: Dict[int, float] = {}
        self.counts: Dict[int, int] = {}
        self.last_used: Dict[int, int] = {}
        self.current_step: int = 0

        # GPU management
        self.max_gpu: int = 0
        self.current_gpu_order: List[int] = []

        # ASYNC PREFETCH INFRASTRUCTURE
        self.transfer_stream = torch.cuda.Stream(self.device)  # Dedicated stream for transfers
        self.prefetch_lock = threading.Lock()
        self.prefetched: Dict[int, Tuple[tuple, torch.cuda.Event]] = {}  # chunk_idx -> (gpu_kv, event)
        self.next_prefetch_steps = set()  # Steps when to trigger prefetch
        self.pending_prefetch = set()  # Chunks waiting for precomputation to finish
        
        # Background processing state
        self.background_precompute = True
        self.precompute_started = False
        
        self._load_config_defaults()

    def get_sparsity_config(self) -> Dict[str, Any]:
        """Get current sparsity configuration"""
        return {
            'ratio': self.sparsity_ratio,
            'enabled': self.sparsity_ratio < 1.0
        }

    def initialize(
        self,
        sample: Dict[str, Any],
        gpu_chunks: Dict[int, Any],
        cpu_chunks: Dict[int, str],
        tokenizer: Any,
        model: Any,
        device: str,
        max_gpu: Optional[int] = None,
    ) -> None:
        """
        Fast initialization - no pre-computation during TTFT
        """
        print(f"[AsyncScheduler] Fast initialization starting...")
        init_start = time.perf_counter()
        
        self.sample = sample
        self.gpu_chunks = dict(gpu_chunks)
        self.cpu_chunks = dict(cpu_chunks)
        self.tokenizer = tokenizer
        self.model = model
        self.max_gpu = max_gpu if max_gpu is not None else len(gpu_chunks)
        self.current_gpu_order = list(gpu_chunks.keys())[:self.max_gpu]

        # Initialize rewards: CPU chunks start higher to encourage exploration
        for idx in set(gpu_chunks.keys()) | set(cpu_chunks.keys()):
            if idx in gpu_chunks:
                self.rewards[idx] = 1.0
                self.counts[idx] = 1
            else:
                # CPU chunks start at 0.8 (not 0.5) to be competitive
                self.rewards[idx] = 0.8
                self.counts[idx] = 1
            self.last_used[idx] = 0

        # Initialize prefetch schedule - first prediction at step 5
        self.next_prefetch_steps = {5}
        
        init_time = time.perf_counter() - init_start
        print(f"[AsyncScheduler] Initialization completed in {init_time*1000:.2f}ms")
        print(f"[AsyncScheduler] First prediction scheduled at step 5, then every {self.scheduler_interval} steps")

        # Start background pre-computation if needed
        if self.cpu_chunks and self.background_precompute:
            threading.Thread(target=self._background_precompute, daemon=True).start()

    def _background_precompute(self):
        """
        Background pre-computation that doesn't affect TTFT
        Precomputes KV caches for CPU chunks
        """
        if self.precompute_started:
            return
        self.precompute_started = True
        
        print(f"[Background] Starting pre-computation for {len(self.cpu_chunks)} CPU chunks...")
        time.sleep(0.01)  # Minimal delay to ensure first token is generated first
        
        start_time = time.perf_counter()
        chunk_indices = [idx for idx in self.cpu_chunks.keys() if idx not in self.gpu_chunks]
        
        with torch.inference_mode():
            for i, idx in enumerate(chunk_indices):
                try:
                    text = self.cpu_chunks.get(idx, "")
                    if text:
                        input_ids = build_chunk_sequence(text, self.tokenizer)
                        current_input = torch.tensor([input_ids], device=self.device)
                        outputs = self.model(current_input, use_cache=True, return_dict=True)
                        kv = outputs.past_key_values
                        
                        if hasattr(kv, 'to_legacy_cache'):
                            kv = kv.to_legacy_cache()
                        
                        # Keep KV on CPU initially
                        cpu_kv = []
                        for (k, v) in kv:
                            cpu_k = k.detach().cpu()
                            cpu_v = v.detach().cpu()
                            cpu_kv.append((cpu_k, cpu_v))
                        
                        self.ready_kv[idx] = tuple(cpu_kv)
                        # Progress update every 2 chunks
                        if (i + 1) % 2 == 0:
                            print(f"[Background] Precomputed {i + 1}/{len(chunk_indices)} chunks...")
                except Exception:
                    pass  # Skip failed chunks
        
        precompute_time = time.perf_counter() - start_time
        print(f"[Background] Pre-computation completed in {precompute_time:.3f}s, {len(self.ready_kv)} chunks ready")

    def _calculate_priority(self, idx: int) -> float:
        """
        Calculate priority with exploration bonus for CPU chunks
        CPU chunks need higher bonus to overcome GPU's higher initial reward
        """
        base_reward = self.rewards.get(idx, 0.0)
        current_time = self.current_step
        last_used = self.last_used.get(idx, -100)
        recency_factor = math.exp(-0.1 * max(0, current_time - last_used))
        
        # EXPLORATION BONUS: CPU chunks get +0.35 bonus to compete with GPU chunks
        # This ensures CPU priority > GPU priority initially
        exploration_bonus = 0.35 if idx not in self.gpu_chunks else 0.0
        
        return base_reward + 0.2 * recency_factor + exploration_bonus
    
    def predict_chunks(self, step: int, generated_tokens: List[int]) -> List[int]:
        """
        Predict which chunks will be needed for future steps
        Uses epsilon-exploration + exploration bonus for CPU chunks
        """
        self.current_step = step - 5
        
        if step < 8:
            return []
        
        candidates = self._get_candidates()
        if not candidates:
            print(f"[Scheduler] WARNING: No candidates available at step {step}")
            return []
        
        # Debug: show candidates
        ready_count = len([c for c in candidates if c in self.ready_kv])
        not_ready_count = len(candidates) - ready_count
        if not_ready_count > 0:
            print(f"[Scheduler] Candidates: {len(candidates)} total ({ready_count} precomputed, {not_ready_count} pending)")

        # EPSILON-EXPLORATION: occasionally explore random chunks
        if np.random.random() < self.epsilon:
            # Exploration: randomly sample chunks to try
            num_to_select = min(self.promote_per_step, len(candidates))
            selected = np.random.choice(candidates, num_to_select, replace=False).tolist()
            print(f"[Scheduler] EXPLORATION: Randomly selected chunks {selected}")
            return selected
        
        # EXPLOITATION: use priority-based scoring with exploration bonus
        scores = {}
        for idx in candidates:
            scores[idx] = self._calculate_priority(idx)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in ranked[:self.promote_per_step]]
        
        return selected

    def prefetch_chunks_async(self, chunk_idx: int):
        """
        ASYNC PREFETCH: Transfer CPU KV cache to GPU in background
        This runs in a separate thread and uses a dedicated CUDA stream
        """
        # If chunk not precomputed yet, add to pending queue
        if chunk_idx not in self.ready_kv:
            self.pending_prefetch.add(chunk_idx)
            print(f"[AsyncPrefetch] Chunk {chunk_idx} queued (waiting for precomputation)")
            return
        
        # If already prefetched or prefetching, skip
        with self.prefetch_lock:
            if chunk_idx in self.prefetched:
                return

        def _transfer_worker():
            try:
                cpu_kv = self.ready_kv[chunk_idx]
                
                # Use dedicated transfer stream for non-blocking transfer
                with torch.cuda.stream(self.transfer_stream):
                    gpu_kv = []
                    for k, v in cpu_kv:
                        gpu_k = k.to(self.device, non_blocking=True)
                        gpu_v = v.to(self.device, non_blocking=True)
                        gpu_kv.append((gpu_k, gpu_v))
                    
                    # Record event when transfer is complete
                    event = torch.cuda.Event()
                    event.record(self.transfer_stream)
                
                # Store the transferred KV cache with completion event
                with self.prefetch_lock:
                    self.prefetched[chunk_idx] = (tuple(gpu_kv), event)
                    print(f"[AsyncPrefetch] Chunk {chunk_idx} transferred to GPU (background)")
                
            except Exception as e:
                print(f"[AsyncPrefetch] Transfer failed for chunk {chunk_idx}: {e}")

        # Launch transfer in background thread
        threading.Thread(target=_transfer_worker, daemon=True).start()
    
    def retry_pending_prefetch(self):
        """
        Retry prefetching chunks that were pending (waiting for precomputation)
        Call this periodically to check if pending chunks are now ready
        """
        if not self.pending_prefetch:
            return
        
        # Check which pending chunks are now ready
        ready_to_prefetch = []
        for chunk_idx in list(self.pending_prefetch):
            if chunk_idx in self.ready_kv:
                ready_to_prefetch.append(chunk_idx)
                self.pending_prefetch.discard(chunk_idx)
        
        # Prefetch them
        if ready_to_prefetch:
            print(f"[AsyncPrefetch] Retrying {len(ready_to_prefetch)} pending chunks: {ready_to_prefetch}")
            for chunk_idx in ready_to_prefetch:
                self.prefetch_chunks_async(chunk_idx)

    def schedule_if_needed(self, step: int, generated_tokens: List[int]):
        """
        Check if it's time to predict and prefetch next chunks
        This is called every generation step but only acts at intervals
        """
        if step in self.next_prefetch_steps:
            # Predict chunks needed for future steps
            future_step = step + self.scheduler_interval
            predicted_chunks = self.predict_chunks(future_step, generated_tokens)
            
            # Launch async prefetch for predicted chunks
            # Note: prefetch_chunks_async handles both ready and not-yet-ready chunks
            for chunk_idx in predicted_chunks:
                if chunk_idx not in self.prefetched and chunk_idx not in self.gpu_chunks:
                    self.prefetch_chunks_async(chunk_idx)
            
            # Schedule next prefetch
            self.next_prefetch_steps = {step + self.scheduler_interval}

    def get_ready_kv(self, chunk_idx: int) -> Optional[tuple]:
        """
        NON-BLOCKING check: return GPU KV if transfer is complete
        This allows generation to continue while transfers happen in background
        """
        with self.prefetch_lock:
            if chunk_idx in self.prefetched:
                gpu_kv, event = self.prefetched[chunk_idx]
                # Non-blocking check if transfer is complete
                if event.query():  # Returns True if event is complete
                    del self.prefetched[chunk_idx]
                    return gpu_kv
        return None

    def get_available_chunks(self) -> List[int]:
        """
        Get list of chunks that are ready to use (either on GPU or transferred)
        """
        available = list(self.gpu_chunks.keys())
        
        with self.prefetch_lock:
            for chunk_idx in self.prefetched:
                gpu_kv, event = self.prefetched[chunk_idx]
                if event.query():  # Transfer complete
                    available.append(chunk_idx)
        
        return available

    def force_sync_chunk(self, chunk_idx: int) -> Optional[tuple]:
        """
        BLOCKING wait for a specific chunk if needed
        Only use this when absolutely necessary
        """
        with self.prefetch_lock:
            if chunk_idx in self.prefetched:
                gpu_kv, event = self.prefetched[chunk_idx]
                event.synchronize()  # Wait for transfer to complete
                del self.prefetched[chunk_idx]
                return gpu_kv
        return None

    def update_rewards(self, used_chunks: List[int], reward: float = 1.0) -> None:
        """
        Reward updates using moving average (like scheduler_v4)
        Moving average naturally prevents rewards from getting stuck at 1.0
        """
        current_time = self.current_step
        
        # Update rewards for used chunks using moving average
        for idx in used_chunks:
            self.last_used[idx] = current_time
            old_reward = self.rewards.get(idx, 0.0)
            old_count = self.counts.get(idx, 0)
            self.counts[idx] = old_count + 1
            alpha = 1.0 / self.counts[idx]  # Decreases over time (moving average)
            self.rewards[idx] = (1 - alpha) * old_reward + alpha * reward

    def get_gpu_chunks(self) -> Dict[int, Any]:
        """Get current GPU chunks including recently transferred ones"""
        current_gpu = dict(self.gpu_chunks)
        
        # Add completed transfers
        with self.prefetch_lock:
            for chunk_idx in list(self.prefetched.keys()):
                gpu_kv, event = self.prefetched[chunk_idx]
                if event.query():
                    current_gpu[chunk_idx] = gpu_kv
                    del self.prefetched[chunk_idx]
        
        return current_gpu

    def get_scheduler_interval(self) -> int:
        return self.scheduler_interval

    def shutdown(self) -> None:
        """Clean shutdown - wait for pending transfers"""
        print("[AsyncScheduler] Shutting down...")
        
        # Wait for any pending transfers
        with self.prefetch_lock:
            for chunk_idx, (gpu_kv, event) in self.prefetched.items():
                event.synchronize()
        
        # Clear state
        self.prefetched.clear()
        print("[AsyncScheduler] Shutdown complete")

    def _load_config_defaults(self) -> None:
        """Load config with optimized defaults"""
        try:
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            sch = cfg.get("scheduler", {})
            if "promote_per_step" in sch:
                self.promote_per_step = int(sch["promote_per_step"])
            if "scheduler_interval" in sch:
                self.scheduler_interval = int(sch["scheduler_interval"])
        except:
            pass

    def _get_candidates(self) -> List[int]:
        """
        Get candidate chunks for scheduling
        Includes both ready_kv (precomputed) and cpu_chunks (not yet precomputed)
        """
        # Include both precomputed chunks AND chunks we know about from CPU
        candidates = list(set(self.ready_kv.keys()) | set(self.cpu_chunks.keys()))
        
        if len(candidates) > self.max_candidates:
            candidates = np.random.choice(candidates, self.max_candidates, replace=False).tolist()
        return candidates

# Maintain backward compatibility with original scheduler name
BanditScheduler = AsyncPrefetchScheduler

class FastKVCacheManager:
    """Fast KV cache operations with async support"""

    @staticmethod
    def fast_concatenate_chunks(gpu_chunks: Dict[int, Any], selected_chunks: List[int]) -> Optional[Any]:
        """Fast KV cache concatenation"""
        if not selected_chunks or not gpu_chunks:
            return None

        valid_caches = []
        for chunk_idx in selected_chunks:
            if chunk_idx in gpu_chunks:
                valid_caches.append(gpu_chunks[chunk_idx])

        if not valid_caches:
            return None

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

        return tuple(combined_kv)