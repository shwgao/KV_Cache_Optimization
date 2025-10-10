#!/usr/bin/env python3

"""
FIXED Scheduler v5 - Resolves TTFT Issue

Critical fix: Removed pre-computation from initialization to avoid TTFT overhead
Pre-computation now happens in background after first token generation
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

class BanditScheduler:
    
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        promote_per_step: Optional[int] = None,
        scheduler_interval: Optional[int] = None,
        exploration_c: float = 1.0,
        max_candidates: int = 10,
        sparsity_ratio: float = 1.0,
        epsilon: float = 0.15,
    ):
        self.config_path = config_path
        self.promote_per_step = promote_per_step or 1
        self.scheduler_interval = scheduler_interval or 20  # Less frequent
        self.exploration_c = exploration_c
        self.max_candidates = max_candidates
        self.sparsity_ratio = sparsity_ratio
        self.epsilon = epsilon
        
        # Core state
        self.sample: Optional[Dict[str, Any]] = None
        self.device: Optional[torch.device] = None
        self.model = None
        self.tokenizer = None
        
        # Simplified data structures
        self.gpu_chunks: Dict[int, Any] = {}
        self.cpu_chunks: Dict[int, str] = {}
        self.ready_kv: Dict[int, Any] = {}
        
        # Simplified bandit state
        self.rewards: Dict[int, float] = {}
        self.counts: Dict[int, int] = {}
        self.last_used: Dict[int, int] = {}
        self.current_step: int = 0
        
        # GPU management
        self.max_gpu: int = 0
        self.current_gpu_order: List[int] = []
        
        # Background processing
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
        FIXED: Fast initialization without pre-computation to avoid TTFT overhead
        """
        print(f"[Scheduler] FIXED initialization starting...")
        init_start = time.perf_counter()
        
        self.sample = sample
        self.gpu_chunks = dict(gpu_chunks)
        self.cpu_chunks = dict(cpu_chunks)
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device(device)
        self.max_gpu = max_gpu if max_gpu is not None else len(gpu_chunks)
        self.current_gpu_order = list(gpu_chunks.keys())[:self.max_gpu]
        
        # Initialize rewards with simple baseline
        for idx in set(gpu_chunks.keys()) | set(cpu_chunks.keys()):
            if idx in gpu_chunks:
                self.rewards[idx] = 1.0
                self.counts[idx] = 1
            else:
                self.rewards[idx] = 0.5
                self.counts[idx] = 1
            self.last_used[idx] = 0
        
        init_time = time.perf_counter() - init_start
        print(f"[Scheduler] FIXED initialization completed in {init_time*1000:.2f}ms (no pre-computation)")
        
        # Schedule background pre-computation if needed
        if self.cpu_chunks and self.background_precompute:
            threading.Thread(target=self._background_precompute, daemon=True).start()
    
    def _background_precompute(self):
        """
        Background pre-computation that doesn't affect TTFT
        Runs in separate thread after initialization
        """
        if self.precompute_started:
            return
        
        self.precompute_started = True
        print(f"[Background] Starting pre-computation for {len(self.cpu_chunks)} CPU chunks...")
        
        # Small delay to ensure first token is generated first
        time.sleep(0.1)
        
        start_time = time.perf_counter()
        chunk_indices = [idx for idx in self.cpu_chunks.keys() if idx not in self.gpu_chunks]
        
        with torch.inference_mode():
            for idx in chunk_indices:
                try:
                    text = self.cpu_chunks.get(idx, "")
                    if text:
                        input_ids = build_chunk_sequence(text, self.tokenizer)
                        current_input = torch.tensor([input_ids], device=self.device)
                        outputs = self.model(current_input, use_cache=True, return_dict=True)
                        kv = outputs.past_key_values
                        
                        if hasattr(kv, 'to_legacy_cache'):
                            kv = kv.to_legacy_cache()
                        
                        # Keep KV on CPU
                        cpu_kv = []
                        for (k, v) in kv:
                            cpu_k = k.detach().cpu()
                            cpu_v = v.detach().cpu()
                            cpu_kv.append((cpu_k, cpu_v))
                        
                        self.ready_kv[idx] = tuple(cpu_kv)
                except Exception:
                    pass  # Skip failed chunks
        
        precompute_time = time.perf_counter() - start_time
        print(f"[Background] Pre-computation completed in {precompute_time:.3f}s, {len(self.ready_kv)} chunks ready")
    
    def predict(self, step: int, generated_tokens: List[int]) -> List[int]:
        """Simplified prediction logic"""
        self.current_step = step - 5
        
        if step < 8:
            return []
        
        candidates = self._get_candidates()
        if not candidates:
            return []
        
        # Simple scoring based on rewards and recency
        scores = {}
        for idx in candidates:
            base_score = self.rewards.get(idx, 0.0)
            recency = math.exp(-0.1 * max(0, self.current_step - self.last_used.get(idx, -100)))
            scores[idx] = base_score + 0.2 * recency
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in ranked[:self.promote_per_step]]
        return selected
    
    def schedule_to_gpu(self) -> List[int]:
        """Optimized scheduling with reduced overhead"""
        if not self.ready_kv:
            return self.current_gpu_order
        
        # Calculate priorities efficiently
        ready_priorities = []
        current_priorities = {}
        
        for idx in self.ready_kv.keys():
            if idx not in self.current_gpu_order:
                priority = self._calculate_priority(idx)
                ready_priorities.append((idx, priority))
        
        for idx in self.current_gpu_order:
            current_priorities[idx] = self._calculate_priority(idx)
        
        if not ready_priorities:
            return self.current_gpu_order
        
        ready_priorities.sort(key=lambda x: x[1], reverse=True)
        new_gpu_order = list(self.current_gpu_order)
        
        # Reduced exploration for better performance
        for idx, priority in ready_priorities[:self.promote_per_step]:
            if idx in self.ready_kv:
                force_explore = (np.random.random() < self.epsilon)
                
                if len(new_gpu_order) >= self.max_gpu:
                    lowest_idx = min(new_gpu_order, key=lambda x: current_priorities.get(x, 0))
                    lowest_priority = current_priorities.get(lowest_idx, 0)
                    
                    should_swap = (priority > lowest_priority) or force_explore
                    if should_swap:
                        if self._demote_chunk_to_cpu(lowest_idx):
                            new_gpu_order.remove(lowest_idx)
                        else:
                            continue
                
                if self._promote_chunk_to_gpu(idx):
                    new_gpu_order.append(idx)
        
        # Remove duplicates
        seen = set()
        deduped_order = []
        for idx in new_gpu_order:
            if idx not in seen:
                seen.add(idx)
                deduped_order.append(idx)
        
        self.current_gpu_order = deduped_order
        return self.current_gpu_order
    
    def _promote_chunk_to_gpu(self, idx: int) -> bool:
        """Efficient CPU->GPU transfer"""
        try:
            if idx not in self.ready_kv:
                return False
            
            cpu_kv = self.ready_kv[idx]
            gpu_kv = []
            
            # Efficient transfer
            with torch.cuda.device(self.device):
                for (k, v) in cpu_kv:
                    gpu_k = k.to(self.device, non_blocking=True)
                    gpu_v = v.to(self.device, non_blocking=True)
                    gpu_kv.append((gpu_k, gpu_v))
                torch.cuda.synchronize()  # Only sync after all transfers
            
            self.gpu_chunks[idx] = tuple(gpu_kv)
            return True
        except Exception:
            return False
    
    def _demote_chunk_to_cpu(self, idx: int) -> bool:
        """Efficient GPU->CPU transfer"""
        try:
            if idx not in self.gpu_chunks:
                return False
            
            gpu_kv = self.gpu_chunks[idx]
            cpu_kv = []
            
            for (k, v) in gpu_kv:
                cpu_k = k.detach().cpu()
                cpu_v = v.detach().cpu()
                cpu_kv.append((cpu_k, cpu_v))
            
            self.ready_kv[idx] = tuple(cpu_kv)
            del self.gpu_chunks[idx]
            torch.cuda.empty_cache()
            return True
        except Exception:
            return False
    
    def update_rewards(self, used_chunks: List[int], reward: float = 1.0) -> None:
        """Simple reward updates"""
        current_time = self.current_step
        for idx in used_chunks:
            self.last_used[idx] = current_time
            old_reward = self.rewards.get(idx, 0.0)
            old_count = self.counts.get(idx, 0)
            self.counts[idx] = old_count + 1
            alpha = 1.0 / self.counts[idx]
            self.rewards[idx] = (1 - alpha) * old_reward + alpha * reward
    
    def get_gpu_chunks(self) -> Dict[int, Any]:
        return self.gpu_chunks
    
    def get_scheduler_interval(self) -> int:
        return self.scheduler_interval
    
    def shutdown(self) -> None:
        pass
    
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
        """Get candidate chunks for scheduling"""
        candidates = list(self.ready_kv.keys())
        if len(candidates) > self.max_candidates:
            candidates = np.random.choice(candidates, self.max_candidates, replace=False).tolist()
        return candidates
    
    def _calculate_priority(self, idx: int) -> float:
        """Simplified priority calculation"""
        base_reward = self.rewards.get(idx, 0.0)
        current_time = self.current_step
        last_used = self.last_used.get(idx, -100)
        recency_factor = math.exp(-0.1 * max(0, current_time - last_used))
        exploration_bonus = 0.1 if idx not in self.gpu_chunks else 0.0
        return base_reward + 0.2 * recency_factor + exploration_bonus

class FastKVCacheManager:
    """Fast KV cache operations"""
    
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