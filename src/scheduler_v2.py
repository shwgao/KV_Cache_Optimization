#!/usr/bin/env python3

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
        exploration_c: float = 1.5,
        max_candidates: int = 50,
    ):
        self.config_path = config_path
        self.promote_per_step = promote_per_step
        self.scheduler_interval = scheduler_interval
        self.exploration_c = exploration_c
        self.max_candidates = max_candidates

        # Loaded at initialize()
        self.sample: Optional[Dict[str, Any]] = None
        self.device: Optional[torch.device] = None
        self.model = None
        self.tokenizer = None

        # Pools
        self.gpu_chunks: Dict[int, Any] = {}  # idx -> past_key_values
        self.cpu_chunks: Dict[int, str] = {}  # idx -> text

        # Staging states
        self.predictions: Dict[int, Dict[str, Any]] = {}
        self.preparing: set[int] = set()
        self.ready: set[int] = set()
        self.ready_kv: Dict[int, Any] = {}

        # Bandit stats - FIXED: Better initialization
        self.rewards: Dict[int, float] = {}
        self.counts: Dict[int, int] = {}
        self.last_scheduled_step: Dict[int, int] = {}
        self.current_step: int = 0

        # GPU plan
        self.max_gpu: int = 0
        self.current_gpu_order: List[int] = []
        self.last_predicted: List[int] = []
        self.last_promoted: List[int] = []
        self.last_demoted: List[int] = []

        # Background materializer
        self._queue: Queue[int] = Queue()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

        # Config load
        self._load_config_defaults()

    # ----------------------- public API -----------------------

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
        self.sample = sample
        self.gpu_chunks = dict(gpu_chunks)
        self.cpu_chunks = dict(cpu_chunks)
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device(device)
        self.max_gpu = max_gpu if max_gpu is not None else len(gpu_chunks)
        self.current_gpu_order = list(gpu_chunks.keys())[: self.max_gpu]

        # FIXED: Better bandit initialization with realistic rewards
        all_chunk_indices = set(gpu_chunks.keys()) | set(cpu_chunks.keys())
        for idx in all_chunk_indices:
            # Initial chunks on GPU get higher starting rewards
            if idx in gpu_chunks:
                self.rewards[idx] = 0.8  # Higher baseline for initial GPU chunks
                self.counts[idx] = 5     # Higher count to make them more stable
                self.last_scheduled_step[idx] = 0
            else:
                self.rewards[idx] = 0.1  # Lower baseline for CPU chunks
                self.counts[idx] = 1

        # Start background worker
        self._stop_event.clear()
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._materializer_loop, daemon=True)
            self._worker.start()
            print(f"Background materializer started, worker thread: {self._worker.is_alive()}")

    def set_step(self, step: int) -> None:
        self.current_step = step

    def predict(self, step: int, generated_tokens: List[int]) -> List[int]:
        """Predict chunks with better exploration"""
        self.current_step = step
        candidates = self._candidate_pool()
        
        if not candidates:
            print(f"Predict T{self.current_step}: NO CANDIDATES AVAILABLE")
            return []

        scores = self._ucb_scores(candidates)
        
        # Add randomness for exploration
        for idx in scores:
            scores[idx] += np.random.normal(0, 0.1)
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected: List[int] = []
        promote_count = int(self.promote_per_step)
        
        # Select top candidates
        for idx, score in ranked[:promote_count]:
            selected.append(idx)
        
        # FIXED: Force exploration every few steps
        if step % 4 == 0 and len(candidates) > promote_count:
            random_candidates = [idx for idx in candidates if idx not in selected]
            if random_candidates:
                random_choice = np.random.choice(random_candidates)
                if len(selected) < promote_count:
                    selected.append(random_choice)
                else:
                    selected[-1] = random_choice  # Replace last with random

        # Queue for materialization
        newly_queued = []
        already_ready = []
        
        for idx in selected:
            self._record_prediction(idx)
            
            if idx in self.ready:
                already_ready.append(idx)
            elif idx not in self.preparing and idx not in self.gpu_chunks:
                self.preparing.add(idx)
                try:
                    self._queue.put_nowait(idx)
                    newly_queued.append(idx)
                except:
                    print(f"Queue full, skipping {idx}")

        print(f"Predict T{self.current_step}: predicted={selected}, queued={newly_queued}, ready_now={already_ready}")
        print(f"  Candidates: {candidates[:10]}...")
        print(f"  UCB scores: {[(idx, f'{scores[idx]:.3f}') for idx in selected]}")
        
        self.last_predicted = list(selected)
        return selected

    def schedule_to_gpu(self) -> List[int]:
        """
        FIXED: Complete promotion/demotion logic with priority comparison
        """
        print(f"DeltaT T{self.current_step}: Starting GPU scheduling...")
        print(f"  Current GPU: {self.current_gpu_order}")
        print(f"  Ready chunks: {list(self.ready)}")
        print(f"  Preparing chunks: {list(self.preparing)}")

        if not self.ready:
            print(f"DeltaT T{self.current_step}: no ready chunks; gpu={self.current_gpu_order}")
            self.last_promoted = []
            self.last_demoted = []
            return self.current_gpu_order

        # FIXED: Score ALL chunks (current GPU + ready) for unified comparison
        all_candidates: List[Tuple[int, float, str]] = []

        # Score current GPU chunks
        for idx in self.current_gpu_order:
            priority = self._calculate_chunk_priority(idx, is_gpu=True)
            all_candidates.append((idx, priority, "gpu"))

        # Score ready chunks
        for idx in list(self.ready):
            if idx in self.ready_kv:  # Only consider chunks with materialized KV
                priority = self._calculate_chunk_priority(idx, is_gpu=False)
                all_candidates.append((idx, priority, "ready"))

        # Sort by priority (highest first)
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  All chunk priorities: {[(idx, f'{priority:.3f}', status) for idx, priority, status in all_candidates[:8]]}")

        # FIXED: Select top max_gpu chunks regardless of current status
        target_gpu_chunks = all_candidates[:self.max_gpu]
        new_gpu_order = [idx for idx, _, _ in target_gpu_chunks]
        
        # Calculate promotions and demotions
        current_gpu_set = set(self.current_gpu_order)
        new_gpu_set = set(new_gpu_order)
        
        promotions = [idx for idx in new_gpu_order if idx not in current_gpu_set]
        demotions = [idx for idx in self.current_gpu_order if idx not in new_gpu_set]

        # FIXED: Execute promotions
        promoted_count = 0
        for idx in promotions:
            if idx in self.ready_kv:
                try:
                    # Move from ready to GPU
                    self.gpu_chunks[idx] = self.ready_kv[idx]
                    self.last_scheduled_step[idx] = self.current_step
                    
                    # Clean up staging
                    self.ready.discard(idx)
                    del self.ready_kv[idx]
                    
                    promoted_count += 1
                    priority = self._calculate_chunk_priority(idx, is_gpu=False)
                    print(f"Promoted chunk {idx} (priority: {priority:.3f})")
                    
                except Exception as e:
                    print(f"Failed to promote chunk {idx}: {e}")

        # FIXED: Execute demotions
        demoted_count = 0
        for idx in demotions:
            if idx in self.gpu_chunks:
                try:
                    # Move from GPU to CPU
                    # Note: In a real implementation, you might want to keep the KV cache on CPU
                    # For now, we'll just remove it since we have the original text
                    del self.gpu_chunks[idx]
                    
                    demoted_count += 1
                    priority = self._calculate_chunk_priority(idx, is_gpu=True)
                    print(f"Demoted chunk {idx} (priority: {priority:.3f})")
                    
                except Exception as e:
                    print(f"Failed to demote chunk {idx}: {e}")

        # Update state
        self.current_gpu_order = new_gpu_order
        self.last_promoted = promotions
        self.last_demoted = demotions

        if promotions or demotions:
            print(f"DeltaT T{self.current_step}: +{promoted_count} promoted, -{demoted_count} demoted")
            print(f"  Promoted: {promotions}")
            print(f"  Demoted: {demotions}")
            print(f"  New GPU: {self.current_gpu_order}")
        else:
            print(f"DeltaT T{self.current_step}: no changes; gpu={self.current_gpu_order}")
            
        return self.current_gpu_order

    def get_gpu_chunks(self) -> Dict[int, Any]:
        return self.gpu_chunks

    def get_scheduler_interval(self) -> int:
        return int(self.scheduler_interval)

    def update_rewards(self, used_chunks: List[int], reward: float = 1.0) -> None:
        """FIXED: Better reward updating with step bonus"""
        for idx in used_chunks:
            # Exponential moving average for rewards
            current_reward = self.rewards.get(idx, 0.0)
            # FIXED: Higher learning rate for more responsive updates
            self.rewards[idx] = 0.7 * current_reward + 0.3 * reward
            self.counts[idx] = self.counts.get(idx, 0) + 1
            
            # Bonus for chunks that were recently promoted
            if idx in self.last_promoted:
                self.rewards[idx] += 0.1  # Promotion bonus

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(-1)  # sentinel
        except:
            pass
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None

    # ----------------------- FIXED internals -----------------------

    def _calculate_chunk_priority(self, idx: int, is_gpu: bool) -> float:
        """
        FIXED: Unified priority calculation for both GPU and ready chunks
        """
        # Base bandit score
        avg_reward = self._avg_reward(idx)
        
        # Recency bonus
        last_step = self.last_scheduled_step.get(idx, -1)
        recency = 0.0
        if last_step >= 0:
            steps_ago = self.current_step - last_step
            recency = max(0.0, 0.3 * math.exp(-steps_ago / 5.0))
        
        # Usage frequency bonus
        usage_count = self.counts.get(idx, 1)
        frequency_bonus = min(0.2, 0.02 * usage_count)
        
        # Prediction recency bonus
        if idx in self.predictions:
            pred_step = self.predictions[idx].get("last_pred_step", -1)
            if pred_step >= 0:
                pred_recency = max(0.0, 0.15 * math.exp(-(self.current_step - pred_step) / 3.0))
                frequency_bonus += pred_recency
        
        # FIXED: Status-specific bonuses
        if is_gpu:
            # GPU chunks get small stability bonus but decay over time
            gpu_bonus = 0.05 * math.exp(-max(0, self.current_step - last_step) / 8.0)
        else:
            # Ready chunks get novelty bonus
            gpu_bonus = 0.1  # Novelty bonus for new chunks
        
        priority = avg_reward + recency + frequency_bonus + gpu_bonus
        
        # Add small random component for tie-breaking
        priority += np.random.uniform(-0.01, 0.01)
        
        return priority

    def _load_config_defaults(self) -> None:
        try:
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            sch = cfg.get("scheduler", {})
            if self.promote_per_step is None:
                self.promote_per_step = int(sch.get("promote_per_step", 2))
            if self.scheduler_interval is None:
                self.scheduler_interval = int(sch.get("scheduler_interval", 5))
        except Exception as e:
            print(f"Config load warning: {e}")
            if self.promote_per_step is None:
                self.promote_per_step = 2
            if self.scheduler_interval is None:
                self.scheduler_interval = 5

    def _candidate_pool(self) -> List[int]:
        """Better candidate selection logic"""
        cpu_candidates = [
            idx for idx in self.cpu_chunks.keys() 
            if idx not in self.preparing
        ]
        
        # Include GPU chunks for re-evaluation
        gpu_candidates = list(self.gpu_chunks.keys())
        
        all_candidates = cpu_candidates + gpu_candidates
        np.random.shuffle(all_candidates)
        
        candidates = all_candidates[:self.max_candidates]
        
        print(f"  Candidate pool: {len(cpu_candidates)} CPU + {len(gpu_candidates)} GPU = {len(candidates)} total")
        
        return candidates

    def _avg_reward(self, idx: int) -> float:
        """Better average reward calculation"""
        count = self.counts.get(idx, 1)
        reward = self.rewards.get(idx, 0.0)
        return reward / count

    def _ucb_scores(self, candidates: List[int]) -> Dict[int, float]:
        """Better UCB scoring"""
        total_steps = max(1, sum(self.counts.values()))
        scores: Dict[int, float] = {}

        for idx in candidates:
            avg_reward = self._avg_reward(idx)
            count = max(1, self.counts.get(idx, 1))
            confidence = math.sqrt(2.0 * math.log(total_steps + 1) / count)
            
            ucb_score = avg_reward + self.exploration_c * confidence
            
            # Additional bonuses
            if idx not in self.predictions:
                ucb_score += 0.2  # Novelty bonus
            
            if idx not in self.gpu_chunks:
                ucb_score += 0.15  # CPU exploration bonus
                
            if idx in self.preparing:
                ucb_score -= 0.05  # Avoid duplication

            scores[idx] = ucb_score

        return scores

    def _record_prediction(self, idx: int) -> None:
        """Better prediction recording"""
        entry = self.predictions.get(idx, {"priority": 0.0, "last_pred_step": -1, "count": 0})
        entry["last_pred_step"] = self.current_step
        entry["count"] = int(entry["count"]) + 1
        
        avg_reward = self._avg_reward(idx)
        entry["priority"] = avg_reward
        
        self.predictions[idx] = entry

    def _materializer_loop(self) -> None:
        """Background materialization with better error handling"""
        print("Background materializer loop started")
        
        while not self._stop_event.is_set():
            try:
                idx = self._queue.get(timeout=0.25)
            except Empty:
                continue

            if idx == -1:  # Shutdown signal
                break

            try:
                print(f"  Materializing chunk {idx}...")
                
                if idx in self.ready_kv or idx in self.gpu_chunks:
                    self.preparing.discard(idx)
                    continue

                text = self.cpu_chunks.get(idx)
                if text is None:
                    print(f"  Chunk {idx} text not found")
                    self.preparing.discard(idx)
                    continue

                # Materialize KV cache
                input_ids = build_chunk_sequence(text, self.tokenizer)
                current_input = torch.tensor([input_ids], device=self.device)

                with torch.inference_mode():
                    outputs = self.model(current_input, use_cache=True, return_dict=True)
                    kv = outputs.past_key_values
                    
                    if hasattr(kv, 'to_legacy_cache'):
                        kv = kv.to_legacy_cache()
                    
                    self.ready_kv[idx] = kv
                    self.ready.add(idx)
                    
                    print(f"  Chunk {idx} materialized and ready (KV shape: {[k[0].shape for k in kv[:2]]})")

            except Exception as e:
                print(f"  Materialization failed for chunk {idx}: {e}")
                # Give small reward to avoid getting stuck
                self.rewards[idx] = self.rewards.get(idx, 0.0) - 0.05
            finally:
                self.preparing.discard(idx)
                
        print("Background materializer loop ended")