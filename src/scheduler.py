#!/usr/bin/env python3
"""
Dynamic Scheduler for Chunk Management
Manages swapping chunks between CPU and GPU based on predictions and usage patterns
"""

import torch
import time
import threading
from typing import List, Dict, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

@dataclass
class SwapOperation:
    """Represents a chunk swap operation"""
    chunk_id: str
    source: str  # "gpu" or "cpu"
    destination: str  # "gpu" or "cpu"
    priority: float
    estimated_time: float
    size_bytes: int
    reason: str

@dataclass
class SchedulerMetrics:
    """Metrics for scheduler performance"""
    total_swaps: int
    gpu_to_cpu_swaps: int
    cpu_to_gpu_swaps: int
    average_swap_time: float
    cache_hit_rate: float
    prediction_accuracy: float
    memory_efficiency: float

class DynamicScheduler:
    """Dynamic scheduler for managing chunk placement between CPU and GPU"""
    
    def __init__(
        self,
        kv_cache_manager,
        speculative_decoder,
        max_concurrent_swaps: int = 2,
        swap_threshold: float = 0.5,
        prediction_weight: float = 0.7,
        usage_weight: float = 0.3,
        device: str = "cuda"
    ):
        self.kv_cache_manager = kv_cache_manager
        self.speculative_decoder = speculative_decoder
        self.max_concurrent_swaps = max_concurrent_swaps
        self.swap_threshold = swap_threshold
        self.prediction_weight = prediction_weight
        self.usage_weight = usage_weight
        self.device = device
        
        # Scheduler state
        self.swap_queue = []
        self.active_swaps = set()
        self.swap_history = deque(maxlen=1000)
        
        # Metrics
        self.metrics = SchedulerMetrics(
            total_swaps=0,
            gpu_to_cpu_swaps=0,
            cpu_to_gpu_swaps=0,
            average_swap_time=0.0,
            cache_hit_rate=0.0,
            prediction_accuracy=0.0,
            memory_efficiency=0.0
        )
        
        # Threading
        self.scheduler_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Initialize background scheduler
        self._start_scheduler()
    
    def _start_scheduler(self):
        """Start the background scheduler thread"""
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Dynamic scheduler started")
    
    def _scheduler_loop(self):
        """Main scheduler loop running in background"""
        while self.running:
            try:
                # Process swap queue
                self._process_swap_queue()
                
                # Update chunk priorities
                self._update_chunk_priorities()
                
                # Sleep for a short interval
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(1.0)
    
    def _process_swap_queue(self):
        """Process the swap queue and execute swaps"""
        with self.lock:
            # Sort queue by priority
            heapq.heapify(self.swap_queue)
            
            # Execute swaps up to max_concurrent_swaps
            while self.swap_queue and len(self.active_swaps) < self.max_concurrent_swaps:
                swap_op = heapq.heappop(self.swap_queue)
                
                # Check if swap is still valid
                if self._is_swap_valid(swap_op):
                    self._execute_swap(swap_op)
    
    def _is_swap_valid(self, swap_op: SwapOperation) -> bool:
        """Check if a swap operation is still valid"""
        # Check if chunk still exists in source location
        if swap_op.source == "gpu":
            if swap_op.chunk_id not in self.kv_cache_manager.gpu_cache:
                return False
        else:
            if swap_op.chunk_id not in self.kv_cache_manager.cpu_cache:
                return False
        
        # Check if destination has space
        if swap_op.destination == "gpu":
            if len(self.kv_cache_manager.gpu_cache) >= self.kv_cache_manager.max_gpu_chunks:
                return False
        else:
            if len(self.kv_cache_manager.cpu_cache) >= self.kv_cache_manager.max_cpu_chunks:
                return False
        
        return True
    
    def _execute_swap(self, swap_op: SwapOperation):
        """Execute a swap operation"""
        start_time = time.time()
        
        try:
            # Mark swap as active
            self.active_swaps.add(swap_op.chunk_id)
            
            # Perform the swap
            if swap_op.source == "gpu" and swap_op.destination == "cpu":
                self._swap_gpu_to_cpu(swap_op)
                self.metrics.gpu_to_cpu_swaps += 1
            elif swap_op.source == "cpu" and swap_op.destination == "gpu":
                self._swap_cpu_to_gpu(swap_op)
                self.metrics.cpu_to_gpu_swaps += 1
            
            # Record swap completion
            swap_time = time.time() - start_time
            self.swap_history.append({
                "chunk_id": swap_op.chunk_id,
                "source": swap_op.source,
                "destination": swap_op.destination,
                "time": swap_time,
                "reason": swap_op.reason
            })
            
            self.metrics.total_swaps += 1
            self._update_average_swap_time(swap_time)
            
            logger.info(f"Swap completed: {swap_op.chunk_id} {swap_op.source}->{swap_op.destination} "
                       f"({swap_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"Swap failed for {swap_op.chunk_id}: {e}")
        finally:
            # Remove from active swaps
            self.active_swaps.discard(swap_op.chunk_id)
    
    def _swap_gpu_to_cpu(self, swap_op: SwapOperation):
        """Swap a chunk from GPU to CPU"""
        entry = self.kv_cache_manager.gpu_cache[swap_op.chunk_id]
        
        # Move to CPU
        entry.keys = entry.keys.cpu()
        entry.values = entry.values.cpu()
        entry.metadata.is_on_gpu = False
        
        # Update cache
        self.kv_cache_manager.cpu_cache[swap_op.chunk_id] = entry
        self.kv_cache_manager.gpu_memory_used -= entry.metadata.size_bytes
        self.kv_cache_manager.cpu_memory_used += entry.metadata.size_bytes
        
        # Remove from GPU cache
        del self.kv_cache_manager.gpu_cache[swap_op.chunk_id]
    
    def _swap_cpu_to_gpu(self, swap_op: SwapOperation):
        """Swap a chunk from CPU to GPU"""
        entry = self.kv_cache_manager.cpu_cache[swap_op.chunk_id]
        
        # Move to GPU
        entry.keys = entry.keys.to(self.device)
        entry.values = entry.values.to(self.device)
        entry.metadata.is_on_gpu = True
        
        # Update cache
        self.kv_cache_manager.gpu_cache[swap_op.chunk_id] = entry
        self.kv_cache_manager.gpu_memory_used += entry.metadata.size_bytes
        self.kv_cache_manager.cpu_memory_used -= entry.metadata.size_bytes
        
        # Remove from CPU cache
        del self.kv_cache_manager.cpu_cache[swap_op.chunk_id]
    
    def _update_chunk_priorities(self):
        """Update priorities for all chunks based on predictions and usage"""
        # Get predictions from speculative decoder
        current_context = self._get_current_context()
        available_chunks = list(self.kv_cache_manager.gpu_cache.keys()) + \
                          list(self.kv_cache_manager.cpu_cache.keys())
        
        # Get chunk embeddings (simplified)
        chunk_embeddings = {chunk_id: None for chunk_id in available_chunks}
        
        predictions = self.speculative_decoder.predict_future_chunks(
            current_context, available_chunks, chunk_embeddings
        )
        
        # Create prediction map
        prediction_map = {pred.chunk_id: pred for pred in predictions}
        
        # Update priorities for all chunks
        for chunk_id in available_chunks:
            priority = self._calculate_chunk_priority(chunk_id, prediction_map)
            self._schedule_swap_if_needed(chunk_id, priority)
    
    def _get_current_context(self):
        """Get current generation context for predictions"""
        # This would be updated by the main pipeline
        # For now, return a placeholder
        from .speculative_decoder import SpeculativeContext
        
        return SpeculativeContext(
            current_tokens=[],
            generated_text="",
            chunk_history=[],
            attention_patterns=torch.randn(1, 32, 64, 64),  # Placeholder
            confidence_scores=torch.randn(1, 64)
        )
    
    def _calculate_chunk_priority(self, chunk_id: str, prediction_map: Dict) -> float:
        """Calculate priority for a chunk based on predictions and usage"""
        priority = 0.0
        
        # Prediction-based priority
        if chunk_id in prediction_map:
            pred = prediction_map[chunk_id]
            priority += self.prediction_weight * pred.probability
        
        # Usage-based priority
        usage_priority = self._calculate_usage_priority(chunk_id)
        priority += self.usage_weight * usage_priority
        
        return priority
    
    def _calculate_usage_priority(self, chunk_id: str) -> float:
        """Calculate priority based on chunk usage patterns"""
        # Get chunk metadata
        entry = None
        if chunk_id in self.kv_cache_manager.gpu_cache:
            entry = self.kv_cache_manager.gpu_cache[chunk_id]
        elif chunk_id in self.kv_cache_manager.cpu_cache:
            entry = self.kv_cache_manager.cpu_cache[chunk_id]
        
        if entry is None:
            return 0.0
        
        # Calculate priority based on access count and recency
        access_count = entry.metadata.access_count
        time_since_access = time.time() - entry.metadata.last_access_time
        
        # Higher priority for frequently accessed chunks
        frequency_score = min(1.0, access_count / 10.0)
        
        # Higher priority for recently accessed chunks
        recency_score = max(0.0, 1.0 - time_since_access / 60.0)  # Decay over 1 minute
        
        return (frequency_score + recency_score) / 2.0
    
    def _schedule_swap_if_needed(self, chunk_id: str, priority: float):
        """Schedule a swap if the chunk priority suggests it should be moved"""
        # Determine current location
        current_location = None
        if chunk_id in self.kv_cache_manager.gpu_cache:
            current_location = "gpu"
        elif chunk_id in self.kv_cache_manager.cpu_cache:
            current_location = "cpu"
        else:
            return
        
        # Determine target location based on priority
        target_location = "gpu" if priority > self.swap_threshold else "cpu"
        
        # Schedule swap if needed
        if current_location != target_location:
            self._schedule_swap(chunk_id, current_location, target_location, priority)
    
    def _schedule_swap(self, chunk_id: str, source: str, destination: str, priority: float):
        """Schedule a swap operation"""
        # Get chunk size
        entry = None
        if source == "gpu":
            entry = self.kv_cache_manager.gpu_cache.get(chunk_id)
        else:
            entry = self.kv_cache_manager.cpu_cache.get(chunk_id)
        
        if entry is None:
            return
        
        # Create swap operation
        swap_op = SwapOperation(
            chunk_id=chunk_id,
            source=source,
            destination=destination,
            priority=-priority,  # Negative for max-heap
            estimated_time=time.time(),
            size_bytes=entry.metadata.size_bytes,
            reason=f"Priority {priority:.3f}"
        )
        
        # Add to queue if not already scheduled
        with self.lock:
            # Check if already in queue
            for existing_op in self.swap_queue:
                if existing_op.chunk_id == chunk_id:
                    return  # Already scheduled
            
            heapq.heappush(self.swap_queue, swap_op)
            logger.debug(f"Scheduled swap: {chunk_id} {source}->{destination} (priority: {priority:.3f})")
    
    def _update_average_swap_time(self, new_time: float):
        """Update average swap time"""
        if self.metrics.total_swaps == 1:
            self.metrics.average_swap_time = new_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_swap_time = (
                alpha * new_time + (1 - alpha) * self.metrics.average_swap_time
            )
    
    def update_cache_hit_rate(self, hit_rate: float):
        """Update cache hit rate metric"""
        self.metrics.cache_hit_rate = hit_rate
    
    def update_prediction_accuracy(self, accuracy: float):
        """Update prediction accuracy metric"""
        self.metrics.prediction_accuracy = accuracy
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            "metrics": self.metrics.__dict__,
            "queue_size": len(self.swap_queue),
            "active_swaps": len(self.active_swaps),
            "recent_swaps": list(self.swap_history)[-10:],  # Last 10 swaps
            "gpu_chunks": len(self.kv_cache_manager.gpu_cache),
            "cpu_chunks": len(self.kv_cache_manager.cpu_cache),
            "gpu_memory_used_gb": self.kv_cache_manager.gpu_memory_used / (1024**3),
            "cpu_memory_used_gb": self.kv_cache_manager.cpu_memory_used / (1024**3)
        }
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        logger.info("Dynamic scheduler stopped")
    
    def clear_queue(self):
        """Clear the swap queue"""
        with self.lock:
            self.swap_queue.clear()
        logger.info("Swap queue cleared")


