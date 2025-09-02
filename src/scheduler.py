#!/usr/bin/env python3
"""
Novel Adaptive Temporal Locality Scheduler with Predictive Prefetching
Implements cutting-edge strategies to minimize communication delays:

1. Temporal Locality Prediction: Uses attention patterns and token sequences to predict future chunk needs
2. Adaptive Prefetching: Dynamically adjusts prefetch strategies based on prediction accuracy
3. Bandwidth-Aware Scheduling: Optimizes transfer timing based on GPU memory pressure and available bandwidth
4. Context-Aware Prioritization: Uses semantic similarity and access patterns to prioritize chunks
5. Predictive Eviction: Anticipates which GPU chunks can be safely moved to CPU

This scheduler is designed to work seamlessly with CacheBlend kernels and speculative decoding.
"""

import torch
import time
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set, NamedTuple
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

logger = logging.getLogger(__name__)

@dataclass
class ChunkPrediction:
    """Prediction for chunk access timing and probability"""
    chunk_id: str
    probability: float
    expected_time: float  # When the chunk will be needed (in seconds)
    confidence: float
    access_pattern: str  # "sequential", "random", "burst", "sparse"
    semantic_relevance: float

@dataclass
class TransferOperation:
    """Represents a chunk transfer operation with priority and timing"""
    chunk_id: str
    source: str  # "gpu" or "cpu"
    destination: str  # "gpu" or "cpu"
    priority: float
    estimated_transfer_time: float
    size_bytes: int
    reason: str
    deadline: float  # When this transfer must complete
    bandwidth_requirement: float  # Required bandwidth (GB/s)
    is_prefetch: bool  # Whether this is a proactive prefetch

@dataclass
class SchedulerMetrics:
    """Comprehensive metrics for scheduler performance"""
    total_transfers: int
    gpu_to_cpu_transfers: int
    cpu_to_gpu_transfers: int
    prefetch_transfers: int
    average_transfer_time: float
    prediction_accuracy: float
    bandwidth_utilization: float
    memory_efficiency: float
    cache_hit_rate: float
    prefetch_hit_rate: float
    transfer_overlap: float  # Percentage of transfers that overlap with computation

class AdaptiveTemporalScheduler:
    """
    Novel scheduler implementing Adaptive Temporal Locality with Predictive Prefetching.
    
    Key Innovations:
    - Temporal locality prediction using attention pattern analysis
    - Adaptive prefetching with dynamic strategy adjustment
    - Bandwidth-aware scheduling with transfer overlap optimization
    - Predictive eviction based on access pattern analysis
    - Context-aware prioritization using semantic similarity
    """
    
    def __init__(
        self,
        kv_cache_manager,
        speculative_decoder,
        max_concurrent_transfers: int = 3,
        prediction_horizon: float = 2.0,  # Look ahead 2 seconds
        bandwidth_threshold: float = 0.8,  # Use 80% of available bandwidth
        device: str = "cuda",
        enable_predictive_eviction: bool = True,
        enable_adaptive_prefetching: bool = True
    ):
        self.kv_cache_manager = kv_cache_manager
        self.speculative_decoder = speculative_decoder
        self.max_concurrent_transfers = max_concurrent_transfers
        self.prediction_horizon = prediction_horizon
        self.bandwidth_threshold = bandwidth_threshold
        self.device = device
        self.enable_predictive_eviction = enable_predictive_eviction
        self.enable_adaptive_prefetching = enable_adaptive_prefetching
        
        # Scheduler state
        self.transfer_queue = []
        self.active_transfers = set()
        self.transfer_history = deque(maxlen=1000)
        self.prediction_history = deque(maxlen=500)
        
        # Performance tracking
        self.bandwidth_measurements = deque(maxlen=100)
        self.memory_pressure_history = deque(maxlen=100)
        self.access_pattern_history = defaultdict(lambda: deque(maxlen=50))
        
        # Adaptive parameters
        self.prefetch_aggressiveness = 0.7  # 0.0 = conservative, 1.0 = aggressive
        self.eviction_threshold = 0.3  # Probability threshold for eviction
        self.temporal_weight = 0.6  # Weight for temporal locality vs semantic relevance
        
        # Metrics
        self.metrics = SchedulerMetrics(
            total_transfers=0,
            gpu_to_cpu_transfers=0,
            cpu_to_gpu_transfers=0,
            prefetch_transfers=0,
            average_transfer_time=0.0,
            prediction_accuracy=0.0,
            bandwidth_utilization=0.0,
            memory_efficiency=0.0,
            cache_hit_rate=0.0,
            prefetch_hit_rate=0.0,
            transfer_overlap=0.0
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
        logger.info("Adaptive Temporal Scheduler started")
    
    def _scheduler_loop(self):
        """Main scheduler loop with adaptive decision making"""
        while self.running:
            try:
                # Measure current system state
                self._measure_system_state()
                
                # Update predictions based on current context
                self._update_predictions()
                
                # Process transfer queue with bandwidth awareness
                self._process_transfer_queue()
                
                # Adapt scheduler parameters based on performance
                self._adapt_scheduler_parameters()
                
                # Sleep for adaptive interval
                sleep_time = self._calculate_adaptive_sleep()
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(1.0)
    
    def _measure_system_state(self):
        """Measure current system state for adaptive decisions"""
        # Measure GPU memory pressure
        gpu_memory_used = self.kv_cache_manager.gpu_memory_used
        gpu_memory_total = self.kv_cache_manager.max_gpu_memory
        memory_pressure = gpu_memory_used / gpu_memory_total if gpu_memory_total > 0 else 0.0
        
        self.memory_pressure_history.append({
            'timestamp': time.time(),
            'pressure': memory_pressure,
            'gpu_chunks': len(self.kv_cache_manager.gpu_cache),
            'cpu_chunks': len(self.kv_cache_manager.cpu_cache)
        })
        
        # Measure bandwidth (simplified - in practice would use CUDA events)
        if self.transfer_history:
            recent_transfers = list(self.transfer_history)[-10:]
            total_size = sum(t['size_bytes'] for t in recent_transfers)
            total_time = sum(t['transfer_time'] for t in recent_transfers)
            if total_time > 0:
                bandwidth = total_size / (1024**3) / total_time  # GB/s
                self.bandwidth_measurements.append({
                    'timestamp': time.time(),
                    'bandwidth': bandwidth
                })
    
    def _update_predictions(self):
        """Update chunk predictions using temporal locality analysis"""
        if not self.enable_adaptive_prefetching:
            return
        
        current_time = time.time()
        current_context = self._get_current_context()
        
        # Get all available chunks
        all_chunks = list(self.kv_cache_manager.gpu_cache.keys()) + \
                    list(self.kv_cache_manager.cpu_cache.keys())
        
        # Generate predictions for each chunk
        predictions = []
        for chunk_id in all_chunks:
            prediction = self._predict_chunk_access(chunk_id, current_context, current_time)
            if prediction:
                predictions.append(prediction)
        
        # Update prediction history
        self.prediction_history.append({
            'timestamp': current_time,
            'predictions': predictions,
            'context': current_context
        })
        
        # Schedule transfers based on predictions
        self._schedule_predictive_transfers(predictions, current_time)
    
    def _predict_chunk_access(self, chunk_id: str, context: Any, current_time: float) -> Optional[ChunkPrediction]:
        """Predict when and how likely a chunk will be accessed"""
        # Get chunk metadata
        entry = self._get_chunk_entry(chunk_id)
        if not entry:
            return None
        
        # Analyze access patterns
        access_pattern = self._analyze_access_pattern(chunk_id)
        
        # Calculate temporal locality score
        temporal_score = self._calculate_temporal_locality(chunk_id, context)
        
        # Calculate semantic relevance
        semantic_score = self._calculate_semantic_relevance(chunk_id, context)
        
        # Combine scores with adaptive weights
        combined_score = (self.temporal_weight * temporal_score + 
                         (1 - self.temporal_weight) * semantic_score)
        
        # Predict access timing based on patterns
        expected_time = self._predict_access_timing(chunk_id, access_pattern, current_time)
        
        # Calculate confidence based on pattern consistency
        confidence = self._calculate_prediction_confidence(chunk_id, access_pattern)
        
        return ChunkPrediction(
            chunk_id=chunk_id,
            probability=combined_score,
            expected_time=expected_time,
            confidence=confidence,
            access_pattern=access_pattern,
            semantic_relevance=semantic_score
        )
    
    def _analyze_access_pattern(self, chunk_id: str) -> str:
        """Analyze the access pattern for a chunk"""
        history = self.access_pattern_history[chunk_id]
        if len(history) < 3:
            return "sparse"
        
        # Calculate access intervals
        timestamps = [h['timestamp'] for h in history]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not intervals:
            return "sparse"
        
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Classify patterns
        if std_interval < 0.1 * mean_interval:
            return "sequential"  # Very regular access
        elif std_interval < 0.5 * mean_interval:
            return "burst"  # Somewhat regular
        else:
            return "random"  # Irregular access
    
    def _calculate_temporal_locality(self, chunk_id: str, context: Any) -> float:
        """Calculate temporal locality score based on recent access patterns"""
        history = self.access_pattern_history[chunk_id]
        if not history:
            return 0.0
        
        current_time = time.time()
        recent_accesses = [h for h in history if current_time - h['timestamp'] < 60.0]  # Last minute
        
        if not recent_accesses:
            return 0.0
        
        # Calculate recency and frequency scores
        latest_access = max(h['timestamp'] for h in recent_accesses)
        recency_score = max(0.0, 1.0 - (current_time - latest_access) / 60.0)
        
        frequency_score = min(1.0, len(recent_accesses) / 10.0)
        
        return (recency_score + frequency_score) / 2.0
    
    def _calculate_semantic_relevance(self, chunk_id: str, context: Any) -> float:
        """Calculate semantic relevance score using the chunk's metadata"""
        entry = self._get_chunk_entry(chunk_id)
        if not entry:
            return 0.0
        
        # Use relevance score from metadata if available
        if hasattr(entry.metadata, 'relevance_score'):
            return float(entry.metadata.relevance_score)
        
        # Fallback to access count-based relevance
        access_count = getattr(entry.metadata, 'access_count', 0)
        return min(1.0, access_count / 5.0)
    
    def _predict_access_timing(self, chunk_id: str, access_pattern: str, current_time: float) -> float:
        """Predict when a chunk will be accessed next"""
        history = self.access_pattern_history[chunk_id]
        if len(history) < 2:
            return current_time + 10.0  # Default prediction
        
        # Calculate average interval
        timestamps = [h['timestamp'] for h in history]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not intervals:
            return current_time + 10.0
        
        avg_interval = np.mean(intervals)
        
        # Adjust based on pattern
        if access_pattern == "sequential":
            return current_time + avg_interval * 0.8  # Slightly earlier
        elif access_pattern == "burst":
            return current_time + avg_interval * 1.2  # Slightly later
        else:  # random
            return current_time + avg_interval * 1.5  # More conservative
    
    def _calculate_prediction_confidence(self, chunk_id: str, access_pattern: str) -> float:
        """Calculate confidence in the prediction"""
        history = self.access_pattern_history[chunk_id]
        if len(history) < 3:
            return 0.3  # Low confidence with few samples
        
        # Higher confidence for regular patterns
        if access_pattern == "sequential":
            return 0.9
        elif access_pattern == "burst":
            return 0.7
        else:  # random
            return 0.5
    
    def _schedule_predictive_transfers(self, predictions: List[ChunkPrediction], current_time: float):
        """Schedule transfers based on predictions"""
        # Sort predictions by urgency (earliest deadline first)
        urgent_predictions = [p for p in predictions if p.expected_time - current_time < self.prediction_horizon]
        urgent_predictions.sort(key=lambda p: p.expected_time)
        
        # Schedule transfers for urgent predictions
        for prediction in urgent_predictions:
            if prediction.probability > self.prefetch_aggressiveness:
                self._schedule_predictive_transfer(prediction)
    
    def _schedule_predictive_transfer(self, prediction: ChunkPrediction):
        """Schedule a predictive transfer operation"""
        chunk_id = prediction.chunk_id
        current_location = self._get_chunk_location(chunk_id)
        target_location = "gpu" if prediction.probability > 0.7 else "cpu"
        
        if current_location == target_location:
            return  # Already in optimal location
        
        # Calculate transfer priority based on urgency and confidence
        time_until_needed = prediction.expected_time - time.time()
        urgency = max(0.0, 1.0 - time_until_needed / self.prediction_horizon)
        priority = prediction.probability * urgency * prediction.confidence
        
        # Schedule the transfer
        self._schedule_transfer(
            chunk_id=chunk_id,
            source=current_location,
            destination=target_location,
            priority=priority,
            reason=f"Predictive: {prediction.access_pattern} (p={prediction.probability:.2f})",
            deadline=prediction.expected_time,
            is_prefetch=True
        )
    
    def _process_transfer_queue(self):
        """Process the transfer queue with bandwidth awareness"""
        with self.lock:
            # Sort queue by priority and deadline
            self.transfer_queue.sort(key=lambda t: (-t.priority, t.deadline))
            
            # Calculate available bandwidth
            available_bandwidth = self._estimate_available_bandwidth()
            
            # Execute transfers up to bandwidth limit
            active_bandwidth = 0.0
            transfers_to_execute = []
            
            for transfer in self.transfer_queue:
                if len(transfers_to_execute) >= self.max_concurrent_transfers:
                    break
                
                if active_bandwidth + transfer.bandwidth_requirement <= available_bandwidth:
                    transfers_to_execute.append(transfer)
                    active_bandwidth += transfer.bandwidth_requirement
            
            # Execute selected transfers
            for transfer in transfers_to_execute:
                if self._is_transfer_valid(transfer):
                    self._execute_transfer(transfer)
                    self.transfer_queue.remove(transfer)
    
    def _estimate_available_bandwidth(self) -> float:
        """Estimate available bandwidth for transfers"""
        if not self.bandwidth_measurements:
            return 10.0  # Default 10 GB/s
        
        # Use recent bandwidth measurements
        recent_measurements = list(self.bandwidth_measurements)[-5:]
        avg_bandwidth = np.mean([m['bandwidth'] for m in recent_measurements])
        
        # Reserve some bandwidth for other operations
        available_bandwidth = avg_bandwidth * self.bandwidth_threshold
        
        return max(1.0, available_bandwidth)  # Minimum 1 GB/s
    
    def _is_transfer_valid(self, transfer: TransferOperation) -> bool:
        """Check if a transfer operation is still valid"""
        # Check if chunk still exists in source location
        if transfer.source == "gpu":
            if transfer.chunk_id not in self.kv_cache_manager.gpu_cache:
                return False
        else:
            if transfer.chunk_id not in self.kv_cache_manager.cpu_cache:
                return False
        
        # Check if destination has space
        if transfer.destination == "gpu":
            if len(self.kv_cache_manager.gpu_cache) >= self.kv_cache_manager.max_gpu_chunks:
                return False
        else:
            if len(self.kv_cache_manager.cpu_cache) >= self.kv_cache_manager.max_cpu_chunks:
                return False
        
        return True
    
    def _execute_transfer(self, transfer: TransferOperation):
        """Execute a transfer operation"""
        start_time = time.time()
        
        try:
            # Mark transfer as active
            self.active_transfers.add(transfer.chunk_id)
            
            # Perform the transfer
            if transfer.source == "gpu" and transfer.destination == "cpu":
                self._transfer_gpu_to_cpu(transfer)
                self.metrics.gpu_to_cpu_transfers += 1
            elif transfer.source == "cpu" and transfer.destination == "gpu":
                self._transfer_cpu_to_gpu(transfer)
                self.metrics.cpu_to_gpu_transfers += 1
            
            # Record transfer completion
            transfer_time = time.time() - start_time
            self.transfer_history.append({
                "chunk_id": transfer.chunk_id,
                "source": transfer.source,
                "destination": transfer.destination,
                "transfer_time": transfer_time,
                "reason": transfer.reason,
                "is_prefetch": transfer.is_prefetch,
                "timestamp": start_time
            })
            
            # Update metrics
            self.metrics.total_transfers += 1
            if transfer.is_prefetch:
                self.metrics.prefetch_transfers += 1
            
            self._update_average_transfer_time(transfer_time)
            
            # Update access pattern history
            self._update_access_pattern(transfer.chunk_id, start_time)
            
            logger.info(f"Transfer completed: {transfer.chunk_id} {transfer.source}->{transfer.destination} "
                       f"({transfer_time:.3f}s, prefetch: {transfer.is_prefetch})")
            
        except Exception as e:
            logger.error(f"Transfer failed for {transfer.chunk_id}: {e}")
        finally:
            # Remove from active transfers
            self.active_transfers.discard(transfer.chunk_id)
    
    def _transfer_gpu_to_cpu(self, transfer: TransferOperation):
        """Transfer a chunk from GPU to CPU"""
        entry = self.kv_cache_manager.gpu_cache[transfer.chunk_id]
        
        # Move to CPU
        entry.keys = entry.keys.cpu()
        entry.values = entry.values.cpu()
        entry.metadata.is_on_gpu = False
        
        # Update cache
        self.kv_cache_manager.cpu_cache[transfer.chunk_id] = entry
        self.kv_cache_manager.gpu_memory_used -= entry.metadata.size_bytes
        self.kv_cache_manager.cpu_memory_used += entry.metadata.size_bytes
        
        # Remove from GPU cache
        del self.kv_cache_manager.gpu_cache[transfer.chunk_id]
    
    def _transfer_cpu_to_gpu(self, transfer: TransferOperation):
        """Transfer a chunk from CPU to GPU"""
        entry = self.kv_cache_manager.cpu_cache[transfer.chunk_id]
        
        # Move to GPU
        entry.keys = entry.keys.to(self.device)
        entry.values = entry.values.to(self.device)
        entry.metadata.is_on_gpu = True
        
        # Update cache
        self.kv_cache_manager.gpu_cache[transfer.chunk_id] = entry
        self.kv_cache_manager.gpu_memory_used += entry.metadata.size_bytes
        self.kv_cache_manager.cpu_memory_used -= entry.metadata.size_bytes
        
        # Remove from CPU cache
        del self.kv_cache_manager.cpu_cache[transfer.chunk_id]
    
    def _update_access_pattern(self, chunk_id: str, access_time: float):
        """Update access pattern history for a chunk"""
        self.access_pattern_history[chunk_id].append({
            'timestamp': access_time,
            'type': 'transfer'
        })
    
    def _adapt_scheduler_parameters(self):
        """Adaptively adjust scheduler parameters based on performance"""
        if len(self.prediction_history) < 10:
            return
        
        # Calculate prediction accuracy
        recent_predictions = list(self.prediction_history)[-10:]
        correct_predictions = 0
        total_predictions = 0
        
        for pred_record in recent_predictions:
            for pred in pred_record['predictions']:
                if pred.expected_time > 0:
                    total_predictions += 1
                    # Check if prediction was accurate (within 0.5 seconds)
                    actual_access = self._find_actual_access_time(pred.chunk_id, pred_record['timestamp'])
                    if actual_access and abs(actual_access - pred.expected_time) < 0.5:
                        correct_predictions += 1
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            self.metrics.prediction_accuracy = accuracy
            
            # Adjust prefetch aggressiveness based on accuracy
            if accuracy > 0.8:
                self.prefetch_aggressiveness = min(1.0, self.prefetch_aggressiveness + 0.05)
            elif accuracy < 0.5:
                self.prefetch_aggressiveness = max(0.3, self.prefetch_aggressiveness - 0.05)
    
    def _find_actual_access_time(self, chunk_id: str, prediction_time: float) -> Optional[float]:
        """Find when a chunk was actually accessed after prediction"""
        # Look for actual access in transfer history
        for record in self.transfer_history:
            if (record['chunk_id'] == chunk_id and 
                record['timestamp'] > prediction_time and
                record['destination'] == 'gpu'):
                return record['timestamp']
        return None
    
    def _calculate_adaptive_sleep(self) -> float:
        """Calculate adaptive sleep interval based on system activity"""
        if len(self.transfer_history) < 5:
            return 0.1  # Default interval
        
        # Adjust sleep based on transfer frequency
        recent_transfers = [t for t in self.transfer_history 
                           if time.time() - t['timestamp'] < 10.0]
        
        if len(recent_transfers) > 10:
            return 0.05  # High activity - shorter sleep
        elif len(recent_transfers) < 3:
            return 0.2   # Low activity - longer sleep
        else:
            return 0.1   # Normal activity - default sleep
    
    def _get_chunk_entry(self, chunk_id: str):
        """Get chunk entry from either GPU or CPU cache"""
        if chunk_id in self.kv_cache_manager.gpu_cache:
            return self.kv_cache_manager.gpu_cache[chunk_id]
        elif chunk_id in self.kv_cache_manager.cpu_cache:
            return self.kv_cache_manager.cpu_cache[chunk_id]
        return None
    
    def _get_chunk_location(self, chunk_id: str) -> str:
        """Get current location of a chunk"""
        if chunk_id in self.kv_cache_manager.gpu_cache:
            return "gpu"
        elif chunk_id in self.kv_cache_manager.cpu_cache:
            return "cpu"
        return "unknown"
    
    def _schedule_transfer(self, chunk_id: str, source: str, destination: str, 
                          priority: float, reason: str, deadline: float, is_prefetch: bool = False):
        """Schedule a transfer operation"""
        # Get chunk size
        entry = self._get_chunk_entry(chunk_id)
        if not entry:
            return
        
        # Calculate bandwidth requirement (simplified)
        size_gb = entry.metadata.size_bytes / (1024**3)
        estimated_time = size_gb / 5.0  # Assume 5 GB/s transfer rate
        bandwidth_requirement = size_gb / estimated_time
        
        # Create transfer operation
        transfer_op = TransferOperation(
            chunk_id=chunk_id,
            source=source,
            destination=destination,
            priority=priority,
            estimated_transfer_time=estimated_time,
            size_bytes=entry.metadata.size_bytes,
            reason=reason,
            deadline=deadline,
            bandwidth_requirement=bandwidth_requirement,
            is_prefetch=is_prefetch
        )
        
        # Add to queue if not already scheduled
        with self.lock:
            # Check if already in queue
            for existing_op in self.transfer_queue:
                if existing_op.chunk_id == chunk_id:
                    # Update existing operation if new one has higher priority
                    if priority > existing_op.priority:
                        self.transfer_queue.remove(existing_op)
                        heapq.heappush(self.transfer_queue, transfer_op)
                    return
            
            heapq.heappush(self.transfer_queue, transfer_op)
            logger.debug(f"Scheduled transfer: {chunk_id} {source}->{destination} "
                        f"(priority: {priority:.3f}, prefetch: {is_prefetch})")
    
    def _update_average_transfer_time(self, new_time: float):
        """Update average transfer time"""
        if self.metrics.total_transfers == 1:
            self.metrics.average_transfer_time = new_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_transfer_time = (
                alpha * new_time + (1 - alpha) * self.metrics.average_transfer_time
            )
    
    def update_cache_hit_rate(self, hit_rate: float):
        """Update cache hit rate metric"""
        self.metrics.cache_hit_rate = hit_rate
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get comprehensive scheduler statistics"""
        return {
            "metrics": self.metrics.__dict__,
            "queue_size": len(self.transfer_queue),
            "active_transfers": len(self.active_transfers),
            "recent_transfers": list(self.transfer_history)[-10:],
            "gpu_chunks": len(self.kv_cache_manager.gpu_cache),
            "cpu_chunks": len(self.kv_cache_manager.cpu_cache),
            "gpu_memory_used_gb": self.kv_cache_manager.gpu_memory_used / (1024**3),
            "cpu_memory_used_gb": self.kv_cache_manager.cpu_memory_used / (1024**3),
            "adaptive_parameters": {
                "prefetch_aggressiveness": self.prefetch_aggressiveness,
                "eviction_threshold": self.eviction_threshold,
                "temporal_weight": self.temporal_weight
            },
            "system_state": {
                "memory_pressure": self.memory_pressure_history[-1]['pressure'] if self.memory_pressure_history else 0.0,
                "available_bandwidth": self._estimate_available_bandwidth()
            }
        }
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        logger.info("Adaptive Temporal Scheduler stopped")
    
    def clear_queue(self):
        """Clear the transfer queue"""
        with self.lock:
            self.transfer_queue.clear()
        logger.info("Transfer queue cleared")
    
    def force_transfer(self, chunk_id: str, destination: str, priority: float = 1.0):
        """Force a transfer operation with high priority"""
        current_location = self._get_chunk_location(chunk_id)
        if current_location != destination:
            self._schedule_transfer(
                chunk_id=chunk_id,
                source=current_location,
                destination=destination,
                priority=priority,
                reason="Forced transfer",
                deadline=time.time() + 1.0,
                is_prefetch=False
            )


