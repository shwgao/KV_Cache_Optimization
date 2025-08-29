#!/usr/bin/env python3
"""
Performance Metrics for CacheBlend Pipeline
Tracks and records various performance indicators during execution
"""

import time
import json
import os
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import torch
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not available. Some system metrics may not be available.")

logger = logging.getLogger(__name__)

@dataclass
class GenerationMetrics:
    """Metrics for text generation performance"""
    total_tokens: int
    generation_time: float
    tokens_per_second: float
    first_token_latency: float
    average_token_latency: float
    cache_hit_rate: float
    memory_usage_gb: float
    gpu_utilization: float

@dataclass
class CacheMetrics:
    """Metrics for cache performance"""
    gpu_hits: int
    cpu_hits: int
    misses: int
    total_requests: int
    hit_rate: float
    gpu_memory_used_gb: float
    cpu_memory_used_gb: float
    swap_operations: int
    average_swap_time: float

@dataclass
class PredictionMetrics:
    """Metrics for speculative prediction performance"""
    total_predictions: int
    correct_predictions: int
    accuracy: float
    average_confidence: float
    prediction_horizon: float
    false_positives: int
    false_negatives: int

@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    cpu_usage_percent: float
    gpu_usage_percent: float
    memory_usage_percent: float
    gpu_memory_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_mb: float

class PerformanceTracker:
    """Tracks and records performance metrics during pipeline execution"""
    
    def __init__(
        self,
        output_dir: str = "results",
        log_interval: float = 1.0,
        max_history: int = 1000
    ):
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.max_history = max_history
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Metrics storage
        self.generation_metrics = deque(maxlen=max_history)
        self.cache_metrics = deque(maxlen=max_history)
        self.prediction_metrics = deque(maxlen=max_history)
        self.system_metrics = deque(maxlen=max_history)
        
        # Current session metrics
        self.session_start_time = time.time()
        self.total_generations = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
        
        # Performance counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(float)
        
        # Initialize system monitoring
        self._init_system_monitoring()
        
        logger.info(f"Performance tracker initialized, output dir: {output_dir}")
    
    def _init_system_monitoring(self):
        """Initialize system monitoring"""
        try:
            # Get initial system state
            self.initial_cpu_times = psutil.cpu_times()
            self.initial_disk_io = psutil.disk_io_counters()
            self.initial_network_io = psutil.net_io_counters()
            
            # Get GPU info using PyTorch
            if torch.cuda.is_available():
                self.gpu_device = torch.cuda.current_device()
                self.gpu_props = torch.cuda.get_device_properties(self.gpu_device)
                self.gpu = True  # Flag to indicate GPU is available
            else:
                self.gpu = None
                self.gpu_device = None
                self.gpu_props = None
                
        except Exception as e:
            logger.warning(f"Failed to initialize system monitoring: {e}")
            self.gpu = None
            self.gpu_device = None
            self.gpu_props = None
    
    def start_generation(self, generation_id: str):
        """Start timing a generation"""
        self.timers[f"generation_{generation_id}"] = time.time()
        self.counters["active_generations"] += 1
    
    def end_generation(
        self,
        generation_id: str,
        tokens_generated: int,
        cache_hits: int,
        cache_misses: int,
        first_token_time: Optional[float] = None
    ):
        """End timing a generation and record metrics"""
        if f"generation_{generation_id}" not in self.timers:
            logger.warning(f"Generation {generation_id} was not started")
            return
        
        generation_time = time.time() - self.timers[f"generation_{generation_id}"]
        del self.timers[f"generation_{generation_id}"]
        
        # Calculate metrics
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        
        # Create generation metrics
        metrics = GenerationMetrics(
            total_tokens=tokens_generated,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            first_token_latency=first_token_time or 0.0,
            average_token_latency=generation_time / tokens_generated if tokens_generated > 0 else 0,
            cache_hit_rate=cache_hit_rate,
            memory_usage_gb=system_metrics.memory_usage_percent * psutil.virtual_memory().total / (1024**3) / 100,
            gpu_utilization=system_metrics.gpu_usage_percent
        )
        
        self.generation_metrics.append(metrics)
        
        # Update session totals
        self.total_generations += 1
        self.total_tokens_generated += tokens_generated
        self.total_generation_time += generation_time
        self.counters["active_generations"] -= 1
        
        logger.info(f"Generation {generation_id} completed: {tokens_per_second:.2f} tokens/s, "
                   f"hit rate: {cache_hit_rate:.2%}")
    
    def record_cache_metrics(
        self,
        gpu_hits: int,
        cpu_hits: int,
        misses: int,
        gpu_memory_used_gb: float,
        cpu_memory_used_gb: float,
        swap_operations: int,
        average_swap_time: float
    ):
        """Record cache performance metrics"""
        total_requests = gpu_hits + cpu_hits + misses
        hit_rate = (gpu_hits + cpu_hits) / total_requests if total_requests > 0 else 0
        
        metrics = CacheMetrics(
            gpu_hits=gpu_hits,
            cpu_hits=cpu_hits,
            misses=misses,
            total_requests=total_requests,
            hit_rate=hit_rate,
            gpu_memory_used_gb=gpu_memory_used_gb,
            cpu_memory_used_gb=cpu_memory_used_gb,
            swap_operations=swap_operations,
            average_swap_time=average_swap_time
        )
        
        self.cache_metrics.append(metrics)
    
    def record_prediction_metrics(
        self,
        total_predictions: int,
        correct_predictions: int,
        average_confidence: float,
        prediction_horizon: float,
        false_positives: int = 0,
        false_negatives: int = 0
    ):
        """Record speculative prediction metrics"""
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        metrics = PredictionMetrics(
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            average_confidence=average_confidence,
            prediction_horizon=prediction_horizon,
            false_positives=false_positives,
            false_negatives=false_negatives
        )
        
        self.prediction_metrics.append(metrics)
    
    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # GPU usage
            gpu_usage = 0.0
            gpu_memory_usage = 0.0
            if self.gpu and self.gpu_device is not None:
                try:
                    # Get GPU memory usage using PyTorch
                    allocated_memory = torch.cuda.memory_allocated(self.gpu_device)
                    cached_memory = torch.cuda.memory_reserved(self.gpu_device)
                    total_memory = self.gpu_props.total_memory
                    
                    gpu_memory_usage = (cached_memory / total_memory) * 100
                    # Note: PyTorch doesn't provide direct GPU utilization percentage
                    # We'll use memory usage as a proxy
                    gpu_usage = gpu_memory_usage
                except Exception as e:
                    logger.warning(f"Failed to get GPU metrics: {e}")
                    gpu_usage = 0.0
                    gpu_memory_usage = 0.0
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = (disk_io.read_bytes - self.initial_disk_io.read_bytes) / (1024**2)
            disk_write_mb = (disk_io.write_bytes - self.initial_disk_io.write_bytes) / (1024**2)
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_mb = (network_io.bytes_sent + network_io.bytes_recv - 
                         self.initial_network_io.bytes_sent - self.initial_network_io.bytes_recv) / (1024**2)
            
            return SystemMetrics(
                cpu_usage_percent=cpu_usage,
                gpu_usage_percent=gpu_usage,
                memory_usage_percent=memory_usage,
                gpu_memory_usage_percent=gpu_memory_usage,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_io_mb=network_mb
            )
            
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return SystemMetrics(
                cpu_usage_percent=0.0,
                gpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                gpu_memory_usage_percent=0.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_io_mb=0.0
            )
    
    def record_system_metrics(self):
        """Record current system metrics"""
        metrics = self._get_system_metrics()
        self.system_metrics.append(metrics)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the current session"""
        if not self.generation_metrics:
            return {"error": "No generation metrics available"}
        
        # Calculate averages
        avg_tokens_per_second = sum(m.tokens_per_second for m in self.generation_metrics) / len(self.generation_metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in self.generation_metrics) / len(self.generation_metrics)
        avg_generation_time = sum(m.generation_time for m in self.generation_metrics) / len(self.generation_metrics)
        
        # Calculate percentiles
        generation_times = [m.generation_time for m in self.generation_metrics]
        generation_times.sort()
        p50_time = generation_times[len(generation_times) // 2]
        p95_time = generation_times[int(len(generation_times) * 0.95)]
        p99_time = generation_times[int(len(generation_times) * 0.99)]
        
        session_duration = time.time() - self.session_start_time
        
        return {
            "session_duration_hours": session_duration / 3600,
            "total_generations": self.total_generations,
            "total_tokens_generated": self.total_tokens_generated,
            "total_generation_time_hours": self.total_generation_time / 3600,
            "average_tokens_per_second": avg_tokens_per_second,
            "average_cache_hit_rate": avg_cache_hit_rate,
            "average_generation_time": avg_generation_time,
            "p50_generation_time": p50_time,
            "p95_generation_time": p95_time,
            "p99_generation_time": p99_time,
            "throughput_tokens_per_hour": self.total_tokens_generated / (session_duration / 3600) if session_duration > 0 else 0
        }
    
    def save_metrics(self, filename: Optional[str] = None):
        """Save all metrics to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert metrics to dictionaries
        data = {
            "session_info": {
                "start_time": self.session_start_time,
                "end_time": time.time(),
                "duration_hours": (time.time() - self.session_start_time) / 3600
            },
            "summary_stats": self.get_summary_stats(),
            "generation_metrics": [asdict(m) for m in self.generation_metrics],
            "cache_metrics": [asdict(m) for m in self.cache_metrics],
            "prediction_metrics": [asdict(m) for m in self.prediction_metrics],
            "system_metrics": [asdict(m) for m in self.system_metrics],
            "counters": dict(self.counters)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {filepath}")
        return filepath
    
    def print_summary(self):
        """Print a summary of current performance metrics"""
        summary = self.get_summary_stats()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Session Duration: {summary['session_duration_hours']:.2f} hours")
        print(f"Total Generations: {summary['total_generations']}")
        print(f"Total Tokens Generated: {summary['total_tokens_generated']:,}")
        print(f"Average Throughput: {summary['average_tokens_per_second']:.2f} tokens/s")
        print(f"Average Cache Hit Rate: {summary['average_cache_hit_rate']:.2%}")
        print(f"Average Generation Time: {summary['average_generation_time']:.3f}s")
        print(f"P95 Generation Time: {summary['p95_generation_time']:.3f}s")
        print(f"Overall Throughput: {summary['throughput_tokens_per_hour']:,.0f} tokens/hour")
        print("="*60)
    
    def clear_metrics(self):
        """Clear all stored metrics"""
        self.generation_metrics.clear()
        self.cache_metrics.clear()
        self.prediction_metrics.clear()
        self.system_metrics.clear()
        self.counters.clear()
        self.timers.clear()
        logger.info("All metrics cleared")
    
    def get_recent_metrics(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get metrics from the recent time window"""
        current_time = time.time()
        window_seconds = window_minutes * 60
        
        # Filter recent metrics
        recent_generation = [
            m for m in self.generation_metrics
            if current_time - self.session_start_time - m.generation_time <= window_seconds
        ]
        
        recent_cache = [
            m for m in self.cache_metrics
            if len(recent_generation) > 0  # Approximate time filtering
        ][-len(recent_generation):] if recent_generation else []
        
        if not recent_generation:
            return {"error": f"No metrics in last {window_minutes} minutes"}
        
        # Calculate recent averages
        avg_tokens_per_second = sum(m.tokens_per_second for m in recent_generation) / len(recent_generation)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_generation) / len(recent_generation)
        
        return {
            "window_minutes": window_minutes,
            "generations_in_window": len(recent_generation),
            "average_tokens_per_second": avg_tokens_per_second,
            "average_cache_hit_rate": avg_cache_hit_rate,
            "recent_generation_metrics": [asdict(m) for m in recent_generation[-10:]],  # Last 10
            "recent_cache_metrics": [asdict(m) for m in recent_cache[-10:]] if recent_cache else []
        }


