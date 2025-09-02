#!/usr/bin/env python3
"""
KV Cache Manager for CacheBlend-based chunk storage
Handles GPU and CPU storage of chunks with dynamic swapping
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import OrderedDict
import time
import os
import sys

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for a cached chunk"""
    chunk_id: str
    text: str
    tokens: List[int]
    relevance_score: float
    access_count: int
    last_access_time: float
    size_bytes: int
    layer_count: int
    is_on_gpu: bool

@dataclass
class KVCacheEntry:
    """Key-Value cache entry for a chunk"""
    keys: torch.Tensor  # Shape: [num_layers, seq_len, num_heads, head_dim]
    values: torch.Tensor  # Shape: [num_layers, seq_len, num_heads, head_dim]
    metadata: ChunkMetadata
    valid_mask: torch.Tensor  # CPU tensor indicating valid positions

class KVCacheManager:
    """Manages Key-Value cache storage on GPU and CPU using CacheBlend kernels"""
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        gpu_memory_limit_gb: float = 40.0,
        cpu_memory_limit_gb: float = 100.0,
        max_gpu_chunks: int = 5,
        max_cpu_chunks: int = 50,
        device: str = "cuda",
        require_kernels: bool = False
    ):
        self.model_config = model_config
        self.gpu_memory_limit = gpu_memory_limit_gb * 1024**3  # Convert to bytes
        self.cpu_memory_limit = cpu_memory_limit_gb * 1024**3
        self.max_gpu_chunks = max_gpu_chunks
        self.max_cpu_chunks = max_cpu_chunks
        self.device = device
        
        # Storage containers
        self.gpu_cache: OrderedDict[str, KVCacheEntry] = OrderedDict()
        self.cpu_cache: OrderedDict[str, KVCacheEntry] = OrderedDict()
        
        # Memory tracking
        self.gpu_memory_used = 0
        self.cpu_memory_used = 0
        
        # Statistics
        self.stats = {
            "gpu_hits": 0,
            "cpu_hits": 0,
            "misses": 0,
            "evictions": 0,
            "swaps": 0
        }
        
        # Kernel enforcement
        self.require_kernels = require_kernels
        
        # Initialize CacheBlend kernels
        self._init_cacheblend_kernels()
    
    def _init_cacheblend_kernels(self):
        """Initialize CacheBlend kernels for efficient KV operations"""
        try:
            # Import CacheBlend kernels (prefer absolute import to work when run as a script)
            try:
                # Ensure module directory is on sys.path for absolute import
                module_dir = os.path.dirname(__file__)
                if module_dir and module_dir not in sys.path:
                    sys.path.insert(0, module_dir)
                from cacheblend_kernels import CacheBlendKernels  
            except ImportError:
                # Fallback to relative if package context exists
                from .cacheblend_kernels import CacheBlendKernels 
            self.kernels = CacheBlendKernels(self.device)
            logger.info("CacheBlend kernels initialized successfully")
        except ImportError as e:
            logger.warning(f"CacheBlend kernels not available: {e}")
            self.kernels = None
        
        if self.require_kernels:
            if getattr(self, "kernels", None) is None:
                raise RuntimeError("CacheBlend kernels are required but not available. Please build and make them importable.")
            # Accept layouts where either attention ops or cache ops are available
            has_cache = getattr(self.kernels, "cache_ops", None) is not None
            has_attn = getattr(self.kernels, "attention_ops", None) is not None
            if not (has_cache or has_attn):
                raise RuntimeError("CacheBlend kernels are required but not available (no attention or cache ops found). Make sure vllm_blend is on PYTHONPATH.")
    
    def calculate_chunk_size(self, tokens: List[int], num_layers: int, 
                           num_heads: int, head_dim: int, dtype: torch.dtype) -> int:
        """Calculate memory size for a chunk's KV cache"""
        seq_len = len(tokens)
        # Each layer has K and V tensors
        kv_size = 2 * num_layers * seq_len * num_heads * head_dim
        # Convert to bytes
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        return kv_size * bytes_per_element
    
    def create_kv_cache_entry(
        self,
        chunk_id: str,
        text: str,
        tokens: List[int],
        relevance_score: float,
        model_outputs: Dict[str, torch.Tensor]
    ) -> KVCacheEntry:
        """Create a KV cache entry from model outputs"""
        
        # Extract and normalize KV cache from model outputs
        keys = []
        values = []
        valid_mask = []
        
        past = model_outputs.get("past_key_values", [])
        num_layers = len(past)
        
        for layer_idx in range(num_layers):
            layer_kv = past[layer_idx]
            if isinstance(layer_kv, (list, tuple)):
                k, v = layer_kv
            else:
                k, v = layer_kv["key"], layer_kv["value"]
            if k.dim() == 4:
                k = k[0].permute(1, 0, 2).contiguous()
            if v.dim() == 4:
                v = v[0].permute(1, 0, 2).contiguous()
            keys.append(k)
            values.append(v)
            mask = torch.ones(k.shape[0], dtype=torch.bool, device="cpu")
            valid_mask.append(mask)
        
        # Stack across layers -> keys_tensor: [num_layers, seq_len, num_heads, head_dim]
        keys_tensor = torch.stack(keys, dim=0)
        values_tensor = torch.stack(values, dim=0)
        valid_mask_tensor = torch.stack(valid_mask, dim=0)
        
        # Calculate size using correct dims
        size_bytes = self.calculate_chunk_size(
            tokens, num_layers, keys_tensor.shape[2], keys_tensor.shape[3], keys_tensor.dtype
        )
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            text=text,
            tokens=tokens,
            relevance_score=relevance_score,
            access_count=0,
            last_access_time=time.time(),
            size_bytes=size_bytes,
            layer_count=num_layers,
            is_on_gpu=False
        )
        
        return KVCacheEntry(
            keys=keys_tensor,
            values=values_tensor,
            metadata=metadata,
            valid_mask=valid_mask_tensor
        )

    def create_placeholder_entry(
        self,
        chunk_id: str,
        text: str,
        tokens: List[int],
        relevance_score: float,
        dtype: torch.dtype = torch.float16,
    ) -> KVCacheEntry:
        """Create a zero-sized placeholder KV entry for CPU-only storage without running prefill.

        Shapes are empty so memory footprint is effectively zero; metadata still tracks size 0.
        """
        empty_k = torch.empty((0, 0, 0, 0), dtype=dtype)
        empty_v = torch.empty((0, 0, 0, 0), dtype=dtype)
        empty_mask = torch.empty((0, 0), dtype=torch.bool)

        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            text=text,
            tokens=tokens,
            relevance_score=relevance_score,
            access_count=0,
            last_access_time=time.time(),
            size_bytes=0,
            layer_count=0,
            is_on_gpu=False,
        )

        return KVCacheEntry(
            keys=empty_k,
            values=empty_v,
            metadata=metadata,
            valid_mask=empty_mask,
        )
    
    def store_chunk(self, chunk_id: str, kv_entry: KVCacheEntry, priority: str = "gpu") -> bool:
        """Store a chunk in GPU or CPU cache with eviction as needed.

        - If priority==gpu: evict/demote LRU from GPU until capacity and memory allow, then store on GPU.
        - Otherwise: evict from CPU until capacity allows, then store on CPU.
        """
        
        if priority == "gpu":
            # Evict/demote until we have capacity and memory headroom
            while (len(self.gpu_cache) >= self.max_gpu_chunks) or \
                  (self.gpu_memory_used + kv_entry.metadata.size_bytes > self.gpu_memory_limit):
                self._evict_gpu_chunk()
                if len(self.gpu_cache) == 0 and \
                   (kv_entry.metadata.size_bytes > self.gpu_memory_limit):
                    logger.warning("KV entry larger than GPU memory limit; cannot store on GPU")
                    break
            # Attempt store on GPU
            if (len(self.gpu_cache) < self.max_gpu_chunks) and \
               (self.gpu_memory_used + kv_entry.metadata.size_bytes <= self.gpu_memory_limit):
                kv_entry.keys = kv_entry.keys.to(self.device)
                kv_entry.values = kv_entry.values.to(self.device)
                self.gpu_cache[chunk_id] = kv_entry
                kv_entry.metadata.is_on_gpu = True
                self.gpu_memory_used += kv_entry.metadata.size_bytes
                logger.info(f"Stored chunk {chunk_id} in GPU cache")
                return True
            # Fall through to CPU if GPU store not possible
            priority = "cpu"
        
        # Store in CPU cache with eviction if needed
        while len(self.cpu_cache) >= self.max_cpu_chunks:
            self._evict_cpu_chunk()
        if self.cpu_memory_used + kv_entry.metadata.size_bytes <= self.cpu_memory_limit:
            kv_entry.keys = kv_entry.keys.cpu()
            kv_entry.values = kv_entry.values.cpu()
            self.cpu_cache[chunk_id] = kv_entry
            kv_entry.metadata.is_on_gpu = False
            self.cpu_memory_used += kv_entry.metadata.size_bytes
            logger.info(f"Stored chunk {chunk_id} in CPU cache")
            return True
        logger.warning(f"Failed to store chunk {chunk_id} - insufficient memory")
        return False

    def materialize_placeholder_to_gpu(self, chunk_id: str, kv_entry: KVCacheEntry) -> bool:
        """Replace a CPU placeholder (if present) with a real KV and promote to GPU with eviction as needed."""
        # If placeholder exists on CPU, remove and free its accounted memory
        if chunk_id in self.cpu_cache:
            prev = self.cpu_cache[chunk_id]
            try:
                del self.cpu_cache[chunk_id]
                self.cpu_memory_used -= prev.metadata.size_bytes
            except Exception:
                pass
        # Store to GPU (will evict/demote if needed)
        return self.store_chunk(chunk_id, kv_entry, priority="gpu")
    
    def retrieve_chunk(self, chunk_id: str) -> Optional[KVCacheEntry]:
        """Retrieve a chunk from cache, promoting to GPU if needed"""
        
        # Check GPU cache first
        if chunk_id in self.gpu_cache:
            entry = self.gpu_cache[chunk_id]
            entry.metadata.access_count += 1
            entry.metadata.last_access_time = time.time()
            self.stats["gpu_hits"] += 1
            return entry
        
        # Check CPU cache
        if chunk_id in self.cpu_cache:
            entry = self.cpu_cache[chunk_id]
            entry.metadata.access_count += 1
            entry.metadata.last_access_time = time.time()
            self.stats["cpu_hits"] += 1
            
            # Promote to GPU if space available
            self._promote_to_gpu(chunk_id, entry)
            return entry
        
        self.stats["misses"] += 1
        return None
    
    def _promote_to_gpu(self, chunk_id: str, entry: KVCacheEntry):
        """Promote a chunk from CPU to GPU cache"""
        if len(self.gpu_cache) >= self.max_gpu_chunks:
            self._evict_gpu_chunk()
        
        if self.gpu_memory_used + entry.metadata.size_bytes <= self.gpu_memory_limit:
            # Move to GPU
            entry.keys = entry.keys.to(self.device)
            entry.values = entry.values.to(self.device)
            entry.metadata.is_on_gpu = True
            
            self.gpu_cache[chunk_id] = entry
            self.gpu_memory_used += entry.metadata.size_bytes
            
            # Remove from CPU
            del self.cpu_cache[chunk_id]
            self.cpu_memory_used -= entry.metadata.size_bytes
            
            self.stats["swaps"] += 1
            logger.info(f"Promoted chunk {chunk_id} from CPU to GPU")
    
    def _evict_gpu_chunk(self):
        """Evict least recently used chunk from GPU cache"""
        if not self.gpu_cache:
            return
        
        # Find LRU chunk
        lru_chunk_id = min(
            self.gpu_cache.keys(),
            key=lambda k: self.gpu_cache[k].metadata.last_access_time
        )
        
        entry = self.gpu_cache[lru_chunk_id]
        
        # Move to CPU if space available
        if len(self.cpu_cache) < self.max_cpu_chunks and \
           self.cpu_memory_used + entry.metadata.size_bytes <= self.cpu_memory_limit:
            
            # Move to CPU
            entry.keys = entry.keys.cpu()
            entry.values = entry.values.cpu()
            entry.metadata.is_on_gpu = False
            
            self.cpu_cache[lru_chunk_id] = entry
            self.cpu_memory_used += entry.metadata.size_bytes
            
            logger.info(f"Demoted chunk {lru_chunk_id} from GPU to CPU")
        else:
            # Completely remove
            logger.info(f"Evicted chunk {lru_chunk_id} from GPU cache")
        
        # Remove from GPU
        del self.gpu_cache[lru_chunk_id]
        self.gpu_memory_used -= entry.metadata.size_bytes
        self.stats["evictions"] += 1
    
    def _evict_cpu_chunk(self):
        """Evict least recently used chunk from CPU cache"""
        if not self.cpu_cache:
            return
        
        lru_chunk_id = min(
            self.cpu_cache.keys(),
            key=lambda k: self.cpu_cache[k].metadata.last_access_time
        )
        
        entry = self.cpu_cache[lru_chunk_id]
        del self.cpu_cache[lru_chunk_id]
        self.cpu_memory_used -= entry.metadata.size_bytes
        self.stats["evictions"] += 1
        
        logger.info(f"Evicted chunk {lru_chunk_id} from CPU cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            **self.stats,
            "gpu_chunks": len(self.gpu_cache),
            "cpu_chunks": len(self.cpu_cache),
            "gpu_memory_used_gb": self.gpu_memory_used / 1024**3,
            "cpu_memory_used_gb": self.cpu_memory_used / 1024**3,
            "gpu_memory_limit_gb": self.gpu_memory_limit / 1024**3,
            "cpu_memory_limit_gb": self.cpu_memory_limit / 1024**3,
            "hit_rate": (self.stats["gpu_hits"] + self.stats["cpu_hits"]) / 
                       max(1, sum(self.stats.values()))
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.gpu_cache.clear()
        self.cpu_cache.clear()
        self.gpu_memory_used = 0
        self.cpu_memory_used = 0
        logger.info("Cache cleared")
    
    def get_top_k_chunks(self, k: int) -> List[KVCacheEntry]:
        """Get top-k chunks based on relevance score and access frequency"""
        all_entries = list(self.gpu_cache.values()) + list(self.cpu_cache.values())
        
        # Sort by relevance score and access count
        sorted_entries = sorted(
            all_entries,
            key=lambda x: (x.metadata.relevance_score, x.metadata.access_count),
            reverse=True
        )
        
        return sorted_entries[:k]


