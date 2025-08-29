#!/usr/bin/env python3
"""
CacheBlend Kernels Wrapper
Provides efficient KV cache operations using CacheBlend CUDA kernels
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
import logging
import os
import sys

logger = logging.getLogger(__name__)

class CacheBlendKernels:
    """Wrapper for CacheBlend CUDA kernels"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._init_kernels()
    
    def _init_kernels(self):
        """Initialize CacheBlend kernels (adapt to vllm_blend layout)."""
        self.cache_ops = None
        self.attention_ops = None
        try:
            # Ensure local vllm_blend is importable
            sys.path.append(os.path.join(os.path.dirname(__file__), "../../vllm_blend"))
            # Newer vllm_blend exposes attention via python in vllm.attention.ops.paged_attn
            from vllm.attention.ops import paged_attn as attention_ops  # type: ignore
            self.attention_ops = attention_ops
            logger.info("CacheBlend attention ops loaded (vllm.attention.ops.paged_attn)")
            # cache_ops may be fused in vllm._C; we don't require them for current flow
        except Exception as e:
            logger.warning(f"CacheBlend kernels not available: {e}")
            self.attention_ops = None
    
    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str = "auto",
        kv_scale: float = 1.0
    ) -> None:
        """Reshape and cache key-value tensors using CacheBlend kernels"""
        if self.cache_ops is None:
            # Fallback to PyTorch implementation
            self._reshape_and_cache_fallback(key, value, key_cache, value_cache, slot_mapping)
            return
        
        try:
            self.cache_ops.reshape_and_cache(
                key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, kv_scale
            )
        except Exception as e:
            logger.warning(f"CacheBlend kernel failed, falling back to PyTorch: {e}")
            self._reshape_and_cache_fallback(key, value, key_cache, value_cache, slot_mapping)
    
    def _reshape_and_cache_fallback(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor
    ):
        """Fallback implementation using PyTorch"""
        # Reshape key and value to match cache format
        key = key.view(-1, key_cache.shape[-2], key_cache.shape[-1])
        value = value.view(-1, value_cache.shape[-2], value_cache.shape[-1])
        
        # Update cache using slot mapping
        key_cache[slot_mapping] = key
        value_cache[slot_mapping] = value
    
    def paged_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_cache_scale: Optional[torch.Tensor],
        value_cache_scale: Optional[torch.Tensor],
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        max_context_len: int,
        alibi_slopes: Optional[torch.Tensor] = None,
        kv_cache_dtype: str = "auto",
        kv_scale: float = 1.0,
        head_mapping: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Perform paged attention using CacheBlend kernels"""
        if self.attention_ops is None:
            # Fallback to PyTorch implementation
            return self._paged_attention_fallback(
                query, key_cache, value_cache, block_tables, context_lens, block_size
            )
        
        try:
            return self.attention_ops.paged_attention(
                query, key_cache, value_cache, key_cache_scale, value_cache_scale,
                block_tables, context_lens, block_size, max_context_len,
                alibi_slopes, kv_cache_dtype, kv_scale, head_mapping, scale
            )
        except Exception as e:
            logger.warning(f"CacheBlend attention kernel failed, falling back to PyTorch: {e}")
            return self._paged_attention_fallback(
                query, key_cache, value_cache, block_tables, context_lens, block_size
            )
    
    def _paged_attention_fallback(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int
    ) -> torch.Tensor:
        """Fallback attention implementation using PyTorch"""
        # Simple attention implementation
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # For simplicity, assume contiguous key/value cache
        key = key_cache.view(-1, num_heads, head_dim)
        value = value_cache.view(-1, num_heads, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Apply causal mask if needed
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
            scores = scores.masked_fill(mask.bool(), float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Compute output
        output = torch.matmul(attn_weights, value)
        
        return output
    
    def copy_blocks(
        self,
        src_k_cache: torch.Tensor,
        src_v_cache: torch.Tensor,
        dst_k_cache: torch.Tensor,
        dst_v_cache: torch.Tensor,
        block_mapping: torch.Tensor,
    ) -> None:
        """Copy blocks between key-value caches using CacheBlend kernels"""
        if self.cache_ops is None:
            # Fallback to PyTorch implementation
            self._copy_blocks_fallback(src_k_cache, src_v_cache, dst_k_cache, dst_v_cache, block_mapping)
            return
        
        try:
            self.cache_ops.copy_blocks(src_k_cache, src_v_cache, dst_k_cache, dst_v_cache, block_mapping)
        except Exception as e:
            logger.warning(f"CacheBlend copy kernel failed, falling back to PyTorch: {e}")
            self._copy_blocks_fallback(src_k_cache, src_v_cache, dst_k_cache, dst_v_cache, block_mapping)
    
    def _copy_blocks_fallback(
        self,
        src_k_cache: torch.Tensor,
        src_v_cache: torch.Tensor,
        dst_k_cache: torch.Tensor,
        dst_v_cache: torch.Tensor,
        block_mapping: torch.Tensor,
    ):
        """Fallback copy implementation using PyTorch"""
        for i, mapping in enumerate(block_mapping):
            src_idx = mapping[0].item()
            dst_idx = mapping[1].item()
            dst_k_cache[dst_idx] = src_k_cache[src_idx].clone()
            dst_v_cache[dst_idx] = src_v_cache[src_idx].clone()
    
    def swap_blocks(
        self,
        src_k_cache: torch.Tensor,
        src_v_cache: torch.Tensor,
        dst_k_cache: torch.Tensor,
        dst_v_cache: torch.Tensor,
        block_mapping: torch.Tensor,
    ) -> None:
        """Swap blocks between key-value caches using CacheBlend kernels"""
        if self.cache_ops is None:
            # Fallback to PyTorch implementation
            self._swap_blocks_fallback(src_k_cache, src_v_cache, dst_k_cache, dst_v_cache, block_mapping)
            return
        
        try:
            self.cache_ops.swap_blocks(src_k_cache, src_v_cache, dst_k_cache, dst_v_cache, block_mapping)
        except Exception as e:
            logger.warning(f"CacheBlend swap kernel failed, falling back to PyTorch: {e}")
            self._swap_blocks_fallback(src_k_cache, src_v_cache, dst_k_cache, dst_v_cache, block_mapping)
    
    def _swap_blocks_fallback(
        self,
        src_k_cache: torch.Tensor,
        src_v_cache: torch.Tensor,
        dst_k_cache: torch.Tensor,
        dst_v_cache: torch.Tensor,
        block_mapping: torch.Tensor,
    ):
        """Fallback swap implementation using PyTorch"""
        for i, mapping in enumerate(block_mapping):
            src_idx = mapping[0].item()
            dst_idx = mapping[1].item()
            
            # Swap key cache
            temp_k = src_k_cache[src_idx].clone()
            src_k_cache[src_idx] = dst_k_cache[dst_idx].clone()
            dst_k_cache[dst_idx] = temp_k
            
            # Swap value cache
            temp_v = src_v_cache[src_idx].clone()
            src_v_cache[src_idx] = dst_v_cache[dst_idx].clone()
            dst_v_cache[dst_idx] = temp_v
    
    def convert_fp8(
        self,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
    ) -> None:
        """Convert cache between different precision formats using CacheBlend kernels"""
        if self.cache_ops is None:
            # Fallback to PyTorch implementation
            self._convert_fp8_fallback(src_cache, dst_cache)
            return
        
        try:
            self.cache_ops.convert_fp8(src_cache, dst_cache)
        except Exception as e:
            logger.warning(f"CacheBlend convert kernel failed, falling back to PyTorch: {e}")
            self._convert_fp8_fallback(src_cache, dst_cache)
    
    def _convert_fp8_fallback(self, src_cache: torch.Tensor, dst_cache: torch.Tensor):
        """Fallback conversion implementation using PyTorch"""
        # Simple conversion - just copy with potential dtype change
        dst_cache.copy_(src_cache)
    
    def blend_kv_cache(
        self,
        retrieved_k: torch.Tensor,
        retrieved_v: torch.Tensor,
        fresh_k: torch.Tensor,
        fresh_v: torch.Tensor,
        valid_mask: torch.Tensor,
        blend_ratio: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Blend retrieved and fresh KV cache using CacheBlend kernels"""
        if self.cache_ops is None:
            # Fallback to PyTorch implementation
            return self._blend_kv_cache_fallback(
                retrieved_k, retrieved_v, fresh_k, fresh_v, valid_mask, blend_ratio
            )
        
        try:
            # This would use CacheBlend's blending kernels
            # For now, use fallback implementation
            return self._blend_kv_cache_fallback(
                retrieved_k, retrieved_v, fresh_k, fresh_v, valid_mask, blend_ratio
            )
        except Exception as e:
            logger.warning(f"CacheBlend blend kernel failed, falling back to PyTorch: {e}")
            return self._blend_kv_cache_fallback(
                retrieved_k, retrieved_v, fresh_k, fresh_v, valid_mask, blend_ratio
            )
    
    def _blend_kv_cache_fallback(
        self,
        retrieved_k: torch.Tensor,
        retrieved_v: torch.Tensor,
        fresh_k: torch.Tensor,
        fresh_v: torch.Tensor,
        valid_mask: torch.Tensor,
        blend_ratio: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback blending implementation using PyTorch"""
        # Simple blending: weighted combination of retrieved and fresh KV
        blended_k = blend_ratio * retrieved_k + (1 - blend_ratio) * fresh_k
        blended_v = blend_ratio * retrieved_v + (1 - blend_ratio) * fresh_v
        
        # Apply valid mask
        blended_k = blended_k * valid_mask.unsqueeze(-1).unsqueeze(-1)
        blended_v = blended_v * valid_mask.unsqueeze(-1).unsqueeze(-1)
        
        return blended_k, blended_v


