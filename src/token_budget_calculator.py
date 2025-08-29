#!/usr/bin/env python3
"""
Token Budget Calculator for GPU Memory Management
Calculates how many chunks can fit in GPU memory based on model configuration
"""

import torch
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not available. Some system metrics may not be available.")
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """GPU information and memory details"""
    name: str
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float
    memory_utilization: float

@dataclass
class TokenBudget:
    """Token budget information for GPU memory management"""
    max_tokens: int
    max_chunks: int
    tokens_per_chunk: int
    memory_required_gb: float
    memory_available_gb: float
    safety_margin_gb: float

class TokenBudgetCalculator:
    """Calculates token budget for GPU memory management"""
    
    def __init__(
        self,
        model_config: Dict,
        safety_margin_gb: float = 2.0,
        max_memory_utilization: float = 0.9
    ):
        self.model_config = model_config
        self.safety_margin_gb = safety_margin_gb
        self.max_memory_utilization = max_memory_utilization
        
        # Extract model parameters
        self.num_layers = model_config.get("num_layers", 32)
        self.num_heads = model_config.get("num_attention_heads", 32)
        self.hidden_size = model_config.get("hidden_size", 4096)
        self.head_dim = self.hidden_size // self.num_heads
        self.vocab_size = model_config.get("vocab_size", 32000)
        
        # Default chunk size (can be adjusted)
        self.default_chunk_size = 256
        
    def get_gpu_info(self) -> Optional[GPUInfo]:
        """Get information about available GPUs using PyTorch"""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available")
                return None
            
            # Get GPU device
            device = torch.cuda.current_device()
            
            # Get GPU properties
            props = torch.cuda.get_device_properties(device)
            
            # Get memory info
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            cached_memory = torch.cuda.memory_reserved(device)
            
            # Calculate memory usage
            total_memory_gb = total_memory / (1024 ** 3)
            allocated_memory_gb = allocated_memory / (1024 ** 3)
            cached_memory_gb = cached_memory / (1024 ** 3)
            free_memory_gb = total_memory_gb - cached_memory_gb
            memory_utilization = cached_memory / total_memory
            
            return GPUInfo(
                name=props.name,
                total_memory_gb=total_memory_gb,
                free_memory_gb=free_memory_gb,
                used_memory_gb=cached_memory_gb,
                memory_utilization=memory_utilization
            )
        except Exception as e:
            logger.warning(f"Failed to get GPU info with PyTorch: {e}")
            return None
    
    def calculate_kv_cache_size(
        self,
        seq_len: int,
        dtype: torch.dtype = torch.float16
    ) -> float:
        """Calculate memory size for KV cache in GB"""
        # Each layer has K and V tensors
        # Shape: [num_layers, seq_len, num_heads, head_dim]
        kv_size = 2 * self.num_layers * seq_len * self.num_heads * self.head_dim
        
        # Convert to bytes
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        size_bytes = kv_size * bytes_per_element
        
        # Convert to GB
        return size_bytes / (1024 ** 3)
    
    def calculate_model_memory(self, dtype: torch.dtype = torch.float16) -> float:
        """Calculate base model memory requirements in GB"""
        # Model parameters
        param_size = (
            self.num_layers * (
                4 * self.hidden_size * self.hidden_size +  # FFN
                4 * self.hidden_size * self.hidden_size +  # Attention
                2 * self.hidden_size  # Layer norms
            ) +
            self.vocab_size * self.hidden_size +  # Embedding
            self.vocab_size * self.hidden_size    # Output projection
        )
        
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        size_bytes = param_size * bytes_per_element
        
        return size_bytes / (1024 ** 3)
    
    def calculate_token_budget(
        self,
        gpu_info: Optional[GPUInfo] = None,
        chunk_size: Optional[int] = None,
        target_chunks: Optional[int] = None
    ) -> TokenBudget:
        """Calculate token budget for GPU memory"""
        
        if gpu_info is None:
            gpu_info = self.get_gpu_info()
        
        if gpu_info is None:
            # Fallback: assume 40GB GPU
            available_memory_gb = 40.0 - self.safety_margin_gb
            logger.warning("GPU info not available, using fallback memory estimate")
        else:
            # Calculate available memory
            max_usable_memory = gpu_info.total_memory_gb * self.max_memory_utilization
            available_memory_gb = max_usable_memory - self.safety_margin_gb
            
            logger.info(f"GPU: {gpu_info.name}")
            logger.info(f"Total memory: {gpu_info.total_memory_gb:.2f} GB")
            logger.info(f"Available memory: {available_memory_gb:.2f} GB")
        
        # Calculate model memory requirements
        model_memory_gb = self.calculate_model_memory()
        logger.info(f"Model memory: {model_memory_gb:.2f} GB")
        
        # Available memory for KV cache
        kv_cache_memory_gb = available_memory_gb - model_memory_gb
        
        if kv_cache_memory_gb <= 0:
            logger.error("Insufficient memory for model + KV cache")
            return TokenBudget(
                max_tokens=0,
                max_chunks=0,
                tokens_per_chunk=chunk_size or self.default_chunk_size,
                memory_required_gb=model_memory_gb,
                memory_available_gb=available_memory_gb,
                safety_margin_gb=self.safety_margin_gb
            )
        
        # Use provided chunk size or default
        tokens_per_chunk = chunk_size or self.default_chunk_size
        
        # Calculate memory per chunk
        memory_per_chunk_gb = self.calculate_kv_cache_size(tokens_per_chunk)
        logger.info(f"Memory per chunk ({tokens_per_chunk} tokens): {memory_per_chunk_gb:.4f} GB")
        
        # Calculate maximum chunks
        max_chunks = int(kv_cache_memory_gb / memory_per_chunk_gb)
        
        # If target chunks specified, adjust chunk size
        if target_chunks is not None and max_chunks < target_chunks:
            # Try to find a smaller chunk size that fits
            for test_chunk_size in [128, 64, 32, 16]:
                memory_per_chunk_gb = self.calculate_kv_cache_size(test_chunk_size)
                max_chunks_with_size = int(kv_cache_memory_gb / memory_per_chunk_gb)
                if max_chunks_with_size >= target_chunks:
                    tokens_per_chunk = test_chunk_size
                    max_chunks = max_chunks_with_size
                    break
        
        max_tokens = max_chunks * tokens_per_chunk
        
        logger.info(f"Token budget: {max_tokens} tokens ({max_chunks} chunks)")
        
        return TokenBudget(
            max_tokens=max_tokens,
            max_chunks=max_chunks,
            tokens_per_chunk=tokens_per_chunk,
            memory_required_gb=model_memory_gb + (max_chunks * memory_per_chunk_gb),
            memory_available_gb=available_memory_gb,
            safety_margin_gb=self.safety_margin_gb
        )
    
    def optimize_chunk_size(
        self,
        target_chunks: int,
        gpu_info: Optional[GPUInfo] = None
    ) -> Tuple[int, TokenBudget]:
        """Find optimal chunk size for target number of chunks"""
        
        if gpu_info is None:
            gpu_info = self.get_gpu_info()
        
        if gpu_info is None:
            available_memory_gb = 40.0 - self.safety_margin_gb
        else:
            max_usable_memory = gpu_info.total_memory_gb * self.max_memory_utilization
            available_memory_gb = max_usable_memory - self.safety_margin_gb
        
        model_memory_gb = self.calculate_model_memory()
        kv_cache_memory_gb = available_memory_gb - model_memory_gb
        
        if kv_cache_memory_gb <= 0:
            logger.error("Insufficient memory for model + KV cache")
            return self.default_chunk_size, TokenBudget(
                max_tokens=0, max_chunks=0, tokens_per_chunk=self.default_chunk_size,
                memory_required_gb=model_memory_gb, memory_available_gb=available_memory_gb,
                safety_margin_gb=self.safety_margin_gb
            )
        
        # Binary search for optimal chunk size
        min_chunk_size = 16
        max_chunk_size = 512
        
        optimal_chunk_size = min_chunk_size
        optimal_budget = None
        
        while min_chunk_size <= max_chunk_size:
            mid_chunk_size = (min_chunk_size + max_chunk_size) // 2
            memory_per_chunk_gb = self.calculate_kv_cache_size(mid_chunk_size)
            max_chunks = int(kv_cache_memory_gb / memory_per_chunk_gb)
            
            if max_chunks >= target_chunks:
                optimal_chunk_size = mid_chunk_size
                optimal_budget = TokenBudget(
                    max_tokens=max_chunks * mid_chunk_size,
                    max_chunks=max_chunks,
                    tokens_per_chunk=mid_chunk_size,
                    memory_required_gb=model_memory_gb + (max_chunks * memory_per_chunk_gb),
                    memory_available_gb=available_memory_gb,
                    safety_margin_gb=self.safety_margin_gb
                )
                min_chunk_size = mid_chunk_size + 1
            else:
                max_chunk_size = mid_chunk_size - 1
        
        if optimal_budget is None:
            # Fallback to default
            optimal_budget = self.calculate_token_budget(gpu_info, self.default_chunk_size)
        
        logger.info(f"Optimal chunk size: {optimal_chunk_size} tokens")
        return optimal_chunk_size, optimal_budget
    
    def get_memory_usage_summary(self, chunk_size: Optional[int] = None) -> Dict:
        """Get comprehensive memory usage summary"""
        gpu_info = self.get_gpu_info()
        budget = self.calculate_token_budget(gpu_info, chunk_size=chunk_size)
        
        return {
            "gpu_info": gpu_info.__dict__ if gpu_info else None,
            "model_config": {
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_size": self.hidden_size,
                "head_dim": self.head_dim,
                "vocab_size": self.vocab_size
            },
            "memory_requirements": {
                "model_memory_gb": self.calculate_model_memory(),
                "chunk_memory_gb": self.calculate_kv_cache_size(chunk_size or self.default_chunk_size),
                "safety_margin_gb": self.safety_margin_gb
            },
            "token_budget": budget.__dict__,
            "recommendations": {
                "optimal_chunk_size": budget.tokens_per_chunk,
                "max_chunks_for_h100": 5,  # Typical for H100
                "memory_efficiency": budget.memory_required_gb / budget.memory_available_gb
            }
        }

def build_prompt_from_chunks(sample, k: int) -> str:
    """Build prompt with the first k ctxs of a MuSiQue sample."""
    ctxs = sample.get("ctxs", [])[:k]
    parts = []
    for i, ch in enumerate(ctxs, 1):
        text = (ch.get("title", "") + "\n" + ch.get("text", "")).strip()
        parts.append(f"Document {i}: {text}")
    q = sample.get("question", "")
    parts.append(f"\nQuestion: {q}")
    parts.append("\nAnswer:")
    return "\n\n".join(parts)

def try_prefill_and_step(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int = 8,
    output_attentions: bool = True,
    attn_implementation: Optional[str] = "eager",  # match your main script
) -> bool:
    """
    Return True if model can prefill and generate `max_new_tokens` under heavy settings
    that mirror the main RAG run. False if CUDA OOM occurs.
    """
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        inputs = tokenizer([prompt], return_tensors="pt", padding=False, truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Apply attention implementation preference on config if supported
        if attn_implementation is not None:
            try:
                if hasattr(getattr(model, "config", object()), "attn_implementation"):
                    model.config.attn_implementation = attn_implementation
            except Exception:
                pass

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,   # keep outputs like in your analysis code
                output_attentions=output_attentions,
            )
        torch.cuda.synchronize()

        # Clean up big tensors so the next trial isn't affected
        del out, inputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return True

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return False
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda oom" in msg:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            return False
        raise



def main():
    import json, sys, argparse, os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Empirical chunk capacity on current GPU")
    parser.add_argument("--model-config", required=True, help="Path to model configuration JSON (kept for record)")
    parser.add_argument("--dataset", required=True, help="Path to MuSiQue JSON (e.g., inputs/musique_s.json)")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="HF model id/path to load")
    parser.add_argument("--precision", default="bf16", choices=["fp16","bf16","int8"], help="Load dtype")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=150, help="Limit number of samples to test")
    parser.add_argument("--verbose", action="store_true", help="Print per-step tries")
    parser.add_argument("--probe-new-tokens", type=int, default=8,
                    help="Decode this many tokens in the probe (mirrors real run)")
    parser.add_argument("--probe-output-attentions", action="store_true",
                        help="If set, enable output_attentions in the probe")
    parser.add_argument("--probe-attn-impl", default="eager",
                    help="Attention implementation to mirror (eager/flash/etc.)")
    args = parser.parse_args()

    # Load model config just to keep it in the report
    with open(args.model_config, "r") as f:
        model_cfg_file = json.load(f)
    logging.info(f"Using model configuration from {args.model_config}")

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        dataset = [dataset]
    dataset = dataset[: args.max_samples]
    if len(dataset) == 0:
        logging.error("Dataset is empty")
        sys.exit(1)

    # Device + model
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logging.error("CUDA not available but --device is CUDA")
        sys.exit(1)
    device = torch.device(args.device)

    logging.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"Loading model: {args.model} ({args.precision})")
    if args.precision in ("fp16","bf16"):
        dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=None).to(device)
    else:
        # 8-bit load for very tight GPUs
        model = AutoModelForCausalLM.from_pretrained(args.model, load_in_8bit=True, device_map="auto")
    model.eval()

    # Empirically find max chunks per sample via binary search
    per_sample = []
    global_min = None

    for i, sample in enumerate(dataset):
        total_chunks = len(sample.get("ctxs", []))
        if total_chunks == 0:
            per_sample.append({"index": i, "max_chunks_fit": 0, "total_chunks": 0})
            global_min = 0 if global_min is None else min(global_min, 0)
            continue

        lo, hi = 0, total_chunks  # invariant: lo fits, hi may fail
        # Optional quick check: if all fit, we will find it
        while lo < hi:
            mid = (lo + hi + 1) // 2  # try to fit 'mid' chunks
            prompt = build_prompt_from_chunks(sample, mid)
            ok = try_prefill_and_step(
                model, tokenizer, prompt, device,
                max_new_tokens=args.probe_new_tokens,
                output_attentions=args.probe_output_attentions,
                attn_implementation=args.probe_attn_impl,
            )
            if args.verbose:
                print(f"[sample {i}] try mid={mid} -> {'OK' if ok else 'OOM'}")
            if ok:
                lo = mid
            else:
                hi = mid - 1

        per_sample.append({"index": i, "max_chunks_fit": lo, "total_chunks": total_chunks})
        global_min = lo if global_min is None else min(global_min, lo)
        logging.info(f"Sample {i}: max_chunks_fit={lo} / total={total_chunks}")

    # Summarize
    ok_counts = [x["max_chunks_fit"] for x in per_sample]
    summary = {
        "model_config_file": model_cfg_file,
        "hf_model": args.model,
        "precision": args.precision,
        "device": args.device,
        "dataset_path": args.dataset,
        "tested_samples": len(per_sample),
        "per_sample_max_chunks_fit": per_sample,
        "global_min_chunks_fit": int(global_min),
        "recommendation_now": {
            "safe_K_for_all_samples": int(global_min),
            "note": "Safe K you can use right now without OOM, empirically measured."
        }
    }

    # Print concise result
    print("\n=== Empirical Chunk Capacity (Now) ===")
    print(f"Tested samples: {len(per_sample)}")
    print(f"Safe K for all samples now: {global_min}")
    print("First 10 per-sample results:", ok_counts[:10])

    # Save JSON report next to model-config
    out_path = args.model_config.replace(".json", "_empirical_capacity.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()