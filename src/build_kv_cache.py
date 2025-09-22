#!/usr/bin/env python3

from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_cache_manager import KVCacheManager, KVCacheEntry


# ----------------------- helpers (unchanged logic) -----------------------

def _tokenize_chunk(tokenizer, text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    inputs = tokenizer([text], return_tensors="pt", padding=False, truncation=False)
    return {k: v.to(device) for k, v in inputs.items()}

def _prefill_get_past(model, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    with torch.inference_mode():
        out = model(
            **inputs,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    pkv = out.past_key_values
    return {"past_key_values": pkv}

def extract_texts(sample: Dict[str, Any]) -> List[Tuple[int, str]]:
    pairs: List[Tuple[int, str]] = []
    if isinstance(sample.get("ctxs"), list):
        for i, ch in enumerate(sample["ctxs"]):
            title = (ch.get("title") or "").strip()
            text = (ch.get("text") or "").strip()
            full = f"{title}\n{text}".strip() if title else text
            if full:
                pairs.append((i, full))
    elif isinstance(sample.get("contents"), list):
        for i, it in enumerate(sample["contents"]):
            if isinstance(it, str):
                s = it.strip()
                if s:
                    pairs.append((i, s))
            elif isinstance(it, dict):
                s = (it.get("text") or it.get("content") or "").strip()
                if s:
                    pairs.append((i, s))

    return pairs

def _save_chunk(cache_dir: str, entry: KVCacheEntry) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    cdir = os.path.join(cache_dir, entry.metadata.chunk_id)
    os.makedirs(cdir, exist_ok=True)

    torch.save(entry.keys.cpu(), os.path.join(cdir, "keys.pt"))
    torch.save(entry.values.cpu(), os.path.join(cdir, "values.pt"))
    torch.save(entry.valid_mask.cpu(), os.path.join(cdir, "valid_mask.pt"))
    meta = {
        "chunk_id": entry.metadata.chunk_id,
        "text": entry.metadata.text,
        "tokens": entry.metadata.tokens,
        "relevance_score": entry.metadata.relevance_score,
        "access_count": entry.metadata.access_count,
        "last_access_time": entry.metadata.last_access_time,
        "size_bytes": entry.metadata.size_bytes,
        "layer_count": entry.metadata.layer_count,
        "is_on_gpu": entry.metadata.is_on_gpu,
    }
    with open(os.path.join(cdir, "metadata.json"), "w") as f:
        json.dump(meta, f)

def _build_model_config(model) -> Dict[str, int]:
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", 4096)
    num_layers = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer", 32)
    num_heads = getattr(model.config, "num_attention_heads", None) or getattr(model.config, "n_head", 32)
    head_dim = hidden_size // num_heads
    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": getattr(model.config, "vocab_size", 32000),
    }

def _build_caches_for_sample(
    manager: KVCacheManager,
    model,
    tokenizer,
    sample: Dict[str, Any],
    k: int,
    device: torch.device,
    text_key_pairs: List[Tuple[int, str]],
):
    # text_key_pairs: list of (chunk_index, chunk_text) in sample order
    top_indices = sample.get("retrieved_indices")
    top_set = set(int(i) for i in top_indices[:k]) if top_indices else set()

    for idx, chunk_text in text_key_pairs:
        tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
        chunk_id = f"{sample.get('id', 'sample')}_chunk{idx}"
        if idx in top_set:
            # prefill and create real KV entry only for top-k
            inputs = _tokenize_chunk(tokenizer, chunk_text, device)
            outputs = _prefill_get_past(model, inputs)
            entry = manager.create_kv_cache_entry(
                chunk_id=chunk_id,
                text=chunk_text,
                tokens=tokens,
                relevance_score=1.0,
                model_outputs=outputs,
            )
            manager.store_chunk(entry.metadata.chunk_id, entry, priority="gpu")
        else:
            # create placeholder CPU entry without prefill
            entry = manager.create_placeholder_entry(
                chunk_id=chunk_id,
                text=chunk_text,
                tokens=tokens,
                relevance_score=0.0,
            )
            manager.store_chunk(entry.metadata.chunk_id, entry, priority="cpu")


# -------------------------- public builder --------------------------

class KVCachesBuilder:
    """
    Wraps the original build flow for easy use inside pipeline.py.

    - build(samples, ...) -> payload dict (same fields as your script)
    - build_from_retrieval_json(retrieval_json, ...) -> payload dict (convenience)
    """

    def build(
        self,
        samples: List[Dict[str, Any]],
        model_id: str,
        device: str = "cuda:0",
        top_k: int = 8,
        max_samples: int = 0,
        max_gpu_chunks: int = 0,
        max_cpu_chunks: int = 1000,
        gpu_mem_gb: float = 40.0,
        cpu_mem_gb: float = 100.0,
        dump_placements: bool = False,
        save_cache_dir: str = "",
        save_placeholders: bool = False,
        retrieval_json_path: Optional[str] = None,  # optional: for provenance in payload
        provided_tokenizer: Optional[Any] = None,
        provided_model: Optional[Any] = None,
        defer_save: bool = False,
    ) -> Dict[str, Any]:
        """
        Build KV caches for the given samples and return a payload dict.
        Preserves original logic & fields, minus argparse/printing.
        """
        # Device / model setup (reuse provided handles if available)
        torch_device = torch.device(device)
        tokenizer = provided_tokenizer if provided_tokenizer is not None else AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = provided_model if provided_model is not None else AutoModelForCausalLM.from_pretrained(model_id)
        # Ensure model is on the requested device
        model = model.to(torch_device)
        model.eval()

        # Samples gating
        if not isinstance(samples, list) or len(samples) == 0:
            raise RuntimeError("Retrieval data is empty or invalid")
        if max_samples > 0:
            samples = samples[: max_samples]

        # Manager
        model_config = _build_model_config(model)
        manager = KVCacheManager(
            model_config=model_config,
            gpu_memory_limit_gb=gpu_mem_gb,
            cpu_memory_limit_gb=cpu_mem_gb,
            max_gpu_chunks=(top_k if max_gpu_chunks == 0 else max_gpu_chunks),
            max_cpu_chunks=max_cpu_chunks,
            device=device,
            require_kernels=True,
        )

        # Build per sample
        start = time.time()
        for sample in samples:
            texts = extract_texts(sample)
            if not texts:
                continue
            _build_caches_for_sample(
                manager=manager,
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                k=top_k,
                device=torch_device,
                text_key_pairs=texts,
            )

        # Assemble payload (same structure)
        stats = manager.get_cache_stats()
        payload: Dict[str, Any] = {
            "retrieval_json": retrieval_json_path or "",  # preserved field for compatibility
            "model": model_id,
            "device": device,
            "top_k": top_k,
            "cache_stats": stats,
            "elapsed_sec": time.time() - start,
        }

        # Expose in-memory GPU KV for immediate use (avoid disk read for initial decode)
        try:
            gpu_in_memory: Dict[int, Dict[str, torch.Tensor]] = {}
            for cid, entry in manager.gpu_cache.items():
                idx = -1
                if isinstance(cid, str) and "_chunk" in cid:
                    try:
                        idx = int(cid.split("_chunk")[1])
                    except Exception:
                        idx = -1
                if idx >= 0:
                    gpu_in_memory[idx] = {"keys": entry.keys, "values": entry.values}
            payload["gpu_in_memory"] = gpu_in_memory
        except Exception:
            # Best-effort: if any error, skip exposing in-memory map
            payload["gpu_in_memory"] = {}

        # Optional saving of cache entries
        if save_cache_dir and not defer_save:
            for cid, entry in manager.gpu_cache.items():
                _save_chunk(save_cache_dir, entry)
            if save_placeholders:
                for cid, entry in manager.cpu_cache.items():
                    _save_chunk(save_cache_dir, entry)
            payload["saved_cache_dir"] = save_cache_dir

        # Optional placements dump
        if dump_placements:
            payload["placements"] = {
                "gpu": [
                    {"chunk_id": cid, "size_bytes": entry.metadata.size_bytes}
                    for cid, entry in manager.gpu_cache.items()
                ],
                "cpu": [
                    {"chunk_id": cid, "size_bytes": entry.metadata.size_bytes}
                    for cid, entry in manager.cpu_cache.items()
                ],
            }

        # If deferred save requested, attach manager handle for caller-side dumping
        if defer_save:
            payload["_manager"] = manager

        return payload

    def dump_to_dir(
        self,
        manager: KVCacheManager,
        save_cache_dir: str,
        save_placeholders: bool = True,
    ) -> None:
        os.makedirs(save_cache_dir, exist_ok=True)
        for cid, entry in manager.gpu_cache.items():
            _save_chunk(save_cache_dir, entry)
        if save_placeholders:
            for cid, entry in manager.cpu_cache.items():
                _save_chunk(save_cache_dir, entry)

    def build_from_retrieval_json(
        self,
        retrieval_json: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper: loads samples from a retrieval JSON file and calls build().
        """
        with open(retrieval_json, "r") as f:
            data = json.load(f)
        samples = data.get("results") if isinstance(data, dict) and "results" in data else data
        if not isinstance(samples, list):
            raise RuntimeError("Retrieval JSON is invalid (expected list or dict with 'results')")
        return self.build(samples=samples, retrieval_json_path=retrieval_json, **kwargs)
