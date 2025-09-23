from __future__ import annotations
import os
import json
import time
import threading
from typing import Any, Dict, List, Tuple, Set, Optional
from queue import Queue, Empty
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm
import numpy as np

# Project helpers
from kv_cache_manager import KVCacheManager, KVCacheEntry, ChunkMetadata
from build_kv_cache import extract_texts, _tokenize_chunk, _prefill_get_past


# ------------------------------ Staging System for Per-Step Decoding ------------------------------

@dataclass
class PredictionEntry:
    """Single chunk prediction with metadata"""
    chunk_id: str
    predicted_for_token: int
    priority: float
    timestamp: float

class ChunkStaging:
    """Manages chunk predictions between scheduler runs"""
    
    def __init__(self):
        self.predictions: Dict[str, PredictionEntry] = {}
        self.preparing: Set[str] = set()
        self.cpu_ready: Set[str] = set()
        self.lock = threading.Lock()
        
    def add_prediction(self, chunk_id: str, predicted_for_token: int, priority: float):
        """Add a new chunk prediction"""
        with self.lock:
            if chunk_id in self.cpu_ready:
                return
            
            entry = PredictionEntry(
                chunk_id=chunk_id,
                predicted_for_token=predicted_for_token,
                priority=priority,
                timestamp=time.time()
            )
            
            if chunk_id not in self.predictions or entry.priority > self.predictions[chunk_id].priority:
                self.predictions[chunk_id] = entry
    
    def get_next_to_materialize(self) -> Optional[str]:
        """Get next chunk for background materialization"""
        with self.lock:
            candidates = []
            for chunk_id, entry in self.predictions.items():
                if chunk_id not in self.preparing and chunk_id not in self.cpu_ready:
                    candidates.append((chunk_id, entry.priority, entry.predicted_for_token))
            
            if not candidates:
                return None
            
            candidates.sort(key=lambda x: (-x[1], x[2]))
            return candidates[0][0]
    
    def mark_preparing(self, chunk_id: str):
        with self.lock:
            self.preparing.add(chunk_id)
    
    def mark_ready(self, chunk_id: str):
        with self.lock:
            self.preparing.discard(chunk_id)
            self.cpu_ready.add(chunk_id)
    
    def get_ready_chunks(self) -> List[Tuple[str, PredictionEntry]]:
        """Get chunks ready for GPU promotion"""
        with self.lock:
            ready_with_meta = []
            for chunk_id in self.cpu_ready:
                if chunk_id in self.predictions:
                    ready_with_meta.append((chunk_id, self.predictions[chunk_id]))
            return ready_with_meta
    
    def cleanup_old_predictions(self, current_token: int, max_age: int = 10):
        """Remove old predictions"""
        with self.lock:
            to_remove = []
            for chunk_id, entry in self.predictions.items():
                if current_token - entry.predicted_for_token > max_age:
                    to_remove.append(chunk_id)
            
            for chunk_id in to_remove:
                del self.predictions[chunk_id]
                self.preparing.discard(chunk_id)
                self.cpu_ready.discard(chunk_id)


# ------------------------------ Lightweight bandit helpers ------------------------------

def _mean_token_embed(tokenizer, model, device, text: str, max_tokens: int) -> torch.Tensor:
    ids = tokenizer.encode(text or "", add_special_tokens=False)[:max_tokens]
    if not ids:
        H = model.get_input_embeddings().weight.shape[1]
        return torch.zeros(H, device=device)
    ids_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    emb = model.get_input_embeddings()(ids_t)
    return emb.mean(dim=1).squeeze(0)

def _compute_query_vec_local(tokenizer, model, device, question: str, max_q_tokens: int = 64) -> torch.Tensor:
    return _mean_token_embed(tokenizer, model, device, str(question or ""), max_q_tokens)

def _compute_chunk_centroids_local(tokenizer, model, device, texts: List[str], max_chunk_tokens: int = 32) -> torch.Tensor:
    H = model.get_input_embeddings().weight.shape[1]
    out = torch.zeros(len(texts), H, device=device)
    for i, t in enumerate(texts):
        out[i] = _mean_token_embed(tokenizer, model, device, t, max_chunk_tokens)
    return out

def _cosine_scores(qvec: torch.Tensor, mats: torch.Tensor) -> torch.Tensor:
    qn = torch.linalg.norm(qvec) + 1e-6
    mn = torch.linalg.norm(mats, dim=1) + 1e-6
    return (mats @ qvec) / (mn * qn)

def _bandit_init(dim: int, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    A_inv = torch.eye(dim) / max(1e-6, alpha)
    theta = torch.zeros(dim)
    return A_inv, theta

def _bandit_score_ucb(A_inv: torch.Tensor, theta: torch.Tensor, x: torch.Tensor, beta: float) -> float:
    a_hat = float(theta @ x)
    ucb = float(beta * torch.sqrt(torch.clamp(x @ (A_inv @ x), min=0.0)))
    return a_hat + ucb

def _bandit_update(A_inv: torch.Tensor, theta: torch.Tensor, x: torch.Tensor, y: float) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.view(-1)
    denom = float(1.0 + (x @ (A_inv @ x)))
    A_inv = A_inv - (A_inv @ torch.outer(x, x) @ A_inv) / max(denom, 1e-6)
    err = y - float(theta @ x)
    theta = theta + (A_inv @ x) * err
    return A_inv, theta

def _make_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def lexical_overlap(text1: str, text2: str) -> float:
    """Simple lexical overlap calculation"""
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0

def _value_centroid(entry: KVCacheEntry) -> Optional[torch.Tensor]:
    try:
        v = entry.values
        # values shape: [L, S, H, D] or on GPU same
        if v is None or v.numel() == 0:
            return None
        # mean over layers, sequence and heads -> [D]
        while v.dim() > 1:
            v = v.mean(dim=0)
        return v  # [D]
    except Exception:
        return None

@torch.inference_mode()

# ------------------------------ Scheduler (pipeline API) ------------------------------

class TangoScheduler:
    """
    Wraps the original TANGO driver for direct use inside pipeline.py.
    """

    def run_per_step_decode(
        self,
        retr_samples: List[Dict[str, Any]],
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "auto",
        max_gpu: int = 5,
        max_samples: int = 1,
        max_new_tokens: int = 32,
        scheduler_interval: int = 5,  # Run heavy scheduler every N tokens
        provided_tokenizer: Optional[Any] = None,
        provided_model: Optional[Any] = None,
        promote_per_step: int = 2,
        initial_gpu_indices: List[int] = None,  # Initial GPU chunk placement
        use_original_decode: bool = False,
    ) -> Dict[str, Any]:
        """
        New per-step decoding with lightweight prediction and periodic heavy scheduling.
        """
        device_t = torch.device(device)
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "fp32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16 if (device_t.type == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability(device_t)[0] >= 8) else None

        tokenizer = provided_tokenizer
        model = provided_model
        model = model.to(device_t)
        model.eval()

        if max_samples > 0:
            retr_samples = retr_samples[:max_samples]


        results = []
        
        for si, sample in enumerate(retr_samples):
            sample_id = f"{si}"
            texts_idx: List[Tuple[int, str]] = extract_texts(sample)
            
            # Optional path: run original decoding (no KV-cache scheduling), for debugging/baseline
            if use_original_decode:
                print(f"[OriginalDecode] Sample {si}: running original step-by-step decoding without KV scheduler")
                result = self._decode_per_step_original(
                    sample=sample,
                    texts_idx=texts_idx,
                    model=model,
                    tokenizer=tokenizer,
                    device_t=device_t,
                    max_new_tokens=max_new_tokens
                )
                results.append({
                    "sample_index": si,
                    "sample_id": sample_id,
                    **result
                })
                continue

            # Initialize KV cache manager
            kv = KVCacheManager(
                model_config={
                    "hidden_size": getattr(model.config, "hidden_size", 4096),
                    "num_layers": getattr(model.config, "num_hidden_layers", 32),
                    "num_attention_heads": getattr(model.config, "num_attention_heads", 32),
                    "head_dim": getattr(model.config, "hidden_size", 4096) // max(1, getattr(model.config, "num_attention_heads", 32)),
                    "vocab_size": getattr(model.config, "vocab_size", 32000),
                },
                gpu_memory_limit_gb=40.0,
                cpu_memory_limit_gb=100.0,
                max_gpu_chunks=max_gpu,
                max_cpu_chunks=10_000,
                device=str(device_t),
                require_kernels=True,
            )
            
            # Initialize staging
            staging = ChunkStaging()
            
            # Background materialization queue
            materialization_queue = Queue()
            
            
            print(f"[PerStepDecode] Sample {si}: Using initial GPU chunks {initial_gpu_indices}")
            
            # Create initial GPU chunks with CORRECT RoPE positions
            cumulative_position = 0
            for idx in initial_gpu_indices:
                if 0 <= idx < len(texts_idx):
                    cid = f"{sample_id}_chunk{idx}"
                    text = texts_idx[idx][1]
                    
                    # Apply correct RoPE positions
                    inputs = _tokenize_chunk(tokenizer, text, device_t)
                    seq_len = inputs["input_ids"].shape[1]

                    # Create position_ids starting from cumulative position
                    position_ids = torch.arange(
                        cumulative_position, 
                        cumulative_position + seq_len, 
                        device=device_t
                    ).unsqueeze(0)
                    inputs["position_ids"] = position_ids
                    
                    print(f"[RoPE] Chunk {idx}: positions [{cumulative_position}:{cumulative_position + seq_len}]")
                    
                    outputs = _prefill_get_past(model, inputs)
                    entry = kv.create_kv_cache_entry(
                        chunk_id=cid,
                        text=text,
                        tokens=tokenizer.encode(text, add_special_tokens=False),
                        relevance_score=1.0,
                        model_outputs=outputs,
                    )

                    kv.store_chunk(cid, entry, priority="gpu")
                    cumulative_position += seq_len
            
            # Create placeholders for remaining chunks
            for idx, text in texts_idx:
                cid = f"{sample_id}_chunk{idx}"
                if cid not in kv.gpu_cache:
                    placeholder = kv.create_placeholder_entry(
                        chunk_id=cid,
                        text=text,
                        tokens=tokenizer.encode(text, add_special_tokens=False),
                        relevance_score=0.0,
                    )
                    kv.store_chunk(cid, placeholder, priority="cpu")
            
            # Prepare bandit features
            texts_only = [t for _, t in texts_idx]
            qvec = _compute_query_vec_local(tokenizer, model, device_t, sample.get("question", ""))
            centroids = _compute_chunk_centroids_local(tokenizer, model, device_t, texts_only)
            align_cos = _cosine_scores(qvec, centroids)
            
            # Token length features
            tok_lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts_only]
            max_len = max(1, max(tok_lens) if tok_lens else 1)
            
            # Initialize bandit
            feat_dim = 6
            A_inv, theta = _bandit_init(feat_dim, alpha=1.0)
            
            # Per-step decoding
            result = self._decode_per_step(
                sample=sample,
                sample_id=sample_id,
                texts_idx=texts_idx,
                kv=kv,
                staging=staging,
                materialization_queue=materialization_queue,
                model=model,
                tokenizer=tokenizer,
                device_t=device_t,
                max_new_tokens=max_new_tokens,
                scheduler_interval=scheduler_interval,
                max_gpu=max_gpu,
                promote_per_step=promote_per_step,
                # Bandit state
                A_inv=A_inv,
                theta=theta,
                align_cos=align_cos,
                tok_lens=tok_lens,
                max_len=max_len
            )
            
            results.append({
                "sample_index": si,
                "sample_id": sample_id,
                **result
            })
        
        return {"results": results}

    def convert_to_cache_format(self, past_kv_tuple, model):
        """Convert to appropriate cache format"""
        if past_kv_tuple is None:
            return None
        
        try:
            from transformers import DynamicCache
            cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(past_kv_tuple):
                cache.update(k, v, layer_idx)
            return cache
        except ImportError:
            pass
        
        # Fallback: return tuple
        return past_kv_tuple
    
    def _decode_per_step(self, sample: Dict[str, Any], sample_id: str, texts_idx: List[Tuple[int, str]], 
                   kv: KVCacheManager, staging: ChunkStaging, materialization_queue: Queue,
                   model, tokenizer, device_t: torch.device, max_new_tokens: int, 
                   scheduler_interval: int, max_gpu: int, promote_per_step: int,
                   A_inv: torch.Tensor, theta: torch.Tensor, align_cos: torch.Tensor, 
                   tok_lens: List[int], max_len: int) -> Dict[str, Any]:
        
        # Extract and display the question
        question_text = sample.get("question", "")
        suffix = f"Question: {question_text}"
        question = suffix.split("Question: ")[1].strip() if "Question: " in suffix else suffix
        print(f"Question: {question}")
        
        # Background materialization worker
        def background_worker():
            while True:
                try:
                    chunk_id = materialization_queue.get(timeout=0.1)
                    if chunk_id is None:  # Shutdown signal
                        break
                    self._materialize_chunk_background(chunk_id, kv, staging, tokenizer, model, device_t, sample_id, texts_idx)
                    materialization_queue.task_done()
                except Empty:
                    continue
                except Exception as e:
                    print(f"Background: Error materializing: {e}")
        
        worker_thread = threading.Thread(target=background_worker, daemon=True)
        worker_thread.start()
        
        try:
            # CRITICAL FIX: Tokenize question to place at position 0
            enc = tokenizer(suffix, return_tensors="pt", padding=True, truncation=False)
            question_input_ids = enc["input_ids"].to(device_t)
            question_length = question_input_ids.shape[1]
            
            print(f"Question tokens: {question_length}, will be positioned at [0:{question_length}]")
            
            generated_tokens = []
            trace = []
            start_time = time.time()
            first_token_time = None
            bw_gbps = 5.0  # EMA bandwidth estimate
            
            for token_step in range(max_new_tokens):
                current_gpu_chunks = set(kv.gpu_cache.keys())
                
                with torch.no_grad():
                    if token_step == 0:
                        # FIRST TOKEN GENERATION WITH CORRECTED POSITION ENCODING
                        
                        # 1. Get cached chunks KV
                        past_kv_chunks = self.build_past_key_values_from_kv(kv, current_gpu_chunks)
                        
                        if past_kv_chunks is not None:
                            # Get chunk sequence length  
                            chunk_seq_len = past_kv_chunks[0][0].shape[-2]
                            print(f"Cached chunks: {chunk_seq_len} tokens, will be positioned at [{question_length}:{question_length + chunk_seq_len}]")
                            
                            # CRITICAL FIX: Question at position 0, chunks after question
                            # Total context: [Question][Chunks] 
                            total_context_len = question_length + chunk_seq_len
                            
                            # Create attention mask for entire context
                            attention_mask = torch.ones(1, total_context_len, device=device_t)
                            
                            # Position IDs for question only (chunks already positioned in KV cache)
                            position_ids = torch.arange(0, question_length, device=device_t).unsqueeze(0)
                            
                            # Model inputs with question first
                            model_inputs = {
                                "input_ids": question_input_ids,
                                "attention_mask": attention_mask,
                                "position_ids": position_ids,
                                "use_cache": True,
                                "return_dict": True
                            }
                            
                            # Add cached chunks as past_key_values
                            try:
                                cache_format = self.convert_to_cache_format(past_kv_chunks, model)
                                if cache_format is not None:
                                    model_inputs["past_key_values"] = cache_format
                                    print(f"Using cached KV with {len(past_kv_chunks)} layers")
                                else:
                                    print("Failed to convert KV cache format, using question only")
                            except Exception as e:
                                print(f"KV cache conversion error: {e}, falling back to question only")
                        
                        else:
                            # No cached chunks available
                            print("No cached chunks available, using question only")
                            model_inputs = {
                                "input_ids": question_input_ids,
                                "attention_mask": torch.ones_like(question_input_ids),
                                "use_cache": True,
                                "return_dict": True
                            }
                    
                    else:
                        
                        # Use last generated token as input
                        last_token = torch.tensor([generated_tokens[-1]], device=device_t)
                        input_token = last_token.unsqueeze(0)
                        
                        # Get current cached chunks
                        past_kv_chunks = self.build_past_key_values_from_kv(kv, current_gpu_chunks)
                        
                        if past_kv_chunks is not None:
                            chunk_seq_len = past_kv_chunks[0][0].shape[-2]
                            
                            # CRITICAL FIX: Calculate next position correctly
                            # Sequence: [Question: 0 to question_length] [Chunks: question_length to question_length+chunk_seq_len] [Generated: continues from there]
                            next_position = question_length + chunk_seq_len + token_step - 1
                            
                            position_ids = torch.tensor([[next_position]], device=device_t)
                            
                            # Total context includes question + chunks + generated tokens so far
                            total_context_len = question_length + chunk_seq_len + token_step
                            attention_mask = torch.ones(1, total_context_len, device=device_t)
                            
                            model_inputs = {
                                "input_ids": input_token,
                                "attention_mask": attention_mask,
                                "position_ids": position_ids,
                                "use_cache": True,
                                "return_dict": True
                            }
                            
                            # Add cached chunks
                            try:
                                cache_format = self.convert_to_cache_format(past_kv_chunks, model)
                                if cache_format is not None:
                                    model_inputs["past_key_values"] = cache_format
                            except Exception as e:
                                print(f"Token {token_step}: KV cache error: {e}")
                        
                        else:
                            # Fallback: no cache available
                            model_inputs = {
                                "input_ids": input_token,
                                "attention_mask": torch.ones_like(input_token),
                                "use_cache": True,
                                "return_dict": True
                            }
                    
                    # Generate next token
                    outputs = model(**model_inputs)
                    next_token_logits = outputs.logits[0, -1, :]
                    next_token = torch.argmax(next_token_logits).item()
                    generated_tokens.append(next_token)
                    
                    # Decode and display progress
                    decoded_token = tokenizer.decode(next_token, skip_special_tokens=True)
                    print(f"Token {token_step}: '{decoded_token}'", end=" " if token_step % 5 != 4 else "\n")
                    
                    if first_token_time is None:
                        first_token_time = time.time()
                    
                    # 2. Chunk prediction using bandit algorithm
                    predictions, A_inv, theta = self.predict_chunks_bandit(
                        sample_id=sample_id, texts_idx=texts_idx, kv=kv, 
                        A_inv=A_inv, theta=theta, align_cos=align_cos, 
                        tok_lens=tok_lens, max_len=max_len, sample=sample, 
                        promote_per_step=promote_per_step, current_token_step=token_step, generated_tokens=generated_tokens
                    )
                    
                    if predictions:
                        pred_chunks = [p[0] if isinstance(p, tuple) else p for p in predictions]
                        print(f"\nPredict T{token_step}: {pred_chunks}")
                    
                    # 3. Stage predictions for future materialization
                    for chunk_idx, priority in predictions:
                        chunk_id = f"{sample_id}_chunk{chunk_idx}"
                        staging.add_prediction(chunk_id, token_step + 1, priority)
                    
                    # 4. Request background materialization of next chunk
                    next_to_materialize = staging.get_next_to_materialize()
                    if next_to_materialize:
                        materialization_queue.put(next_to_materialize)
                    
                    # 5. Run heavy scheduler periodically
                    if token_step % scheduler_interval == 0:
                        gpu_before = len(kv.gpu_cache)
                        ready_before = len(staging.get_ready_chunks())
                        self.run_heavy_scheduler_step(kv, staging, max_gpu, token_step, bw_gbps)
                        gpu_after = len(kv.gpu_cache)
                        ready_after = len(staging.get_ready_chunks())
                        print(f"\nScheduler T{token_step}: GPU {gpu_before}->{gpu_after}, Ready {ready_before}->{ready_after}")
                    
                    # 6. Cleanup old predictions
                    staging.cleanup_old_predictions(token_step)
                    
                    # Record trace for debugging
                    trace.append({
                        "token_step": token_step,
                        "token": next_token,
                        "decoded_token": decoded_token,
                        "predictions": predictions,
                        "gpu_chunks": len(kv.gpu_cache),
                        "cpu_chunks": len(kv.cpu_cache),
                        "staging_stats": {
                            "predictions": len(staging.predictions),
                            "preparing": len(staging.preparing),
                            "ready": len(staging.cpu_ready)
                        }
                    })
                    
                    # 7. Check stopping conditions
                    if next_token == tokenizer.eos_token_id:
                        print(f"\nStop: EOS token at step {token_step}")
                        break
                    
                    # Stop on repetitive patterns
                    if len(generated_tokens) >= 3:
                        if generated_tokens[-1] == generated_tokens[-2] == generated_tokens[-3]:
                            print(f"\nStop: Repetitive pattern detected at step {token_step}")
                            break
                    
                    # Max tokens reached
                    if token_step == max_new_tokens - 1:
                        print(f"\nStop: Maximum tokens ({max_new_tokens}) reached")
                        break
            
            # Shutdown background worker
            materialization_queue.put(None)
            worker_thread.join(timeout=1.0)
            
            # Calculate timing and performance metrics
            end_time = time.time()
            total_time = end_time - start_time
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Display final results
            print(f"\nFinal Answer: {generated_text.strip()}")
            print(f"Performance: Generated {len(generated_tokens)} tokens in {total_time:.2f}s")
            
            # Calculate TPOT (Time Per Output Token)
            tpot = total_time / len(generated_tokens) if len(generated_tokens) > 0 else 0.0
            ttft = first_token_time - start_time if first_token_time else 0.0
            
            return {
                "answer": generated_text.strip(),
                "ttft": ttft,
                "e2e_latency": total_time,
                "throughput": len(generated_tokens) / total_time if total_time > 0 else 0,
                "tpot": tpot,
                "decoded_tokens": len(generated_tokens),
                "trace": trace
            }
        
        finally:
            # Ensure cleanup
            try:
                materialization_queue.put(None)
                worker_thread.join(timeout=1.0)
            except:
                pass

    def run_heavy_scheduler_step(self, kv: KVCacheManager, staging: ChunkStaging, 
                           max_gpu: int, token_step: int, bw_gbps: float):
        """
        GENERALIZED FIX: Heavy scheduler with intelligent eviction for all question types
        """
        try:
            ready_chunks = staging.get_ready_chunks()
            current_gpu_chunks = list(kv.gpu_cache.keys())
            
            print(f"HeavyScheduler: {len(ready_chunks)} ready chunks: {[cid.split('_')[-1] for cid, _ in ready_chunks]}")
            print(f"HeavyScheduler: Current GPU {len(current_gpu_chunks)}/{max_gpu}: {[cid.split('_')[-1] for cid in current_gpu_chunks]}")
            
            # CRITICAL FIX: Unified priority calculation for ALL chunks (GPU + CPU)
            all_candidate_chunks = []
            
            # Current GPU chunks with decay-based priority
            for chunk_id in current_gpu_chunks:
                if chunk_id in staging.predictions:
                    # Recent predictions get higher priority
                    prediction_entry = staging.predictions[chunk_id]
                    base_priority = prediction_entry.priority
                    
                    # Time-based decay: older predictions lose priority
                    time_decay = max(0.1, 1.0 - (token_step - prediction_entry.predicted_for_token) * 0.1)
                    final_priority = base_priority * time_decay
                else:
                    # Legacy chunks without recent predictions get low priority
                    final_priority = 0.2
                
                all_candidate_chunks.append((chunk_id, final_priority, "gpu"))
            
            # Ready CPU chunks with their prediction priority
            for chunk_id, pred_entry in ready_chunks:
                # Use bandit-predicted priority directly (already incorporates relevance)
                final_priority = pred_entry.priority
                all_candidate_chunks.append((chunk_id, final_priority, "cpu"))
            
            # CRITICAL: Select top max_gpu chunks by priority (enables eviction)
            all_candidate_chunks.sort(key=lambda x: x[1], reverse=True)
            target_gpu_chunks = all_candidate_chunks[:max_gpu]
            
            target_gpu_ids = set(cid for cid, _, _ in target_gpu_chunks)
            current_gpu_ids = set(current_gpu_chunks)
            
            print(f"HeavyScheduler: Target GPU chunks: {[(cid.split('_')[-1], f'{priority:.2f}') for cid, priority, location in target_gpu_chunks]}")
            
            # Step 1: Promote worthy CPU chunks
            promoted_count = 0
            for chunk_id, priority, location in target_gpu_chunks:
                if location == "cpu" and chunk_id not in current_gpu_ids:
                    if chunk_id in kv.cpu_cache:
                        try:
                            # Transfer CPU → GPU
                            entry = kv.cpu_cache[chunk_id]
                            entry.keys = entry.keys.to(kv.device)
                            entry.values = entry.values.to(kv.device)
                            kv.gpu_cache[chunk_id] = entry
                            del kv.cpu_cache[chunk_id]
                            
                            # Update metadata
                            if hasattr(entry.metadata, 'is_on_gpu'):
                                entry.metadata.is_on_gpu = True
                            
                            promoted_count += 1
                            print(f"HeavyScheduler: Promoted {chunk_id.split('_')[-1]} (priority={priority:.2f})")
                            
                            # Clean up staging
                            staging.cpu_ready.discard(chunk_id)
                            
                        except Exception as e:
                            print(f"HeavyScheduler: Failed to promote {chunk_id}: {e}")
            
            # Step 2: Evict unworthy GPU chunks
            evicted_count = 0
            for chunk_id in list(current_gpu_ids):
                if chunk_id not in target_gpu_ids:
                    try:
                        # Transfer GPU → CPU
                        entry = kv.gpu_cache[chunk_id]
                        entry.keys = entry.keys.cpu()
                        entry.values = entry.values.cpu()
                        kv.cpu_cache[chunk_id] = entry
                        del kv.gpu_cache[chunk_id]
                        
                        # Update metadata
                        if hasattr(entry.metadata, 'is_on_gpu'):
                            entry.metadata.is_on_gpu = False
                        
                        evicted_count += 1
                        print(f"HeavyScheduler: Evicted {chunk_id.split('_')[-1]}")
                        
                    except Exception as e:
                        print(f"HeavyScheduler: Failed to evict {chunk_id}: {e}")
            
            # Final status
            final_gpu_chunks = list(kv.gpu_cache.keys())
            print(f"HeavyScheduler: Final GPU chunks: {[cid.split('_')[-1] for cid in final_gpu_chunks]}")
            print(f"HeavyScheduler: Changes: +{promoted_count} promoted, -{evicted_count} evicted")
            
            return promoted_count, evicted_count
            
        except Exception as e:
            print(f"HeavyScheduler: Error: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0

    
    def predict_chunks_bandit(self, sample_id: str, texts_idx: List[Tuple[int, str]], 
                         kv: KVCacheManager, A_inv: torch.Tensor, theta: torch.Tensor, 
                         align_cos: torch.Tensor, tok_lens: List[int], max_len: int, 
                         sample: Dict[str, Any], promote_per_step: int, 
                         current_token_step: int, generated_tokens: List[int] = None) -> Tuple[List[Tuple[int, float]], torch.Tensor, torch.Tensor]:
        """
        FIXED: Bandit algorithm with proper exploration, reward feedback, and diversity
        """
        
        def chunk_features(idx: int, now_ts: float) -> torch.Tensor:
            """Enhanced feature computation without tokenizer dependency"""
            cid = f"{sample_id}_chunk{idx}"
            
            # Enhanced retrieval score with lexical overlap (NO TOKENIZER NEEDED)
            try:
                question_text = sample.get('question', '')
                if question_text and idx < len(texts_idx):
                    chunk_text = texts_idx[idx][1][:300]  # First 200 chars
                    retrscore = lexical_overlap(question_text.lower(), chunk_text.lower())
                else:
                    retrscore = float(align_cos[idx]) if 0 <= idx < len(align_cos) else 0.0
            except Exception:
                retrscore = float(align_cos[idx]) if 0 <= idx < len(align_cos) else 0.0
                
            # Normalized chunk length
            loglen = float(torch.log(torch.tensor(tok_lens[idx] + 1.0))) / float(torch.log(torch.tensor(max_len + 1.0)))
            
            # GPU status (reduced bias)
            ongpu = 0.1 if cid in kv.gpu_cache else 0.0  # Reduced from 1.0
            
            # Access patterns
            entry = kv.gpu_cache.get(cid) or kv.cpu_cache.get(cid)
            if entry is not None:
                acc = float(getattr(entry.metadata, 'accesscount', 0.0))
                accnorm = min(1.0, acc / 5.0)
                lastt = float(getattr(entry.metadata, 'lastaccesstime', 0.0))
                recency = max(0.1, 1.0 / (1.0 + now_ts - lastt))  # Decay function
            else:
                accnorm = 0.0
                recency = 1.0  # New chunks get high recency
                
            # Position-based diversity feature (encourages exploring different positions)
            position_diversity = 0.5 + 0.5 * np.sin(idx * 0.1)  # Sinusoidal variation
                
            return torch.tensor([retrscore, loglen, ongpu, accnorm, recency, position_diversity], dtype=torch.float32)
    
        
        now_ts = time.time()
        current_gpu_idx = [int(cid.split("chunk")[1]) for cid in kv.gpu_cache.keys() if "chunk" in cid]
        cpu_candidates = [i for i, _ in texts_idx if i not in current_gpu_idx]
        
        if not cpu_candidates:
            return [], A_inv, theta
        
        # ENHANCED EXPLORATION: Dynamic beta based on token step
        base_beta = 0.9  # Higher exploration initially
        decay_rate = 0.95
        beta = max(0.4, base_beta * (decay_rate ** (current_token_step / 10)))
        
        scored = []
        features_used = []
        
        for i in cpu_candidates:
            x = chunk_features(i, now_ts)
            v_base = _bandit_score_ucb(A_inv, theta, x, beta=beta)
            scored.append((i, v_base))
            features_used.append((i, x))
        
        # DIVERSITY ENHANCEMENT: Add randomization to break ties
        scored.sort(key=lambda t: t[1] + np.random.normal(0, 0.1), reverse=True)
        
        # Select diverse chunks (avoid clustering)
        selected = []
        for chunk_idx, score in scored:
            if len(selected) >= promote_per_step:
                break
            # Ensure diversity - don't select adjacent chunks
            if not selected or all(abs(chunk_idx - s[0]) > 1 for s in selected):
                selected.append((chunk_idx, score))
        
        # Fill remaining slots if needed
        remaining_slots = promote_per_step - len(selected)
        if remaining_slots > 0:
            remaining_candidates = [x for x in scored if x[0] not in [s[0] for s in selected]]
            selected.extend(remaining_candidates[:remaining_slots])
        
        print(f"Bandit: Explored {len(cpu_candidates)} candidates, beta={beta:.3f}, selected={[s[0] for s in selected]}")
        
        return selected, A_inv, theta
    
    def _materialize_chunk_background(self, chunk_id: str, kv: KVCacheManager, staging: ChunkStaging, 
                               tokenizer, model, device_t: torch.device, sample_id: str, 
                               texts_idx: List[Tuple[int, str]]):
        """
        FIXED: Background chunk materialization with proper error handling
        """
        staging.mark_preparing(chunk_id)
        
        try:
            # Parse chunk ID properly
            if "_chunk" in chunk_id:
                # New format: "0_chunk7"
                parts = chunk_id.split("_chunk")
                if len(parts) != 2:
                    print(f"Background: Invalid chunk_id format: {chunk_id}")
                    staging.preparing.discard(chunk_id)
                    return
                idx = int(parts[1])

            else:
                print(f"Background: No 'chunk' found in ID: {chunk_id}")
                staging.preparing.discard(chunk_id)
                return
            
            # Create texts dictionary efficiently
            texts_dict = {i: text for i, text in texts_idx}
            if idx not in texts_dict:
                print(f"Background: Chunk index {idx} not found in texts_idx")
                return
                
            text = texts_dict[idx]
            
            # Check if already materialized in CPU cache
            if chunk_id in kv.cpu_cache:
                entry = kv.cpu_cache[chunk_id]
                if entry.keys is not None and entry.keys.numel() > 0:
                    print(f"Background: Chunk {chunk_id} already materialized")
                    staging.mark_ready(chunk_id)
                    return
            
            print(f"Background: Materializing chunk {chunk_id} (idx={idx})")
            
            # Tokenize and create KV cache entry
            with torch.inference_mode():
                inputs = _tokenize_chunk(tokenizer, text, device_t)
                seq_len = inputs["input_ids"].shape[1]
                
                # Use estimated positions for background chunks
                avg_chunk_length = 500  # Rough estimate
                estimated_start_position = idx * avg_chunk_length
                
                position_ids = torch.arange(
                    estimated_start_position, estimated_start_position + seq_len, 
                    device=device_t
                ).unsqueeze(0)
                inputs["position_ids"] = position_ids
                
                print(f"RoPE: Background chunk {chunk_id} estimated positions [{estimated_start_position}:{estimated_start_position + seq_len}]")
                
                # Generate KV cache
                outputs = _prefill_get_past(model, inputs)
                
                # Create cache entry
                entry = kv.create_kv_cache_entry(
                    chunk_id=chunk_id,
                    text=text,
                    tokens=tokenizer.encode(text, add_special_tokens=False),
                    relevance_score=0.0,
                    model_outputs=outputs
                )
                
                # Store in CPU cache
                kv.store_chunk(chunk_id, entry, priority="cpu")
                entry.metadata.is_on_gpu = False
                
                # Mark as ready for promotion
                staging.mark_ready(chunk_id)
                
                print(f"Background: Successfully materialized {chunk_id}")
                
        except Exception as e:
            print(f"Background: Failed to materialize {chunk_id}: {e}")
            staging.preparing.discard(chunk_id)  # Remove from preparing set

    def build_past_key_values_from_kv(self, kv: KVCacheManager, current_gpu_chunks: Set[str]):
        """Proper KV cache concatenation"""
        if not current_gpu_chunks:
            return None
            
        entries = []
        for chunk_id in sorted(current_gpu_chunks):
            if chunk_id in kv.gpu_cache:
                entry = kv.gpu_cache[chunk_id]
                entries.append((chunk_id, entry))
        
        if not entries:
            return None
        
        # Get layer count from first entry
        first_entry = entries[0][1]
        num_layers = first_entry.keys.shape[0] if first_entry.keys.ndim >= 3 else 0
        if num_layers == 0:
            return None
        
        print(f"KVCache: Building past_key_values from {len(entries)} chunks")
        past_key_values = []
        total_seq_len = 0
        
        for layer in range(num_layers):
            layer_keys = []
            layer_values = []
            
            for chunk_id, entry in entries:
                # Expected: entry.keys/values are [num_layers, seq_len, num_heads, head_dim]
                if entry.keys.ndim == 4:  # [L, S, H, D]
                    k = entry.keys[layer]  # [S, H, D]
                    v = entry.values[layer]  # [S, H, D]
                    
                    # Convert to [B, H, S, D] format (B=1 for single batch)
                    if k.ndim == 3:  # [S, H, D] -> [1, H, S, D]
                        k = k.permute(1, 0, 2).unsqueeze(0)  # [H, S, D] -> [1, H, S, D]
                        v = v.permute(1, 0, 2).unsqueeze(0)  # [H, S, D] -> [1, H, S, D]
                else:
                    print(f"KVCache: Skipping chunk {chunk_id} with unexpected shape {entry.keys.shape}")
                    continue
                
                layer_keys.append(k)
                layer_values.append(v)
            
            if not layer_keys:
                continue
            
            # FIXED: Proper concatenation along sequence dimension
            merged_k = torch.cat(layer_keys, dim=-2)  # [B, H, total_seq, D]
            merged_v = torch.cat(layer_values, dim=-2)  # [B, H, total_seq, D]
            past_key_values.append((merged_k, merged_v))
            
            if layer == 0:  # Debug print for first layer only
                total_seq_len = merged_k.shape[-2]
                print(f"KVCache: Layer {layer} merged KV shape {merged_k.shape}, total_seq={total_seq_len}")
        
        if past_key_values:
            print(f"KVCache: Built past_key_values with {len(past_key_values)} layers, total sequence length {total_seq_len}")
            return tuple(past_key_values)
        else:
            print("KVCache: Failed to build past_key_values - no valid entries")
            return None

    def _decode_per_step_original(self, sample: Dict[str, Any], texts_idx: List[Tuple[int, str]],
                                  model, tokenizer, device_t: torch.device, max_new_tokens: int) -> Dict[str, Any]:
        """
        Original step-by-step decoding adapted from debug.py, without KV scheduler.
        Uses current model/tokenizer and builds a simple passages+question prompt.
        """
        # Build prompt roughly matching debug.py style
        prefix_prompt = (
            "You will be asked a question after reading several passages. "
            "Please directly answer the question based on the given passages. "
            "Do NOT repeat the question. The answer should be within 10 words..\nPassages:\n"
        )
        query_prompt = (
            "\n\nAnswer the question directly based on the given passages. "
            "Do NOT repeat the question. The answer should be within 10 words. \nQuestion:"
        )

        passages = [t for _, t in texts_idx]
        question_text = str(sample.get("question", "")).strip()

        # Create list of document prompts similar to debug.py's build_qa_prompt output
        doc_prompts = passages
        q_prompt = query_prompt + question_text

        doc_chunk_ids = [tokenizer.encode(doc, add_special_tokens=True)[1:] for doc in doc_prompts]
        q_ids = tokenizer.encode(q_prompt, add_special_tokens=True)[1:]

        # Static tokens copied from debug.py for Mistral; leave as-is for baseline testing
        s_start_full = [733, 16289, 28793] + tokenizer.encode(prefix_prompt, add_special_tokens=True)[1:]
        s_start = []
        s_start_1_len = len(s_start) + 1
        s_end = [733, 28748, 16289, 28793]

        # Assemble input ids as in debug.py
        doc_chunk_ids = [s_start + chunk_ids for chunk_ids in doc_chunk_ids]
        doc_chunk_ids = [s_start_full] + doc_chunk_ids
        doc_chunk_ids = doc_chunk_ids + [s_start + q_ids + s_end]

        input_ids_list: List[int] = []
        for i in range(len(doc_chunk_ids)):
            if i == 0:
                temp_ids = doc_chunk_ids[i]
            else:
                temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
            input_ids_list += temp_ids

        # Run original stepwise generation with model cache only
        generated_tokens: List[int] = []
        trace: List[Dict[str, Any]] = []
        start_time = time.time()
        first_token_time: Optional[float] = None

        with torch.no_grad():
            current_input = torch.tensor([input_ids_list], device=device_t)
            past_key_values = None

            for step in range(int(max_new_tokens)):
                if step == 0:
                    outputs = model(
                        current_input,
                        use_cache=True,
                        return_dict=True
                    )
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(next_token_logits / 0.7, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    outputs = model(
                        next_token.unsqueeze(-1),
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(next_token_logits / 0.7, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

                token_id = int(next_token.item())

                if first_token_time is None:
                    first_token_time = time.time()

                if token_id == getattr(tokenizer, 'eos_token_id', None):
                    break

                generated_tokens.append(token_id)
                decoded_piece = tokenizer.decode([token_id], skip_special_tokens=True)
                trace.append({"token_step": step, "token": token_id, "decoded_token": decoded_piece})

        end_time = time.time()
        total_time = end_time - start_time
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        tpot = (total_time / len(generated_tokens)) if generated_tokens else 0.0
        ttft = (first_token_time - start_time) if first_token_time else 0.0

        return {
            "answer": generated_text,
            "ttft": ttft,
            "e2e_latency": total_time,
            "throughput": (len(generated_tokens) / total_time) if total_time > 0 else 0.0,
            "tpot": tpot,
            "decoded_tokens": len(generated_tokens),
            "trace": trace
        }