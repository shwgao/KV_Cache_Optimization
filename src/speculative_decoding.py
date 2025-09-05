#!/usr/bin/env python3

from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # progress optional


# --------------------- helpers (same logic) ---------------------

def extract_texts(sample: Dict[str, Any]) -> List[Tuple[int, str]]:
    """Returns [(idx, text), ...] for the sample."""
    if "ctxs" in sample and isinstance(sample["ctxs"], list):
        out = []
        for i, c in enumerate(sample["ctxs"]):
            title = (c.get("title") or "").strip()
            text = (c.get("text") or "").strip()
            out.append((i, f"{title}\n{text}".strip() if title and text else (text or title)))
        return out
    if "texts" in sample and isinstance(sample["texts"], list):
        return [(i, t or "") for i, t in enumerate(sample["texts"])]
    raise ValueError("Sample does not contain 'ctxs' or 'texts'.")

@torch.no_grad()
def compute_query_vec(tokenizer, model, device, question: str, max_q_tokens: int = 64) -> torch.Tensor:
    ids = tokenizer.encode(str(question or ""), add_special_tokens=False)[:max_q_tokens]
    if not ids:
        return torch.zeros(model.get_input_embeddings().weight.shape[1], device=device)
    ids_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    emb = model.get_input_embeddings()(ids_t)
    return emb.mean(dim=1).squeeze(0)

@torch.no_grad()
def compute_chunk_centroids(tokenizer, model, device, texts: List[str], max_chunk_tokens: int = 32) -> torch.Tensor:
    """Returns a [N, H] tensor of per-chunk centroid embeddings from first K tokens."""
    embed = model.get_input_embeddings()
    H = embed.weight.shape[1]
    centroids = torch.zeros(len(texts), H, device=device)

    tokenized = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max_chunk_tokens)
    for i, ids in enumerate(tokenized["input_ids"]):
        if not ids:
            continue
        ids_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        e = embed(ids_t)
        centroids[i] = e.mean(dim=1).squeeze(0)
    return centroids

@torch.no_grad()
def score_chunks(qvec: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between query vector [H] and centroids [N, H] -> [N]."""
    qn = torch.linalg.norm(qvec) + 1e-6
    cn = torch.linalg.norm(centroids, dim=1) + 1e-6
    sims = (centroids @ qvec) / (cn * qn)
    return sims

def evict_from_gpu(gpu_set: set, sims: torch.Tensor, num_to_evict: int) -> List[int]:
    """
    Pick the lowest-priority (lowest similarity) chunk indices from gpu_set to evict.
    Returns a list of evicted indices.
    """
    if num_to_evict <= 0 or not gpu_set:
        return []
    ordered = sorted(list(gpu_set), key=lambda i: float(sims[i]))
    victims = ordered[:num_to_evict]
    for v in victims:
        gpu_set.discard(v)
    return victims


# --------------------- predictor (pipeline API) ---------------------

class SpeculativeChunkPredictor:
    """
    Wraps the original driver logic so pipeline.py can call it directly.
    """

    def predict(
        self,
        samples: List[Dict[str, Any]],
        model_id: str,
        device: str = "cuda:0",
        top_k: int = 5,
        steps: int = 16,
        promote_per_step: int = 2,
        max_gpu: int = 5,
        max_samples: int = 1,
        enable_progress: bool = False,
        out_path: Optional[str] = None,  # if provided, save JSON like original
    ) -> Dict[str, Any]:
        """
        Run prediction over samples and return {"trace": [...], "results": [...]}.
        """
        if not isinstance(samples, list) or not samples:
            raise RuntimeError("Retrieval data is empty or invalid")

        # sample limit
        if max_samples > 0:
            samples = samples[:max_samples]

        torch_device = torch.device(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
        model.eval()

        trace: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []

        samp_iter = samples
        if enable_progress and tqdm is not None:
            samp_iter = tqdm(samples, desc="Samples", total=len(samples))

        for si, sample in enumerate(samp_iter):
            texts_idx = extract_texts(sample)
            if not texts_idx:
                continue
            indices, texts = zip(*texts_idx)
            indices = list(indices)
            texts = list(texts)

            # initial in-use = retrieved[:top_k], bounded by max_gpu
            retrieved = list(int(i) for i in (sample.get("retrieved_indices") or []) if isinstance(i, (int, str)))
            retrieved = [int(i) for i in retrieved if 0 <= int(i) < len(texts)]
            initial = retrieved[:top_k]
            initial_gpu = initial[:max_gpu]

            # Precompute embeddings & similarities
            qvec = compute_query_vec(tokenizer, model, torch_device, sample.get("question", ""))
            centroids = compute_chunk_centroids(tokenizer, model, torch_device, texts)
            sims = score_chunks(qvec, centroids)

            # Live sets
            gpu_set = set(initial_gpu)
            cpu_set = set(i for i in indices if i not in gpu_set)
            promoted_order: List[int] = []

            step_iter = range(steps)
            if enable_progress and tqdm is not None:
                step_iter = tqdm(step_iter, desc=f"Steps (sample {si})", leave=False)

            for step in step_iter:
                if not cpu_set:
                    trace.append({
                        "sample_index": si,
                        "step": step,
                        "promoted_indices": [],
                        "evicted_indices": [],
                        "gpu_chunks": len(gpu_set),
                        "cpu_chunks": len(cpu_set),
                    })
                    continue

                # Rank CPU candidates by similarity (desc) and pick promotions
                ranked_cpu = sorted(list(cpu_set), key=lambda i: float(sims[i]), reverse=True)
                promote = ranked_cpu[:promote_per_step]

                # Evict if needed before promoting to satisfy max_gpu
                needed_slots = max(0, (len(gpu_set) + len(promote)) - max_gpu)
                evicted = evict_from_gpu(gpu_set, sims, needed_slots)
                for v in evicted:
                    cpu_set.add(v)

                # Promote
                for i in promote:
                    if i in cpu_set:
                        cpu_set.remove(i)
                    gpu_set.add(i)

                promoted_order.extend(promote)

                trace.append({
                    "sample_index": si,
                    "step": step,
                    "promoted_indices": promote,
                    "evicted_indices": evicted,
                    "gpu_chunks": len(gpu_set),   # will not exceed max_gpu
                    "cpu_chunks": len(cpu_set),
                })

            results.append({
                "sample_index": si,
                "predicted_promotion_order": promoted_order,
                "question": sample.get("question", ""),
                "initial_gpu": sorted(list(initial_gpu)),
                "max_gpu": max_gpu,
            })

        payload = {"trace": trace, "results": results}

        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)

        return payload

    def predict_from_retrieval_json(
        self,
        retrieval_json: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Convenience: load samples from retrieval JSON and call predict().
        """
        with open(retrieval_json, "r") as f:
            data = json.load(f)
        samples = data.get("results") if isinstance(data, dict) and "results" in data else data
        if not isinstance(samples, list) or not samples:
            raise RuntimeError("Retrieval JSON is empty or invalid")
        return self.predict(samples=samples, **kwargs)
