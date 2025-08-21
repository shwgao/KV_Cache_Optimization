#!/usr/bin/env python3
"""
RAG Chunk Temporal Attention Analysis (H3a & H3b)

H3a (Local Coherence): consecutive-step working sets are similar.
H3b (Global Drift): similarity of working sets decays as interval k grows.

This version:
- Loads MuSiQue-style JSON (single sample via utils.data_loader.load_rag_chunks,
  or a list of samples directly).
- For each sample: runs generation with output_attentions=True and aggregates
  attention per decoding step onto chunk spans.
- Computes H3a/H3b per sample, then aggregates across samples to plot/report.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, Iterable

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_rag_chunks  # single-sample loader

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# ------------------------- Prompt & Attention Utils -------------------------

def build_rag_context(chunks: List[str], query: str) -> str:
    parts = []
    for i, ch in enumerate(chunks):
        parts.append(f"Document {i+1}: {ch}")
    parts.append(f"\nQuestion: {query}")
    parts.append("\nAnswer:")
    return "\n\n".join(parts)

def token_len(tokenizer: AutoTokenizer, s: str) -> int:
    return tokenizer(s, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[-1]

def compute_chunk_token_spans(tokenizer: AutoTokenizer, chunks: List[str], query: str) -> List[Tuple[int,int]]:
    """
    Returns [(start, end_exclusive)] token spans for each chunk in the KEY sequence.
    Mirrors build_rag_context() exactly (including separators).
    """
    spans: List[Tuple[int,int]] = []
    cursor = 0
    for i, ch in enumerate(chunks):
        prefix = f"Document {i+1}: "
        piece = prefix + ch
        lp = token_len(tokenizer, prefix)
        lpiece = token_len(tokenizer, piece)
        start = cursor + lp
        end = cursor + lpiece
        spans.append((start, end))
        cursor += lpiece
        cursor += token_len(tokenizer, "\n\n")
    # ignore Question/Answer spans
    return spans

def aggregate_step_chunk_attention(attentions: List[Tuple[torch.Tensor, ...]],
                                  chunk_spans: List[Tuple[int,int]]) -> Dict[int, torch.Tensor]:
    """
    attentions: list over generated steps; each element is a tuple over layers of [B,H,Q,K].
    Return: {step: tensor[num_chunks]} average attention on each chunk for the new token at that step.
    """
    step_attention: Dict[int, torch.Tensor] = {}
    for t, per_layer in enumerate(attentions):
        L = torch.stack(per_layer, dim=0)     # [L,B,H,Q,K]
        A = L.mean(dim=0).mean(dim=1)         # [B,Q,K]
        a_last = A[0, -1]                     # [K]

        scores = []
        K = a_last.shape[0]
        for (s,e) in chunk_spans:
            s_clip = min(max(s, 0), K)
            e_clip = min(max(e, 0), K)
            if e_clip > s_clip:
                scores.append(a_last[s_clip:e_clip].mean())
            else:
                scores.append(torch.tensor(0.0, device=a_last.device))
        step_attention[t] = torch.stack(scores)  # [num_chunks]
    return step_attention

def working_set(scores: np.ndarray, top_k: int) -> Set[int]:
    """Indices of top_k chunks by score (descending)."""
    if scores.size == 0 or top_k <= 0:
        return set()
    idx = np.argsort(scores)[::-1][:top_k]
    return set(idx.tolist())

def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


# ------------------------- Per-sample H3a / H3b -------------------------

def h3a_local_coherence(step2scores: Dict[int, torch.Tensor], top_k: int) -> Dict[str, Any]:
    """
    J(W_t, W_{t+1}) across steps, where W_t is top_k by scores at step t.
    Returns dict with 'jaccard_scores' (list of floats), mean/median, and per-step sets (optional).
    """
    steps = sorted(step2scores.keys())
    if len(steps) < 2:
        return {"jaccard_scores": [], "mean_jaccard": 0.0, "median_jaccard": 0.0}

    W: Dict[int, Set[int]] = {}
    for t in steps:
        s = step2scores[t].detach().float().cpu().numpy()
        s_norm = s / (s.sum() + 1e-8)
        W[t] = working_set(s_norm, top_k)

    sims: List[float] = []
    for i in range(len(steps)-1):
        sims.append(jaccard(W[steps[i]], W[steps[i+1]]))

    return {
        "jaccard_scores": sims,
        "mean_jaccard": float(np.mean(sims)) if sims else 0.0,
        "median_jaccard": float(np.median(sims)) if sims else 0.0,
        "working_sets_per_step": {str(t): sorted(list(W[t])) for t in steps}
    }

def h3b_global_drift(step2scores: Dict[int, torch.Tensor], top_k: int, intervals: List[int]) -> Dict[str, Any]:
    """
    AvgSim(k) over pairs (t, t+k).
    Returns dict {'avg_similarity_per_interval': {k: value}}
    """
    steps = sorted(step2scores.keys())
    if len(steps) < 2:
        return {"avg_similarity_per_interval": {}}

    W: Dict[int, Set[int]] = {}
    for t in steps:
        s = step2scores[t].detach().float().cpu().numpy()
        s_norm = s / (s.sum() + 1e-8)
        W[t] = working_set(s_norm, top_k)

    out: Dict[int, float] = {}
    for k in intervals:
        if k <= 0:
            continue
        vals = []
        for i in range(len(steps)-k):
            vals.append(jaccard(W[steps[i]], W[steps[i+k]]))
        if vals:
            out[k] = float(np.mean(vals))
    return {"avg_similarity_per_interval": out}


# ------------------------- Dataset loading -------------------------

def extract_chunks_question_from_sample(sample: Dict[str, Any]) -> Tuple[List[str], str]:
    """MuSiQue-style extractor from a single sample dict."""
    chunks = [c.get("text", "").strip() for c in sample.get("ctxs", []) if isinstance(c, dict) and c.get("text")]
    query = sample.get("question", "")
    return chunks, query

def load_dataset_samples(json_path: str) -> List[Tuple[List[str], str]]:
    """
    Returns list of (chunks, question) for:
     - a single MuSiQue-style dict, or
     - a list of such dicts.
    Uses load_rag_chunks for single-sample files to honor the user's loader,
    and direct iteration if the file contains many samples.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[Tuple[List[str], str]] = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "ctxs" in item:
                chunks, q = extract_chunks_question_from_sample(item)
                if chunks and q:
                    samples.append((chunks, q))
        return samples

    # single-sample cases → use the shared loader
    chunks, q, _ = load_rag_chunks(json_path)
    if chunks and (q is not None):
        samples.append((chunks, q))
    return samples


# ------------------------- Plotting helpers -------------------------

def plot_h3a_hist(all_jaccards: List[float], out_path: str):
    plt.figure(figsize=(10,6))
    if SEABORN_AVAILABLE:
        sns.histplot(all_jaccards, bins=30, kde=True, stat="density")
    else:
        plt.hist(all_jaccards, bins=30, density=True, alpha=0.8)
    mean_j = np.mean(all_jaccards) if all_jaccards else 0.0
    median_j = np.median(all_jaccards) if all_jaccards else 0.0
    plt.axvline(mean_j, linestyle="--", linewidth=2, label=f"Mean {mean_j:.2f}")
    plt.axvline(median_j, linestyle="--", linewidth=2, label=f"Median {median_j:.2f}")
    plt.title("H3a: Jaccard similarity between consecutive steps (pooled over samples)")
    plt.xlabel("Jaccard(t, t+1)")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_h3b_decay(mean_per_k: Dict[int,float], std_per_k: Dict[int,float], out_path: str):
    ks = sorted(mean_per_k.keys())
    ys = [mean_per_k[k] for k in ks]
    plt.figure(figsize=(10,6))
    plt.plot(ks, ys, marker="o", linewidth=2, label="Mean similarity")
    if std_per_k:
        stds = [std_per_k.get(k, 0.0) for k in ks]
        upper = [y+s for y,s in zip(ys,stds)]
        lower = [y-s for y,s in zip(ys,stds)]
        plt.fill_between(ks, lower, upper, alpha=0.2, label="±1 std")
    plt.xscale("log")
    plt.xticks(ks, [str(k) for k in ks])
    plt.ylim(0, 1.05)
    plt.xlabel("Interval k (log scale)")
    plt.ylabel("Avg Jaccard(W_t, W_{t+k})")
    plt.title("H3b: Similarity decay vs interval k (averaged over samples)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------- Main -------------------------

def main():
    p = argparse.ArgumentParser(description="Temporal analysis of RAG chunk attention (H3a & H3b)")
    p.add_argument("--dataset-json", required=True, help="MuSiQue-style JSON (single sample or list of samples)")
    p.add_argument("--model", required=True, help="HF model id or path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--top-k-chunks", type=int, default=5)
    p.add_argument("--intervals", default="1,5,10,25,50", help="Comma-separated k values")
    p.add_argument("--out-dir", default="results/rag_temporal")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset samples
    samples = load_dataset_samples(args.dataset_json)
    if not samples:
        print("No valid samples found.")
        sys.exit(1)
    print(f"Loaded {len(samples)} samples")

    # Model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        attn_implementation="eager"
    )
    model.eval()

    # Prepare intervals
    intervals = [int(x) for x in args.intervals.split(",") if x.strip()]
    intervals = [k for k in intervals if k > 0]

    # Aggregate across samples
    pooled_jaccards: List[float] = []
    per_sample_h3a_means: List[float] = []
    per_sample_h3b_maps: List[Dict[int,float]] = []

    for si, (chunks, query) in enumerate(samples, 1):
        # Build prompt & get attentions
        prompt = build_rag_context(chunks, query)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_attentions=True,
                use_cache=True,
            )

        spans = compute_chunk_token_spans(tokenizer, chunks, query)
        step2scores = aggregate_step_chunk_attention(out.attentions, spans)  # {t: tensor[num_chunks]}

        # H3a per sample
        h3a = h3a_local_coherence(step2scores, args.top_k_chunks)
        pooled_jaccards.extend(h3a["jaccard_scores"])
        per_sample_h3a_means.append(h3a["mean_jaccard"])

        # H3b per sample
        h3b = h3b_global_drift(step2scores, args.top_k_chunks, intervals)
        per_sample_h3b_maps.append(h3b["avg_similarity_per_interval"])

        if si % 10 == 0 or si == len(samples):
            print(f"Processed {si}/{len(samples)} samples")

    # Aggregate H3b across samples (mean/std over samples at each k)
    mean_per_k: Dict[int,float] = {}
    std_per_k: Dict[int,float] = {}
    for k in intervals:
        vals = [m.get(k) for m in per_sample_h3b_maps if k in m]
        if vals:
            mean_per_k[k] = float(np.mean(vals))
            std_per_k[k] = float(np.std(vals))
        else:
            mean_per_k[k] = 0.0
            std_per_k[k] = 0.0

    # Plots
    plot_h3a_hist(pooled_jaccards, str(out_dir / "h3a_local_coherence_hist.png"))
    plot_h3b_decay(mean_per_k, std_per_k, str(out_dir / "h3b_global_drift.png"))

    # Save JSON summary
    summary = {
        "num_samples": len(samples),
        "h3a": {
            "pooled_jaccard_count": len(pooled_jaccards),
            "pooled_mean_jaccard": float(np.mean(pooled_jaccards)) if pooled_jaccards else 0.0,
            "pooled_median_jaccard": float(np.median(pooled_jaccards)) if pooled_jaccards else 0.0,
            "per_sample_mean_jaccard_mean": float(np.mean(per_sample_h3a_means)) if per_sample_h3a_means else 0.0,
            "per_sample_mean_jaccard_std": float(np.std(per_sample_h3a_means)) if per_sample_h3a_means else 0.0,
        },
        "h3b": {
            "intervals": intervals,
            "mean_per_k": {str(k): v for k, v in mean_per_k.items()},
            "std_per_k": {str(k): v for k, v in std_per_k.items()},
        }
    }
    with open(out_dir / "temporal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print("\n=== TEMPORAL ANALYSIS COMPLETE ===")
    print(f"H3a pooled mean Jaccard (t,t+1): {summary['h3a']['pooled_mean_jaccard']:.3f}")
    if intervals:
        kmin, kmax = min(intervals), max(intervals)
        sim_min = mean_per_k.get(kmin, 0.0); sim_max = mean_per_k.get(kmax, 0.0)
        decay = sim_min - sim_max
        print(f"H3b decay (mean across samples): AvgSim({kmin}) - AvgSim({kmax}) = {decay:.3f}")
    print(f"Saved plots & summary to: {out_dir}")

if __name__ == "__main__":
    main()
