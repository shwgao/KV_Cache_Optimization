#!/usr/bin/env python3
"""
RAG Chunk Attention Sparsity Analysis (H2a)

This version supports:
- Single sample: one (chunks, question) -> heatmap + metrics JSON
- Dataset: list of samples with {"ctxs":[...], "question": "..."} -> per-query metrics,
  averaged curves across queries, and optional per-sample heatmaps.

Key changes vs prior:
- NO hooks. We use `outputs.attentions` and do span-aware aggregation onto chunk token spans
  (same approach as the coverage script) for reliable per-step chunk scores.
- Removed --chunks-dir. Use --chunks-json for single or dataset.
- New flags: --dataset-save-heatmaps (optional), --sample-limit.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent for project imports if needed (kept for compatibility, not strictly used here)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:
    SEABORN_AVAILABLE = False

# ---------------- Utils: data loading ----------------

def _extract_chunks_and_question_from_obj(obj: dict) -> Tuple[List[str], Optional[str]]:
    """
    Accepts a single Musique-style object with keys:
      {"ctxs":[{"text":...}, ...], "question": "...", ...}
    Returns (chunks, question)
    """
    chunks = []
    if isinstance(obj, dict) and "ctxs" in obj:
        chunks = [c.get("text", "").strip() for c in obj["ctxs"] if c.get("text")]
    q = obj.get("question") if isinstance(obj, dict) else None
    return chunks, q

def load_single_or_dataset(path: str,
                           override_query: Optional[str]) -> Tuple[bool, List[Tuple[List[str], str]]]:
    """
    Returns (is_dataset, samples)
      - is_dataset=False: one sample [(chunks, query)]
      - is_dataset=True:  many samples [(chunks, query), ...]
    """
    with open(path, "r") as f:
        data = json.load(f)

    samples: List[Tuple[List[str], str]] = []

    # Case A: a single object with ctxs/question
    if isinstance(data, dict) and "ctxs" in data:
        chunks, file_q = _extract_chunks_and_question_from_obj(data)
        q = override_query or (file_q or "")
        if not chunks or not q:
            raise ValueError("Single-sample JSON: missing chunks or question.")
        samples.append((chunks, q))
        return False, samples

    # Case B: a list of objects (dataset)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "ctxs" in data[0]:
        for obj in data:
            chunks, file_q = _extract_chunks_and_question_from_obj(obj)
            q = override_query or (file_q or "")
            if chunks and q:
                samples.append((chunks, q))
        if not samples:
            raise ValueError("Dataset JSON had no valid (chunks, question) pairs.")
        return True, samples

    # Case C: list of raw chunks (single sample); require query
    if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], str)):
        if not override_query:
            raise ValueError("List-of-strings JSON given; please provide --query.")
        chunks = [str(x).strip() for x in data if str(x).strip()]
        samples.append((chunks, override_query))
        return False, samples

    # Case D: dict with "chunks" array (single sample)
    if isinstance(data, dict) and "chunks" in data:
        chunks = [str(x).strip() for x in data["chunks"] if str(x).strip()]
        if not override_query:
            raise ValueError("Dict with 'chunks' given; please provide --query.")
        samples.append((chunks, override_query))
        return False, samples

    raise ValueError("Unrecognized JSON format for --chunks-json.")

# ---------------- Core analyzer ----------------

class RAGChunkSparsityAnalyzer:
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation="eager",
        )
        self.model.eval()

    # ---- Prompt and span mapping (match coverage script style) ----

    def _build_rag_context(self, chunks: List[str], query: str) -> str:
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(f"Document {i+1}: {chunk}")
        parts.append(f"\nQuestion: {query}")
        parts.append("\nAnswer:")
        return "\n\n".join(parts)

    def _compute_chunk_token_spans(self, chunks: List[str]) -> List[Tuple[int, int]]:
        """
        Returns token spans [(start, end_exclusive), ...] in the *key* sequence
        for each chunk. This mirrors _build_rag_context exactly and intentionally
        ignores the final 'Question/Answer' segment for chunk spans.
        """
        spans: List[Tuple[int, int]] = []
        cursor = 0

        def tlen(s: str) -> int:
            return self.tokenizer(s, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[-1]

        for i, ch in enumerate(chunks):
            prefix = f"Document {i+1}: "
            piece = prefix + ch
            len_prefix = tlen(prefix)
            len_piece = tlen(piece)
            start = cursor + len_prefix
            end = cursor + len_piece
            spans.append((start, end))
            cursor += len_piece
            cursor += tlen("\n\n")  # separator

        # we skip spans for question/answer tail
        return spans

    def _aggregate_step_chunk_attention(self, attentions, chunk_spans: List[Tuple[int, int]]) -> Dict[int, torch.Tensor]:
        """
        attentions: list[step] of tuples(layers) of tensors [B,H,Q,K]
        Returns {step: tensor[num_chunks]} where values are mean attention over each chunk span.
        """
        step_attention: Dict[int, torch.Tensor] = {}
        for t, per_layer in enumerate(attentions):
            # [L, B, H, Q, K] -> mean over layers & heads -> [B, Q, K]
            L = torch.stack(per_layer, dim=0)    # [L,B,H,Q,K]
            A = L.mean(dim=0).mean(dim=1)        # [B,Q,K]
            a_last = A[0, -1]                    # [K] — attention for the token generated at step t

            scores = []
            K = a_last.shape[0]
            for (s, e) in chunk_spans:
                s_clip = min(max(s, 0), K)
                e_clip = min(max(e, 0), K)
                if e_clip > s_clip:
                    scores.append(a_last[s_clip:e_clip].mean())
                else:
                    scores.append(torch.tensor(0.0, device=a_last.device))
            step_attention[t] = torch.stack(scores)  # [num_chunks]
        return step_attention

    # ---- Single-sample analysis ----

    def analyze_single(self, chunks: List[str], query: str, max_new_tokens: int = 256) -> Dict:
        context = self._build_rag_context(chunks, query)
        enc = self.tokenizer(context, return_tensors="pt", add_special_tokens=False).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_attentions=True,
                use_cache=True,
            )

        chunk_spans = self._compute_chunk_token_spans(chunks)
        step_attention = self._aggregate_step_chunk_attention(outputs.attentions, chunk_spans)
        sparsity_metrics = self._compute_sparsity_metrics(step_attention, len(chunks))
        return {
            "chunks": chunks,
            "query": query,
            "step_attention": {str(k): v.detach().cpu().numpy().tolist() for k, v in step_attention.items()},
            "sparsity_metrics": sparsity_metrics,  # per-step scalars
            "total_steps": len(step_attention),
        }

    def _compute_sparsity_metrics(self, step_attention: Dict[int, torch.Tensor], n_chunks: int) -> Dict[str, Dict[str, float]]:
        """
        For each step, compute:
          - entropy (in nats)
          - normalized_entropy = entropy / log(n_chunks)   (0..1, comparable across different n_chunks)
          - gini = 1 - sum p_i^2
          - top_25_percent_ratio (mass of top max(1, n_chunks//4) chunks)
        Returns {str(step): {...}} with vanilla Python floats.
        """
        out: Dict[str, Dict[str, float]] = {}
        if n_chunks == 0:
            return out

        for step, scores in step_attention.items():
            p = scores / (scores.sum() + 1e-8)
            entropy = float((-p * (p + 1e-8).log()).sum().item())
            gini = float((1.0 - (p * p).sum()).item())
            k = max(1, n_chunks // 4)
            top_k_ratio = float(torch.topk(p, k=k).values.sum().item())
            norm_entropy = float(entropy / (np.log(n_chunks) if n_chunks > 1 else 1.0))

            out[str(step)] = {
                "entropy": entropy,
                "normalized_entropy": norm_entropy,
                "gini_coefficient": gini,
                "top_25_percent_ratio": top_k_ratio,
                "max_attention": float(scores.max().item()),
                "min_attention": float(scores.min().item()),
                "attention_std": float(scores.std().item()),
            }
        return out

    # ---- Plots ----

    def save_single_heatmap(self, analysis_results: Dict, output_path: str) -> str:
        step_attention = analysis_results.get("step_attention", {})
        if not step_attention:
            return ""
        # assemble matrix [num_chunks x num_steps]
        steps = sorted(int(s) for s in step_attention.keys())
        mat = np.stack([np.array(step_attention[str(s)], dtype=float) for s in steps], axis=1)  # [C x T]

        plt.figure(figsize=(12, 8))
        if SEABORN_AVAILABLE:
            sns.heatmap(mat, cmap="viridis", cbar_kws={"label": "Attention Score"})
        else:
            plt.imshow(mat, cmap="viridis", aspect="auto")
            plt.colorbar(label="Attention Score")
        plt.title("Validation of H2a - Sparsity: Attention distribution is sparse at each step.")
        plt.xlabel("Steps/Tokens")
        plt.ylabel("RAG Chunk Index")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    def save_dataset_metric_curves(self, per_sample_metrics: List[Dict[str, Dict[str, float]]],
                                   output_dir: str) -> str:
        """
        Average per-step metrics across samples (for the common prefix of steps present in all samples)
        and plot curves vs step.
        """
        if not per_sample_metrics:
            return ""

        # Align by common number of steps
        per_lengths = [len(m) for m in per_sample_metrics]
        T = min(per_lengths)
        steps = list(range(T))

        # Collect arrays
        def metric_array(name: str) -> np.ndarray:
            arr = np.stack([
                np.array([per_sample_metrics[i][str(t)][name] for t in steps], dtype=float)
                for i in range(len(per_sample_metrics))
            ], axis=0)  # [N x T]
            return arr

        names = ["normalized_entropy", "gini_coefficient", "top_25_percent_ratio"]
        means = {}
        for nm in names:
            A = metric_array(nm)
            means[nm] = A.mean(axis=0)  # [T]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, means["normalized_entropy"], label="Normalized Entropy")
        plt.plot(steps, means["gini_coefficient"], label="Gini Coefficient")
        plt.plot(steps, means["top_25_percent_ratio"], label="Top-25% Mass")
        plt.xlabel("Decoding Step (t)")
        plt.ylabel("Average Metric (across queries)")
        plt.title("Average Sparsity Metrics vs Step (H2a) — Dataset")
        plt.grid(True, alpha=0.3)
        plt.legend()
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "sparsity_metrics_vs_step.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return out_path


# ---------------- Main ----------------

def main():
    p = argparse.ArgumentParser(description="Analyze RAG chunk attention sparsity (H2a)")
    p.add_argument("--model", required=True, help="HF model name/path")
    p.add_argument("--chunks-json", required=True, help="JSON file (single sample or dataset list)")
    p.add_argument("--query", default="", help="Override question for single-sample/raw-chunks cases")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="results/rag_sparsity")
    p.add_argument("--sample-limit", type=int, default=0, help="Limit number of dataset samples (0 = all)")
    p.add_argument("--dataset-save-heatmaps", action="store_true",
                   help="Also save per-sample heatmaps under out-dir/per_sample/")
    p.add_argument("--save-first-n-heatmaps", type=int, default=3,
               help="If >0, save heatmaps for the first N samples. "
                    "If --dataset-save-heatmaps is set, all samples are saved.")
    p.add_argument("--chunk-limit", type=int, default=0, help="Limit number of chunks per sample (0 = all chunks)")

    
    args = p.parse_args()

    is_dataset, samples = load_single_or_dataset(args.chunks_json, args.query)
    if args.sample_limit and is_dataset:
        samples = samples[:args.sample_limit]

    analyzer = RAGChunkSparsityAnalyzer(args.model, args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    if not is_dataset:
        # Single sample path: run once, save heatmap + JSON
        chunks, query = samples[0]
        # Limit chunks per sample if specified
        original_chunk_count = len(chunks)
        if args.chunk_limit > 0 and len(chunks) > args.chunk_limit:
            chunks = chunks[:args.chunk_limit]
            print(f"Loaded 1 sample | chunks={len(chunks)} (limited from {original_chunk_count})")
        else:
            print(f"Loaded 1 sample | chunks={len(chunks)}")
        res = analyzer.analyze_single(chunks, query, args.max_new_tokens)

        # Save JSON
        single_json = os.path.join(args.out_dir, "sparsity_analysis.json")
        with open(single_json, "w") as f:
            json.dump(res, f, indent=2)
        print("Saved:", single_json)

        # Heatmap
        heatmap = os.path.join(args.out_dir, "attention_sparsity_heatmap.png")
        analyzer.save_single_heatmap(res, heatmap)
        print("Saved:", heatmap)

        # Tiny summary
        sm = res["sparsity_metrics"]
        if sm:
            steps = sorted(int(s) for s in sm.keys())
            mean_gini = float(np.mean([sm[str(t)]["gini_coefficient"] for t in steps]))
            mean_top25 = float(np.mean([sm[str(t)]["top_25_percent_ratio"] for t in steps]))
            print(f"Mean Gini: {mean_gini:.3f} | Mean Top-25% mass: {mean_top25:.3f}")

    else:
        # Dataset path: iterate and average per-step metrics across queries
        print(f"Loaded dataset with {len(samples)} samples")
        per_sample_metrics: List[Dict[str, Dict[str, float]]] = []

        per_sample_dir = os.path.join(args.out_dir, "per_sample")
        if args.dataset_save_heatmaps or args.save_first_n_heatmaps > 0:
            os.makedirs(per_sample_dir, exist_ok=True)

        for idx, (chunks, query) in enumerate(samples, 1):
            # Limit chunks per sample if specified
            original_chunk_count = len(chunks)
            if args.chunk_limit > 0 and len(chunks) > args.chunk_limit:
                chunks = chunks[:args.chunk_limit]
                print(f"[{idx}/{len(samples)}] chunks={len(chunks)} (limited from {original_chunk_count})")
            else:
                print(f"[{idx}/{len(samples)}] chunks={len(chunks)}")
            res = analyzer.analyze_single(chunks, query, args.max_new_tokens)
            per_sample_metrics.append(res["sparsity_metrics"])

            # Optional: save per-sample heatmap for inspection
            if args.dataset_save_heatmaps or (args.save_first_n_heatmaps > 0 and idx <= args.save_first_n_heatmaps):
                heatmap = os.path.join(per_sample_dir, f"sample_{idx:05d}_heatmap.png")
                analyzer.save_single_heatmap(res, heatmap)
                print(f"Saved heatmap: {heatmap}")

        # Plot averaged curves
        plot_path = analyzer.save_dataset_metric_curves(per_sample_metrics, args.out_dir)
        if plot_path:
            print("Saved averaged curves:", plot_path)

        # Save a compact dataset summary
        #  - overall means across all steps (using common prefix length)
        T = min(len(m) for m in per_sample_metrics)
        steps = list(range(T))
        def avg_over(name: str) -> float:
            return float(np.mean([
                np.mean([per_sample_metrics[i][str(t)][name] for t in steps])
                for i in range(len(per_sample_metrics))
            ]))

        summary = {
            "num_samples": len(per_sample_metrics),
            "common_steps": T,
            "mean_normalized_entropy_overall": avg_over("normalized_entropy"),
            "mean_gini_overall": avg_over("gini_coefficient"),
            "mean_top25_overall": avg_over("top_25_percent_ratio"),
        }
        out_json = os.path.join(args.out_dir, "sparsity_dataset_summary.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print("Saved summary:", out_json)


if __name__ == "__main__":
    main()
