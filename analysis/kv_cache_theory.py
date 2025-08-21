import argparse
import itertools
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(__file__))
from configs import MODEL_CONFIGS, PRECISION_BYTES, GPU_CAPACITIES, bytes_to_gib


def compute_kv_bytes(num_layers: int, hidden_size: int, seq_len: int,
                     precision_bytes: int, batch_size: int) -> int:
    """KV size in bytes: 2 (K and V) × L × S × d_model × bytes/elt × B"""
    return int(2 * num_layers * seq_len * hidden_size * precision_bytes * batch_size)


def compute_max_seq_len_per_gpu(num_layers: int, hidden_size: int,
                                precision_bytes: int, batch_size: int) -> Dict[str, int]:
    """Max sequence length that fits per GPU (theoretical KV only)."""
    denom = 2 * num_layers * hidden_size * precision_bytes * batch_size
    results: Dict[str, int] = {}
    for gpu_name, vram_bytes in GPU_CAPACITIES.items():
        max_seq_len = int(vram_bytes // denom)
        # Column names are normalized to be easy to read/sort
        results[f"max_seq_len_{gpu_name.replace('-', '_')}"] = max_seq_len
    return results


def build_grid(models: List[str], seq_lens: List[int],
               batch_sizes: List[int], precisions: List[str]) -> pd.DataFrame:
    rows = []
    for model_key, seq_len, batch_size, precision in itertools.product(models, seq_lens, batch_sizes, precisions):
        spec = MODEL_CONFIGS[model_key]
        pbytes = PRECISION_BYTES[precision]
        kv_bytes = compute_kv_bytes(spec.num_layers, spec.hidden_size, seq_len, pbytes, batch_size)
        kv_gib = bytes_to_gib(kv_bytes)

        # Per‑GPU max seq lens + fit flags
        max_per_gpu = compute_max_seq_len_per_gpu(spec.num_layers, spec.hidden_size, pbytes, batch_size)

        row = {
            "model": spec.name,
            "layers": spec.num_layers,
            "d_model": spec.hidden_size,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "precision": precision,
            "kv_bytes": kv_bytes,
            "kv_gib": kv_gib,
        }
        row.update(max_per_gpu)

        # Add boolean "fits_<gpu>" flags
        for gpu_key in max_per_gpu.keys():
            row[f"fits_{gpu_key[len('max_seq_len_'):]}"] = seq_len <= max_per_gpu[gpu_key]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure stable/pretty column order
    base_cols = ["model", "layers", "d_model", "seq_len", "batch_size", "precision", "kv_bytes", "kv_gib"]
    max_cols = sorted([c for c in df.columns if c.startswith("max_seq_len_")])
    fit_cols = sorted([c for c in df.columns if c.startswith("fits_")])
    return df[base_cols + max_cols + fit_cols]


def save_csv(df: pd.DataFrame, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    # Keep precision in the file
    df.to_csv(out_csv, index=False, float_format="%.4f")
    return out_csv

def plot_vs_gpus(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    capacities_gib = {k: bytes_to_gib(v) for k, v in GPU_CAPACITIES.items()}

    # Plot per (model, precision, batch_size)
    grouped = df.groupby(["model", "precision", "batch_size"])  # type: ignore
    plots = []
    for (model, precision, batch_size), gdf in grouped:
        gdf = gdf.sort_values("seq_len")
        plt.figure(figsize=(7, 4))
        plt.plot(gdf["seq_len"], gdf["kv_gib"], label=f"KV ({precision}, B={batch_size})")

        # Draw each GPU's VRAM as a dashed line
        for gpu, cap_gib in capacities_gib.items():
            plt.hlines(cap_gib, xmin=gdf["seq_len"].min(), xmax=gdf["seq_len"].max(),
                       linestyles="dashed", label=gpu)

        plt.xlabel("Sequence length (tokens)")
        plt.ylabel("KV cache size (GiB)")
        plt.title(f"KV cache vs GPU VRAM: {model}")
        # Deduplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        plt.legend(dedup.values(), dedup.keys(), loc="upper left", fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"kv_vs_gpus__{model}__{precision}__b{batch_size}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        plots.append(out_path)
    return plots


def main():
    parser = argparse.ArgumentParser(description="Compute theoretical KV cache sizes and compare to GPU VRAM")
    parser.add_argument("--models", nargs="+", default=["llama3-8b", "llama3-70b"],
                        help="Model keys from configs.MODEL_CONFIGS")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[16000, 32000, 64000, 96000, 128000])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--precisions", nargs="+", default=["fp16", "int8"], choices=list(PRECISION_BYTES.keys()))
    parser.add_argument("--out-csv", default="baselines/CacheBlend/analysis/results/kv_cache_theory.csv")
    parser.add_argument("--out-plots-dir", default="baselines/CacheBlend/analysis/results/kv_plots_theory")
    args = parser.parse_args()

    df = build_grid(args.models, args.seq_lens, args.batch_sizes, args.precisions)
    save_csv(df, args.out_csv)
    plot_paths = plot_vs_gpus(df, args.out_plots_dir)

    print("Saved:", args.out_csv)
    print("Plots:")
    for p in plot_paths:
        print(" -", p)
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
