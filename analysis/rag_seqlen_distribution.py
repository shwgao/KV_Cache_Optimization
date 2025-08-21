import argparse
import csv
import json
import os
from typing import List, Dict, Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def iter_texts_from_path(path: str, file_format: str, text_fields: List[str]) -> Iterable[str]:
    if file_format == "txt":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    elif file_format == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                parts = [str(obj.get(k, "")) for k in text_fields]
                joined = "\n".join([p for p in parts if p])
                if joined:
                    yield joined
    elif file_format == "csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parts = [str(row.get(k, "")) for k in text_fields]
                joined = "\n".join([p for p in parts if p])
                if joined:
                    yield joined
    else:
        raise ValueError(f"Unsupported format: {file_format}")


def auto_format(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return "jsonl"
    if lower.endswith(".csv"):
        return "csv"
    return "txt"


def main():
    parser = argparse.ArgumentParser(description="Compute and plot sequence length distribution for RAG corpora.")
    parser.add_argument("--inputs", nargs="+", default=["../../inputs/musique_s.json"], help="Paths to txt/jsonl/csv files")
    parser.add_argument("--text-fields", nargs="*", default=["query", "context", "answer"], help="Fields to join for jsonl/csv")
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B", help="HF tokenizer id or path")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--out-csv", default="baselines/CacheBlend/analysis/results/rag_seq_lengths.csv")
    parser.add_argument("--out-plot", default="baselines/CacheBlend/analysis/results/rag_seq_distribution.png")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lengths: List[int] = []
    for path in args.inputs:
        fmt = auto_format(path)
        for text in iter_texts_from_path(path, fmt, args.text_fields):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            lengths.append(len(ids))
            if len(lengths) >= args.max_samples:
                break
        if len(lengths) >= args.max_samples:
            break

    if not lengths:
        raise RuntimeError("No texts found to compute lengths.")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = pd.DataFrame({"seq_len": lengths})
    df.to_csv(args.out_csv, index=False)

    # Stats
    stats = {
        "count": len(lengths),
        "mean": float(np.mean(lengths)),
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(np.max(lengths)),
    }
    print("Stats:", stats)

    # Plot
    plt.figure(figsize=(7, 4))
    plt.hist(lengths, bins=50, alpha=0.7, color="#4e79a7")
    plt.xlabel("Sequence length (tokens)")
    plt.ylabel("Count")
    plt.title("RAG sequence length distribution")
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    plt.close()
    print("Saved:", args.out_csv)
    print("Saved:", args.out_plot)


if __name__ == "__main__":
    main()


