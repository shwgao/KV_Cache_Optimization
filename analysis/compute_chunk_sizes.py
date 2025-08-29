import argparse
import json
from typing import List, Dict, Any

from transformers import AutoTokenizer


def build_doc_prompts(example: Dict[str, Any]) -> List[str]:
    doc_prompts = []
    for ctx in example.get("ctxs", []):
        title = ctx.get("title", "")
        text = ctx.get("text", "")
        doc_prompts.append(f"{title}\n\n{text}\n\n")
    return doc_prompts


def compute_chunk_lengths_for_sample(example: Dict[str, Any], tokenizer) -> List[int]:
    doc_prompts = build_doc_prompts(example)
    chunk_lengths: List[int] = []
    for doc in doc_prompts:
        ids = tokenizer.encode(doc, add_special_tokens=False)
        chunk_lengths.append(len(ids))
    return chunk_lengths


def compute_average_chunk_size(dataset: List[Dict[str, Any]], tokenizer_id: str) -> float:
    """Return the overall average chunk size (in tokens) across entire dataset."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_chunks = 0
    total_tokens = 0
    for example in dataset:
        lens = compute_chunk_lengths_for_sample(example, tokenizer)
        total_chunks += len(lens)
        total_tokens += sum(lens)
    return float(total_tokens) / total_chunks if total_chunks > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Compute tokenized chunk sizes for a dataset (e.g., musique_s.json)")
    parser.add_argument("--dataset", default="/nfs/hpc/share/jainc/SemCache/baselines/CacheBlend/inputs/musique_s.json")
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--limit", type=int, default=-1, help="Number of samples to process (-1 = all)")
    parser.add_argument("--index", type=int, default=None, help="Specific sample index to process (overrides --limit)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determine which samples to process
    if args.index is not None:
        indices = [args.index]
    else:
        num_print = len(data) if args.limit is None or args.limit < 0 else min(args.limit, len(data))
        indices = list(range(num_print))

    # Print selected samples' chunk lengths and per-sample averages
    for i in indices:
        lens = compute_chunk_lengths_for_sample(data[i], tokenizer)
        if len(lens) == 0:
            print(f"Sample {i}: no chunks")
        else:
            avg_len = sum(lens) / len(lens)
            print(f"Sample {i}: chunk_lens={lens} avg={avg_len:.1f}")

    # Compute overall average chunk size across entire dataset
    total_chunks = 0
    total_tokens = 0
    for i in range(len(data)):
        lens = compute_chunk_lengths_for_sample(data[i], tokenizer)
        total_chunks += len(lens)
        total_tokens += sum(lens)

    if total_chunks == 0:
        print("Overall average chunk size (dataset): 0 tokens (no chunks found)")
    else:
        overall_avg_all = total_tokens / total_chunks
        print(f"Overall average chunk size (entire dataset): {overall_avg_all:.1f} tokens across {total_chunks} chunks and {len(data)} samples")

    # Note: empirical measurement belongs in token_budget_calculator.py


if __name__ == "__main__":
    main()


