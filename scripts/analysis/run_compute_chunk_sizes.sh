#!/usr/bin/env bash
set -euo pipefail

DATASET="inputs/musique_s.json"
TOKENIZER="meta-llama/Meta-Llama-3-8B"
LIMIT="5"
OUT_CSV="results/analysis/chunk_sizes.csv"

python analysis/compute_chunk_sizes.py \
  --dataset "${DATASET}" \
  --tokenizer "${TOKENIZER}" \
  --limit "${LIMIT}" \


