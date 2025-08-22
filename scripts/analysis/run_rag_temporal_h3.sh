#!/usr/bin/env bash
set -e

# Memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paths (edit if needed)
PYTHON_SCRIPT="analysis/rag_temporal_h3.py"
DATASET_JSON="inputs/musique_s.json"
OUT_DIR="results/analysis/rag_temporal_h3"

# Model / run config
MODEL="meta-llama/Meta-Llama-3-8B"
DEVICE="cuda:0"
TOP_K_CHUNKS=2
ATTN_THRESHOLD=0.001
INTERVALS="1,5,10,25,50,75,100,150,200"

# Memory limits (adjust these to avoid CUDA errors)
MAX_CHUNKS=5
MAX_SAMPLES=5
MAX_NEW_TOKENS=128

python "$PYTHON_SCRIPT" \
  --dataset-json "$DATASET_JSON" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --top-k-chunks "$TOP_K_CHUNKS" \
  --attention-threshold "$ATTN_THRESHOLD" \
  --intervals "$INTERVALS" \
  --max-chunks "$MAX_CHUNKS" \
  --max-samples "$MAX_SAMPLES" \
  --out-dir "$OUT_DIR"
