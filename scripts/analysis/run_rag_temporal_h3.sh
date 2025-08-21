#!/usr/bin/env bash
set -e

# Paths (edit if needed)
PYTHON_SCRIPT="../../analysis/rag_temporal_h3.py"
DATASET_JSON="../../inputs/musique_s.json"
OUT_DIR="../../analysis/results/rag_temporal"

# Model / run config
MODEL="mistralai/Mistral-7B-Instruct-v0.2"
DEVICE="cuda:0"
MAX_NEW_TOKENS=256
TOP_K_CHUNKS=20
INTERVALS="1,5,10,25,50"

python "$PYTHON_SCRIPT" \
  --dataset-json "$DATASET_JSON" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --top-k-chunks "$TOP_K_CHUNKS" \
  --intervals "$INTERVALS" \
  --out-dir "$OUT_DIR"
