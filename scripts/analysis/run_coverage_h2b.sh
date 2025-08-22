#!/usr/bin/env bash
set -e

MODEL="meta-llama/Meta-Llama-3-8B"
CHUNKS_JSON="inputs/musique_s.json"
DEVICE="cuda:0"
MAX_NEW_TOKENS=256
ATTN_THRESHOLD=0.01
OUT_DIR="results/analysis/coverage"
SAMPLE_LIMIT=5
CHUNK_LIMIT=5

mkdir -p "$OUT_DIR"

python analysis/rag_chunk_coverage.py \
  --model "$MODEL" \
  --chunks-json "$CHUNKS_JSON" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --attention-threshold "$ATTN_THRESHOLD" \
  --device "$DEVICE" \
  --out-dir "$OUT_DIR" \
  --sample-limit "$SAMPLE_LIMIT" \
  --chunk-limit "$CHUNK_LIMIT"
