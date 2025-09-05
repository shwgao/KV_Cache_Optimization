#!/usr/bin/env bash
set -euo pipefail

# --------- user config ----------
MODEL="meta-llama/Meta-Llama-3-8B"
DEVICE="cuda:0"
DTYPE="auto"

PREDICT_JSON="results/decoding/speculative_next.json"
RETRIEVAL_JSON="results/retrieval/musique_s_rag_both_k5.json"
OUT_JSON="results/scheduler/scheduler_trace.json"

# NEW: precomputed KV dir (each subdir is a chunk_id, e.g., sample_chunk0)
LOAD_CACHE_DIR="results/kv_caches"
CACHE_FILTER_PREFIX="auto"   # auto => f"{sample_id}_chunk"
LOAD_INITIAL_TO_GPU="false"  # normally keep CPU_READY; set to "true" if you want direct GPU load

MAX_GPU=5
STEP_MS=50
SAFETY_MS=30
MAX_SAMPLES=1
# --------------------------------

mkdir -p "$(dirname "$OUT_JSON")"

python3 src/scheduler.py \
  --predict-json "$PREDICT_JSON" \
  --retrieval-json "$RETRIEVAL_JSON" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --max-gpu "$MAX_GPU" \
  --step-duration-ms "$STEP_MS" \
  --safety-margin-ms "$SAFETY_MS" \
  --max-samples "$MAX_SAMPLES" \
  --load-cache-dir "$LOAD_CACHE_DIR" \
  --cache-filter-prefix "$CACHE_FILTER_PREFIX" \
  ${LOAD_INITIAL_TO_GPU:+--load-initial-to-gpu} \
  --out "$OUT_JSON"
