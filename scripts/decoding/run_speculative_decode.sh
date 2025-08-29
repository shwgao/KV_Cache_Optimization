#!/usr/bin/env bash
set -e

OUT=${OUT:-results/decoding/speculative_trace.json}
mkdir -p "$(dirname "$OUT")"

ARGS=(
  --retrieval-json "${RETRIEVAL_JSON:-results/retrieval/musique_s_rag_both_k5.json}"
  --model "${MODEL:-meta-llama/Meta-Llama-3-8B}"
  --device "${DEVICE:-cuda:0}"
  --top-k "${TOP_K:-5}"
  --steps "${STEPS:-16}"
  --promote-per-step "${PROMOTE_PER_STEP:-2}"
  --max-samples "${MAX_SAMPLES:-150}"
  --gpu-mem-gb "${GPU_MEM_GB:-80}"
  --cpu-mem-gb "${CPU_MEM_GB:-128}"
  --max-gpu-chunks "${MAX_GPU_CHUNKS:-0}"
  --max-cpu-chunks "${MAX_CPU_CHUNKS:-1000}"
  --out "$OUT"
  --gen-max-new-tokens "${GEN_MAX_NEW_TOKENS:-32}"
)

# Always load KV cache from this directory (can override via env)
LOAD_CACHE_DIR=${LOAD_CACHE_DIR:-results/kv_caches}
ARGS+=( --load-cache-dir "$LOAD_CACHE_DIR" )

# Optional async prefetch controls
if [[ "${ASYNC_PREFETCH:-0}" != "0" ]]; then
  ARGS+=( --async-prefetch )
fi
if [[ -n "${PREFETCH_INTERVAL_MS:-}" ]]; then
  ARGS+=( --prefetch-interval-ms "${PREFETCH_INTERVAL_MS}" )
fi
if [[ -n "${PREFETCH_BATCH_SIZE:-}" ]]; then
  ARGS+=( --prefetch-batch-size "${PREFETCH_BATCH_SIZE}" )
fi

python src/speculative_decode.py "${ARGS[@]}"
