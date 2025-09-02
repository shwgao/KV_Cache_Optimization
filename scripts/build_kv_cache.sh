#!/usr/bin/env bash
set -e
EFFECTIVE_TOPK=${TOP_K:-5}
DEFAULT_OUT_PATH="results/kv_caches/musique_s_kv_top${EFFECTIVE_TOPK}.json"
OUT_PATH=${OUT_PATH:-$DEFAULT_OUT_PATH}
OUT_DIR=$(dirname "$OUT_PATH")
mkdir -p "$OUT_DIR"

ARGS=(
  --retrieval-json "${RETRIEVAL_JSON:-results/retrieval/musique_s_rag_both_k5.json}"
  --model "${MODEL:-meta-llama/Meta-Llama-3-8B}"
  --device "${DEVICE:-cuda:0}"
  --top-k "${TOP_K:-5}"
  --max-samples "${MAX_SAMPLES:-1}"
  --out_path "$OUT_PATH"
  --max-gpu-chunks "${MAX_GPU_CHUNKS:-5}"
  --max-cpu-chunks "${MAX_CPU_CHUNKS:-5}"
  --gpu-mem-gb "${GPU_MEM_GB:-100}"
  --cpu-mem-gb "${CPU_MEM_GB:-100}"
  --save-cache-dir "${SAVE_CACHE_DIR:-results/kv_caches}"
)

# Boolean flag: only append if enabled (no value expected by argparse)
if [[ "${DUMP_PLACEMENTS:-1}" != "0" ]]; then
  ARGS+=( --dump-placements )
fi

# Boolean flag for saving placeholders (default ON): set SAVE_PLACEHOLDERS=0 to disable
if [[ "${SAVE_PLACEHOLDERS:-1}" != "0" ]]; then
  ARGS+=( --save-placeholders )
fi

python src/build_kv_cache.py "${ARGS[@]}"


