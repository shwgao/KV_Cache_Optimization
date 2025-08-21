#!/usr/bin/env bash
set -euo pipefail

MODELS=${MODELS:-llama3-8b llama3-70b}
PRECISIONS=${PRECISIONS:-fp16 int8}
SEQ_LENS=${SEQ_LENS:-16384 32768 65536 131072 262144}
BATCH_SIZES=${BATCH_SIZES:-1}

python3 analysis/kv_cache_theory.py \
    --models $MODELS \
    --precisions $PRECISIONS \
    --seq-lens $SEQ_LENS \
    --batch-sizes $BATCH_SIZES \
    --out-plots-dir "results/analysis/kv_plots_theory" \
    --out-csv "results/analysis/kv_cache_theory.csv"



