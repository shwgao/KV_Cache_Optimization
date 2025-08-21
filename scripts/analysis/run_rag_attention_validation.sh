#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-meta-llama/Meta-Llama-3-8B}
DEVICE=${DEVICE:-cuda:0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
ATTENTION_THRESHOLD=${ATTENTION_THRESHOLD:-0.1}
OUT_DIR=${OUT_DIR:-../../analysis/results/rag_attention_validation}

CHUNKS_JSON=${CHUNKS_JSON:-../../inputs/musique_s.json}
CHUNKS_DIR=${CHUNKS_DIR:-}
QUERY=${QUERY:-"Please provide a comprehensive summary of the given documents."}

if [[ -z "$CHUNKS_JSON" && -z "$CHUNKS_DIR" ]]; then
    echo "Error: Must provide either CHUNKS_JSON or CHUNKS_DIR"
    exit 1
fi

if [[ -n "$CHUNKS_JSON" ]]; then
    CHUNKS_ARG="--chunks-json $CHUNKS_JSON"
else
    CHUNKS_ARG="--chunks-dir $CHUNKS_DIR"
fi

python3 ../../analysis/rag_chunk_coverage.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --attention-threshold "$ATTENTION_THRESHOLD" \
    --out-dir "$OUT_DIR" \
    --query "$QUERY" \
    $CHUNKS_ARG

python3 ../../analysis/rag_chunk_sparsity.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --attention-threshold "$ATTENTION_THRESHOLD" \
    --out-dir "$OUT_DIR" \
    --query "$QUERY" \
    $CHUNKS_ARG

python3 ../../analysis/rag_chunk_temporal_analysis.py \
    --input-json "$OUT_DIR/sparsity_analysis.json" \
    --output-dir "$OUT_DIR/temporal" \
    --top-k-chunks 5 \
    --intervals "1,5,10,25,50,100"
