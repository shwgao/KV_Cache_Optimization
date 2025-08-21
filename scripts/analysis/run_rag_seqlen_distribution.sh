#!/usr/bin/env bash
set -euo pipefail

TOKENIZER=${TOKENIZER:-meta-llama/Meta-Llama-3-8B}
MAX_SAMPLES=${MAX_SAMPLES:-5000}
OUT_CSV=${OUT_CSV:-../../analysis/results/rag_seq_lengths.csv}
OUT_PLOT=${OUT_PLOT:-../../analysis/results/rag_seq_distribution.png}

INPUTS=${INPUTS:-../../inputs/musique_s.json}
TEXT_FIELDS=${TEXT_FIELDS:-query,context,answer}

python3 ../../analysis/rag_seqlen_distribution.py \
    --inputs "$INPUTS" \
    --text-fields $TEXT_FIELDS \
    --tokenizer "$TOKENIZER" \
    --max-samples "$MAX_SAMPLES" \
    --out-csv "$OUT_CSV" \
    --out-plot "$OUT_PLOT"


