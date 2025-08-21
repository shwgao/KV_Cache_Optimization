#!/usr/bin/env bash
set -e

MODEL="meta-llama/Meta-Llama-3-8B"
CHUNKS_JSON="inputs/musique_s.json"
DEVICE="cuda:0"
MAX_NEW_TOKENS=256  # Reduced to save memory
OUT_DIR="results/analysis/sparsity"

mkdir -p "$OUT_DIR"

# Clear CUDA cache before running
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

python analysis/rag_chunk_sparsity.py \
  --model "$MODEL" \
  --chunks-json "$CHUNKS_JSON" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --save-first-n-heatmaps 3 \
  --device "$DEVICE" \
  --out-dir "$OUT_DIR" \
  --sample-limit 5  \
  --chunk-limit 5
