#!/usr/bin/env bash
#
# Shell wrapper to run the full KV reuse baseline.  This script mirrors
# ``scripts/pipeline.sh`` but invokes ``full_kv_reuse.py`` instead.  It
# automatically creates an output directory and passes through the
# configuration and input dataset paths.

set -e

PYTHON=python3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="configs/config.yaml"
INPUT="inputs/musique_s.json"
OUTPUT="results/full_kv_reuse_results"

mkdir -p "$OUTPUT"

"$PYTHON" "src/full_kv_reuse_v1.py" \
  --config "$CONFIG" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --retrieval_json retrieval/retrieval_topk.json \
  --top_k 5
