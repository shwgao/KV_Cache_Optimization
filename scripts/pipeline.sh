#!/usr/bin/env bash

set -e

PYTHON=python3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="configs/config.yaml"           
INPUT="inputs/musique_s.json"       
OUTPUT="results/pipeline_results"   

mkdir -p "$OUTPUT"

"$PYTHON" "src/pipeline.py" \
  --config "$CONFIG" \
  --input "$INPUT" \
  --output "$OUTPUT"
