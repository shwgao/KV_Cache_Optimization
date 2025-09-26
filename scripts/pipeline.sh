#!/usr/bin/env bash

# set -e

# PYTHON=python3
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CONFIG="configs/config.yaml"           
# INPUT="inputs/musique_s.json"       
# OUTPUT="results/pipeline_results"   

mkdir -p "results/pipeline_v2_results/musique"

# "$PYTHON" "src/pipeline.py" \
#   --config "$CONFIG" \
#   --input "$INPUT" \
#   --output "$OUTPUT"

#!/usr/bin/env bash

python3 src/pipeline_v2.py \
    --input inputs/musique_s.json \
    --output_dir results/pipeline_v2_results/musique \
    --model_id mistralai/Mistral-7B-Instruct-v0.2 \
    --top_k 5 \
    --max_tokens 20 \
    --device cuda:0

