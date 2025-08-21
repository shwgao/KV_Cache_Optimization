#!/usr/bin/env bash
set -euo pipefail

# -------- What you usually change --------
MODELS=("meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-70B")
PRECISIONS=("fp16")
DEVICE="cuda:0"
SEQ_LENS=("16384" "32768" "65536" "131072" "262144")
BATCH_SIZES=("1")                            

SCRIPT="analysis/kv_cache_benchmark.py"
DATA_DIR="inputs/musique_s.json"

# Run benchmark for each model and precision combination
for model in "${MODELS[@]}"; do
    for precision in "${PRECISIONS[@]}"; do
        echo "Running benchmark for model: $model, precision: $precision"
        
        OUT_CSV="results/analysis/kv_cache_benchmark_${model//\//_}_${precision}.csv"
        
        python "$SCRIPT" \
          --model "$model" \
          --precision "$precision" \
          --device "$DEVICE" \
          --data "$DATA_DIR" \
          --seq-lens "${SEQ_LENS[@]}" \
          --batch-sizes "${BATCH_SIZES[@]}" \
          --out-csv "$OUT_CSV"
        
        echo "Results saved to: $OUT_CSV"
    done
done

echo "All benchmarks completed!"
