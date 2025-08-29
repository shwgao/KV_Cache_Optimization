#!/usr/bin/env bash
set -e

python src/token_budget_calculator.py \
  --model-config configs/llama3_8b.json \
  --dataset inputs/musique_s.json \
  --model meta-llama/Meta-Llama-3-8B \
  --precision bf16 \
  --device cuda:0 \
  --max-samples 150 \
  --probe-new-tokens 16 \
  --probe-output-attentions \
  --probe-attn-impl eager
