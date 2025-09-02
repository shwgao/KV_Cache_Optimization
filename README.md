## Problem Statement
Large language models (LLMs) incur substantial latency and memory overhead when conditioning on many retrieved documents. Traditional RAG pipelines either truncate context or re-encode long contexts repeatedly. CacheBlend addresses this by:
- Precomputing and storing per-chunk KV caches using an LLM.
- Keeping only the top-K most relevant chunks on GPU while staging remaining chunks on CPU as placeholders.
- During speculative decoding, proactively promoting likely-needed CPU chunks to GPU using a lightweight relevance proxy, enabling fast, memory-aware decoding.

This approach aims to balance quality (by retaining more context across many candidates) and performance (by efficiently managing GPU memory and minimizing redundant computation).

## Repository Structure
```
baselines/CacheBlend/
  analysis/                # Analysis utilities (optional)
  colbert_index/           # ColBERT/RAGatouille indices (generated)
  configs/                 # Optional configuration files
  data/                    # Optional sample data
  example/                 # Example usage artifacts
  inputs/                  # Provided datasets (e.g., MuSiQue, Samsum)
  results/
    analysis/              # Analysis outputs
    decoding/              # Speculative decode traces
    retrieval/             # Retrieval outputs (top-k indices)
    kv_caches/             # Saved KV cache entries (per-chunk folders)
  scripts/
    analysis/
    retrieval/
      run_retrieval.sh     # Run RAG indexing + retrieval
    build_kv_cache.sh      # Build per-sample KV caches from retrieval JSON
    decoding/
      run_speculative_decode.sh  # Run speculative decode with promotions
    run_token_budget_test.sh
  src/
    build_kv_cache.py      # Build GPU/CPU KV caches from retrieval output
    config.py              # Pipeline configuration utilities
    kv_cache_manager.py    # CPU/GPU cache manager with CacheBlend kernels
    rag_retrieval.py       # RAGatouille (ColBERT) indexing + retrieval
    run_pipeline.py        # End-to-end demo pipeline (optional)
    scheduler.py           # Scheduling logic (if used)
    speculative_decode.py  # Speculative decode with proactive promotions
    token_budget_calculator.py
  utils/
  vllm_blend/
  requirements.txt
  README.md
```

## Environment Setup
- Python requirements (minimal): see `baselines/CacheBlend/requirements.txt`.
- Additional dependencies:
  - Retrieval uses RAGatouille/ColBERT: `ragatouille` and its dependencies.
  - Transformers for model loading: `transformers`.
  - tqdm for progress bars.

Example installation:
```bash
cd CacheBlend/vllm_blend
pip install -e .
cd ..
pip install -r requirements.txt
```

Ensure you have access to the target HF model (eg, `meta-llama/Meta-Llama-3-8B`) and appropriate GPU/CPU memory.

## Datasets
Sample inputs are provided under `baselines/CacheBlend/inputs/`, eg:
- `musique_s.json`
- `wikimqa_s.json`
- `samsum.json`

## End-to-End Workflow
There are three main steps:

1) Retrieval (build RAG index and compute top-k per sample)
- Script: `scripts/retrieval/run_retrieval.sh`
- Writes: `results/retrieval/<dataset>_rag_both_k<k>.json`

```bash
cd baselines/CacheBlend
bash scripts/retrieval/run_retrieval.sh
# Default dataset: inputs/musique_s.json
# Output: results/retrieval/musique_s_rag_both_k5.json
```

2) Build KV Caches (prefill top-K on GPU, placeholders on CPU)
- Script: `scripts/build_kv_cache.sh`
- Reads: retrieval JSON from step 1
- Writes:
  - Summary: `results/kv_caches/musique_s_kv_top5.json` (default)
  - Per-chunk KV folders under `results/kv_caches/` when `--save-cache-dir` is enabled (default in script)

```bash
cd baselines/CacheBlend
bash scripts/build_kv_cache.sh
# Output summary: results/kv_caches/musique_s_kv_top5.json
# Saved chunk KV: results/kv_caches/<sample_chunk_id>/{keys.pt,values.pt,valid_mask.pt,metadata.json}
```

3) Speculative Decode
- Script: `scripts/decoding/run_speculative_decode.sh`
- Reads:
  - Retrieval JSON (from step 1)
  - Optionally loads cached KV from `results/kv_caches/` via `--load-cache-dir`
- Writes: `results/decoding/speculative_trace.json`

```bash
cd baselines/CacheBlend
bash scripts/decoding/run_speculative_decode.sh
# Output: results/decoding/speculative_trace.json
```

## Where to Find Results
- Retrieval outputs: `baselines/CacheBlend/results/retrieval/`
  - e.g `musique_s_rag_both_k5.json`
- KV cache summary and per-chunk KV: `baselines/CacheBlend/results/kv_caches/`
  - e.g `musique_s_kv_top5.json`, plus per-chunk folders
- Speculative decode trace and answers: `baselines/CacheBlend/results/decoding/`
  - e.g `speculative_trace.json`


## Notes
- Default values in the provided shell scripts can be overridden via environment variables.
- Ensure sufficient GPU/CPU memory for the chosen model and `top-k`.
- If CacheBlend kernels are required (`require_kernels=True` in `KVCacheManager`), ensure `vllm_blend` is importable.

## Quickstart
```bash
cd baselines/CacheBlend
# 1) Retrieval
bash scripts/retrieval/run_retrieval.sh
# 2) Build KV caches
bash scripts/build_kv_cache.sh
# 3) Speculative decode
bash scripts/decoding/run_speculative_decode.sh
```
