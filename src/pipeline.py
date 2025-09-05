#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import logging
from typing import Any, Dict, List, Optional

import yaml
import torch
from threading import Thread
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# --- local modules (v1 variants) ---
from rag_retrieval import RetrievalConfig, ColbertRetrieval
from build_kv_cache import KVCachesBuilder
from speculative_decoding import SpeculativeChunkPredictor
from scheduler import TangoScheduler


# ---------------- utilities ----------------

def _read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def _load_samples(path: str) -> List[Dict[str, Any]]:
    data = _read_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return [data]

def _abs_path(base_dir: str, maybe_path: str) -> str:
    if not maybe_path:
        return ""
    return maybe_path if os.path.isabs(maybe_path) else os.path.join(base_dir, maybe_path)

def _setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("pipeline_v1")

def _load_existing_retrieval(retrieval_json_path: str, logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """If retrieval json exists and is parseable -> return samples; else None."""
    if not retrieval_json_path or not os.path.exists(retrieval_json_path):
        return None
    try:
        data = _read_json(retrieval_json_path)
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and "results" in data:
            samples = data["results"]
        else:
            samples = [data]
        if not isinstance(samples, list) or not samples:
            return None
        logger.info(f"[Retrieval] Found existing retrieval file. Skipping retrieval: {retrieval_json_path}")
        logger.info(f"[Retrieval] Loaded {len(samples)} samples from existing retrieval JSON")
        return samples
    except Exception as e:
        logger.warning(f"[Retrieval] Failed to read existing retrieval JSON ({retrieval_json_path}): {e}")
        return None

def _build_prompt(sample: Dict[str, Any]) -> str:
    parts = []
    ctxs = sample.get("ctxs") or []
    for i, ch in enumerate(ctxs[:3]):
        title = (ch.get("title") or "").strip()
        text  = (ch.get("text") or "").strip()
        combined = f"{title}\n{text}".strip() if title else text
        if combined:
            parts.append(f"Document {i+1}: {combined}")
    parts.append(f"\nQuestion: {sample.get('question','')}")
    parts.append("\nAnswer:")
    return "\n\n".join(p for p in parts if p.strip())

def _restructure_kv_cache_dir_to_numeric(sample_cache_dir: str, expected_sample_id: str, logger: logging.Logger):
    """
    Rename each per-chunk subdir from '<chunk_id>' to its numeric '{idx}' under sample_cache_dir.
    Relies on 'metadata.json' having 'chunk_id' like '<sample_id>_chunk<idx>'.
    """
    if not sample_cache_dir or not os.path.isdir(sample_cache_dir):
        return
    for name in os.listdir(sample_cache_dir):
        src = os.path.join(sample_cache_dir, name)
        if not os.path.isdir(src):
            continue

        meta_path = os.path.join(src, "metadata.json")
        if not os.path.isfile(meta_path):
            # Try to parse idx from directory name in case metadata is missing
            # Name might already be numeric; skip if so
            if name.isdigit():
                continue
            if "_chunk" in name:
                try:
                    idx = int(name.split("_chunk")[-1])
                    dst = os.path.join(sample_cache_dir, str(idx))
                    if src != dst and not os.path.exists(dst):
                        os.rename(src, dst)
                    continue
                except Exception:
                    pass
            # otherwise leave as is
            continue

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            chunk_id = str(meta.get("chunk_id", ""))
            # Expect "<sample_id>_chunk<idx>"
            if "_chunk" not in chunk_id:
                continue
            sample_id_part, idx_part = chunk_id.rsplit("_chunk", 1)
            if expected_sample_id and sample_id_part != expected_sample_id:
                # different sample's chunk folder somehow landed here; skip
                continue
            idx = int(idx_part)
            dst = os.path.join(sample_cache_dir, str(idx))
            if src != dst and not os.path.exists(dst):
                os.rename(src, dst)
        except Exception as e:
            logger.warning(f"[KV] Could not restructure '{src}' -> numeric: {e}")

# ---------------- decoding helpers (reuse model across samples) ----------------

class FinalDecoder:
    def __init__(self, model_id: str, device: str):
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.mdl = AutoModelForCausalLM.from_pretrained(model_id).to(self.device).eval()

    def decode(self, sample: Dict[str, Any], max_new_tokens: int = 128) -> Dict[str, Any]:
        prompt = _build_prompt(sample)
        inputs = self.tok([prompt], return_tensors="pt", padding=False, truncation=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            streamer=streamer,
            return_dict_in_generate=True,
        )

        first_tok_t = None
        pieces: List[str] = []

        def consume():
            nonlocal first_tok_t, pieces
            for chunk in streamer:
                if first_tok_t is None:
                    first_tok_t = time.time()
                pieces.append(chunk)

        t0 = time.time()
        t = Thread(target=lambda: self.mdl.generate(**gen_kwargs))
        t.start()
        consume()
        t.join()
        t1 = time.time()

        ttft_s = (first_tok_t - t0) if first_tok_t is not None else None
        gen_time_s = t1 - (first_tok_t or t0)
        tps = (len(pieces) or 1) / max(1e-9, gen_time_s)

        return {
            "answer": "".join(pieces).strip(),
            "ttft_s": ttft_s,
            "tokens_per_sec": tps,
            "decoded_tokens": len(pieces),
        }


# ---------------- main pipeline ----------------

def main():
    ap = argparse.ArgumentParser("CacheBlend Pipeline v1 (sample-by-sample, streaming outputs)")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--input", required=True, help="Path to input dataset JSON")
    ap.add_argument("--output", default="pipeline_results", help="Output directory")
    ap.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    logger = _setup_logging(args.log_level)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # --- basics ---
    model_id = cfg["model"]["model_name"]
    device = cfg["model"].get("device", "cuda:0")
    logger.info(f"[Config] Model: {model_id} | Device: {device}")
    logger.info(f"[Paths] Output root: {args.output}")

    # Load dataset
    logger.info(f"[Input] Loading dataset: {args.input}")
    samples = _load_samples(args.input)
    logger.info(f"[Input] Loaded {len(samples)} raw samples")

    # Output paths (single file per task)
    retrieval_json_path = _abs_path(args.output, cfg["paths"]["retrieval_json"])
    kv_summary_path     = _abs_path(args.output, cfg["paths"]["kv_summary_json"])
    pred_out_path       = _abs_path(args.output, cfg["speculative"].get("out_path", "results/prediction/speculative_next.json"))
    sched_out_path      = _abs_path(args.output, cfg["scheduler"].get("out_path", "results/scheduler/tango_trace.json"))
    final_results_path  = os.path.join(args.output, "final_results.json")
    summary_path        = os.path.join(args.output, "summary.json")

    # Try to skip retrieval if an aggregate retrieval file already exists
    maybe_existing = _load_existing_retrieval(retrieval_json_path, logger)
    if maybe_existing is not None:
        samples = maybe_existing

    # Prepare objects to reuse
    retriever = None
    if maybe_existing is None:
        rconf = RetrievalConfig(
            model_id         = cfg["retrieval"]["model_id"],
            dataset_name     = cfg["retrieval"]["dataset_name"],
            r_text_index_key = cfg["retrieval"]["r_text_index_key"],
            doc_key          = cfg["retrieval"]["doc_key"],
            question_key     = cfg["retrieval"]["question_key"],
            retrieved_key    = cfg["retrieval"]["retrieved_key"],
            page_id_key      = cfg["retrieval"]["page_id_key"],
            top_k            = int(cfg["retrieval"]["top_k"]),
        )
        retriever = ColbertRetrieval(rconf)

    kv_builder = KVCachesBuilder()
    predictor  = SpeculativeChunkPredictor()
    scheduler  = TangoScheduler()
    decoder    = FinalDecoder(model_id=model_id, device=device)

    # Aggregates (we will FLUSH after each sample)
    aggregated_kv: List[Dict[str, Any]] = []
    pred_trace_all: List[Dict[str, Any]] = []
    pred_results_all: List[Dict[str, Any]] = []
    sched_trace_all: List[Dict[str, Any]] = []
    final_answers: List[Dict[str, Any]] = []

    # KV root per requested structure
    kv_root = _abs_path(args.output, cfg["kv_builder"].get("save_cache_dir", "results/pipeline/kv_cache"))

    logger.info("[Pipeline] Running stages sample-by-sample with streaming writes")
    for si in tqdm(range(len(samples)), desc="Samples", unit="sample"):
        sample = samples[si]
        sample_id = str(sample.get("id", f"sample{si}"))

        # 1) Retrieval (only if not preloaded) -> FLUSH retrieval.json NOW
        if maybe_existing is None:
            enriched = retriever.prepare([sample])
            enriched = retriever.retrieve(enriched)
            sample = enriched[0]
            samples[si] = sample
            _write_json(retrieval_json_path, samples)  # <— flush

        # 2) KV Cache Build (per-sample) -> save under kv_root/sample{si+1}/..., then FLUSH kv_summary.json
        sample_dir = os.path.join(kv_root, f"sample{si+1}")
        os.makedirs(sample_dir, exist_ok=True)

        kv_payload = kv_builder.build(
            samples=[sample],
            model_id=model_id,
            device=device,
            top_k=int(cfg["cache"]["max_gpu_chunks"]),
            max_samples=1,
            max_gpu_chunks=int(cfg["cache"]["max_gpu_chunks"]),
            max_cpu_chunks=int(cfg["cache"]["max_cpu_chunks"]),
            gpu_mem_gb=float(cfg["cache"]["gpu_memory_limit_gb"]),
            cpu_mem_gb=float(cfg["cache"]["cpu_memory_limit_gb"]),
            dump_placements=bool(cfg["kv_builder"].get("dump_placements", False)),
            save_cache_dir=sample_dir,
            save_placeholders=bool(cfg["kv_builder"].get("save_placeholders", False)),
            retrieval_json_path=retrieval_json_path if os.path.exists(retrieval_json_path) else "",
        )
        _restructure_kv_cache_dir_to_numeric(sample_dir, expected_sample_id=str(sample_id), logger=logger)
        aggregated_kv.append({"sample_index": si, **kv_payload})
        _write_json(kv_summary_path, {"per_sample": aggregated_kv, "kv_root": kv_root})  # <— flush

        # 3) Speculative Prediction (per-sample) -> FLUSH speculative.json
        pred_payload = predictor.predict(
            samples=[sample],
            model_id=model_id,
            device=device,
            top_k=int(cfg["speculative"]["top_k"]),
            steps=int(cfg["speculative"]["steps"]),
            promote_per_step=int(cfg["speculative"]["promote_per_step"]),
            max_gpu=int(cfg["speculative"]["max_gpu"]),
            max_samples=1,
            enable_progress=False,
            out_path=None,
        )
        for row in pred_payload.get("trace", []):
            row["sample_index"] = si
        for row in pred_payload.get("results", []):
            row["sample_index"] = si
        pred_trace_all.extend(pred_payload.get("trace", []))
        pred_results_all.extend(pred_payload.get("results", []))
        _write_json(pred_out_path, {"trace": pred_trace_all, "results": pred_results_all})  # <— flush

        # 4) Scheduler (per-sample) -> FLUSH scheduler.json
        sched_payload = scheduler.run(
            pred_payload=pred_payload,
            retr_samples=[sample],
            model_id=model_id,
            device=device,
            dtype=str(cfg["model"].get("dtype", "auto")),
            max_gpu=int(cfg["scheduler"]["max_gpu"]),
            step_duration_ms=int(cfg["scheduler"]["step_duration_ms"]),
            safety_margin_ms=int(cfg["scheduler"]["safety_margin_ms"]),
            max_samples=1,
            load_cache_dir=sample_dir,
            cache_filter_prefix="",    # numeric subdirs
            load_initial_to_gpu=bool(cfg["scheduler"].get("load_initial_to_gpu", False)),
            out_path=None,
            enable_progress=False,
        )
        for row in sched_payload.get("trace", []):
            row["sample_index"] = si
        sched_trace_all.extend(sched_payload.get("trace", []))
        _write_json(sched_out_path, {"trace": sched_trace_all})  # <— flush

        # 5) Final Decoding (per-sample) -> FLUSH final_results.json + rolling summary
        decode_metrics = decoder.decode(sample, max_new_tokens=128)
        final_answers.append({
            "sample_index": si,
            "question": sample.get("question", ""),
            **decode_metrics,
        })
        ttfts = [x["ttft_s"] for x in final_answers if x.get("ttft_s") is not None]
        avg_ttft = (sum(ttfts) / len(ttfts)) if ttfts else None
        _write_json(final_results_path, {"per_sample": final_answers, "avg_ttft_s": avg_ttft, "count": len(final_answers)})  # <— flush

        # Rolling summary while running (in case of early stop)
        _write_json(summary_path, {
            "status": "running",
            "processed": si + 1,
            "total": len(samples),
            "model": model_id,
            "device": device,
            "outputs": {
                "retrieval_json": retrieval_json_path,
                "kv_summary_json": kv_summary_path,
                "speculative_json": pred_out_path,
                "scheduler_json": sched_out_path,
                "final_results_json": final_results_path,
                "kv_root": kv_root,
            },
            "metrics": {"avg_ttft_s": avg_ttft},
            "timestamps": {"last_update": time.strftime("%Y-%m-%d %H:%M:%S")},
        })

    # -------- Finalize --------
    # If we generated retrieval, ensure the final state is saved.
    if maybe_existing is None:
        _write_json(retrieval_json_path, samples)
        logger.info(f"[Output] Retrieval JSON: {retrieval_json_path}")

    # Final summary
    ttfts = [x["ttft_s"] for x in final_answers if x.get("ttft_s") is not None]
    avg_ttft = (sum(ttfts) / len(ttfts)) if ttfts else None
    _write_json(summary_path, {
        "status": "completed",
        "model": model_id,
        "device": device,
        "counts": {"samples": len(samples)},
        "outputs": {
            "retrieval_json": retrieval_json_path,
            "kv_summary_json": kv_summary_path,
            "speculative_json": pred_out_path,
            "scheduler_json": sched_out_path,
            "final_results_json": final_results_path,
            "kv_root": kv_root,
        },
        "metrics": {"avg_ttft_s": avg_ttft},
        "timestamps": {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    })
    logger.info(f"[pipeline] Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
