#!/usr/bin/env python3
import os
import json
import time
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

import yaml
import torch
from threading import Thread
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# --- local modules ---
from rag_retrieval import RetrievalConfig, ColbertRetrieval
from build_kv_cache import KVCachesBuilder
from speculative_decoding import SpeculativeChunkPredictor
from scheduler import TangoScheduler


# ---------------- utilities ----------------

def _read_json(path: str) -> Any:
    """Read a JSON file and return the parsed object."""
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: str, obj: Any):
    """Write an object to JSON, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _load_samples(path: str) -> List[Dict[str, Any]]:
    """Load dataset samples from JSON supporting several common shapes."""
    data = _read_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return [data]


def _abs_path(base_dir: str, maybe_path: str) -> str:
    """Resolve a possibly relative path against a base directory."""
    if not maybe_path:
        return ""
    return maybe_path if os.path.isabs(maybe_path) else os.path.join(base_dir, maybe_path)


def _setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure module logging and return the pipeline logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("pipeline_full_kv_reuse_fixed")


def _load_existing_retrieval(retrieval_json_path: str, logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """If an existing retrieval JSON exists, load and return its samples."""
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


def _restructure_kv_cache_dir_to_numeric(sample_cache_dir: str, expected_sample_id: str, logger: logging.Logger):
    """Normalize chunk subfolders to numeric names (0,1,2,...) to simplify later joins."""
    if not sample_cache_dir or not os.path.isdir(sample_cache_dir):
        return
    for name in os.listdir(sample_cache_dir):
        src = os.path.join(sample_cache_dir, name)
        if not os.path.isdir(src):
            continue
        meta_path = os.path.join(src, "metadata.json")
        if not os.path.isfile(meta_path):
            # try parse from folder name
            if name.isdigit():
                continue
            if "_chunk" in name:
                try:
                    idx = int(name.split("_chunk")[-1])
                    dst = os.path.join(sample_cache_dir, str(idx))
                    if src != dst and not os.path.exists(dst):
                        os.rename(src, dst)
                except Exception:
                    pass
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            chunk_id = str(meta.get("chunk_id", ""))
            if "_chunk" not in chunk_id:
                continue
            sample_id_part, idx_part = chunk_id.rsplit("_chunk", 1)
            if expected_sample_id and sample_id_part != expected_sample_id:
                continue
            idx = int(idx_part)
            dst = os.path.join(sample_cache_dir, str(idx))
            if src != dst and not os.path.exists(dst):
                os.rename(src, dst)
        except Exception as e:
            logger.warning(f"[KV] Could not restructure '{src}' -> numeric: {e}")


def _topk_indices_from_sample(sample: Dict[str, Any], cfg: Dict[str, Any]) -> List[int]:
    """Return the numeric indices for the top‑k passages we want on GPU initially."""
    rkey = cfg["retrieval"]["retrieved_key"]
    retrieved = sample.get(rkey, []) or []
    max_gpu = int(cfg["cache"]["max_gpu_chunks"])
    return [int(i) for i in (retrieved[:max_gpu] if isinstance(retrieved, list) else [])]


def _prune_non_topk_kv_files(sample_dir: str, keep_indices: List[int], logger: logging.Logger):
    """Delete K/V tensor files for non‑top‑k chunks, keeping only metadata/placeholders."""
    keep_set = set(keep_indices)
    for name in os.listdir(sample_dir):
        if not name.isdigit():
            continue
        idx = int(name)
        if idx in keep_set:
            continue
        cdir = os.path.join(sample_dir, name)
        for fname in ("keys.pt", "values.pt", "valid_mask.pt"):
            fpath = os.path.join(cdir, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except Exception as e:
                    logger.warning(f"[KV] Could not remove {fpath}: {e}")


def _compute_final_gpu_indices(initial_gpu_indices: List[int], sched_payload: Dict[str, Any]) -> List[int]:
    """Apply scheduler promotions/evictions on top of our initial top‑k GPU set."""
    gpu_set = set(int(i) for i in initial_gpu_indices)
    for row in sched_payload.get("trace", []):
        for cid in row.get("promoted", []):
            if isinstance(cid, str) and "_chunk" in cid:
                try:
                    gpu_set.add(int(cid.split("_chunk")[1]))
                except Exception:
                    pass
            elif isinstance(cid, int):
                gpu_set.add(cid)
        for cid in row.get("evicted", []):
            if isinstance(cid, str) and "_chunk" in cid:
                try:
                    gpu_set.discard(int(cid.split("_chunk")[1]))
                except Exception:
                    pass
            elif isinstance(cid, int):
                gpu_set.discard(cid)
    return sorted(gpu_set)


# ---------------- KV load/merge + decoding ----------------

def _load_and_merge_kvs(
    sample_dir: str,
    indices: List[int],
    device: torch.device,
    model_dtype: torch.dtype,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Load K/V tensors for the requested chunk indices and concatenate along the
    sequence dimension: returns a past_key_values structure compatible with HF.
    Skips indices without real K/V files (e.g., placeholders).
    """
    if not indices:
        return None

    merged_k: List[List[torch.Tensor]] = []
    merged_v: List[List[torch.Tensor]] = []
    num_layers = None

    def _normalize_kv(k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Accept [L,H,S,D] or [L,B,H,S,D] -> return [L,1,H,S,D]
        if k.ndim == 4:
            k = k.unsqueeze(1)
        if v.ndim == 4:
            v = v.unsqueeze(1)
        return k, v

    for idx in indices:
        cdir = os.path.join(sample_dir, str(idx))
        k_path = os.path.join(cdir, "keys.pt")
        v_path = os.path.join(cdir, "values.pt")
        if not (os.path.isfile(k_path) and os.path.isfile(v_path)):
            continue  # placeholder; background worker should materialize it
        k = torch.load(k_path, map_location="cpu")
        v = torch.load(v_path, map_location="cpu")
        k, v = _normalize_kv(k, v)  # [L,1,H,S,D]
        if num_layers is None:
            num_layers = k.shape[0]
            merged_k = [[] for _ in range(num_layers)]
            merged_v = [[] for _ in range(num_layers)]
        assert k.shape[0] == num_layers and v.shape[0] == num_layers
        for L in range(num_layers):
            merged_k[L].append(k[L, 0])  # [H,S,D]
            merged_v[L].append(v[L, 0])

    if num_layers is None:
        return None

    past: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for L in range(num_layers):
        if not merged_k[L]:
            return None
        k_cat = torch.cat(merged_k[L], dim=1).unsqueeze(0).to(device=device, dtype=model_dtype)  # [1,H,S,D]
        v_cat = torch.cat(merged_v[L], dim=1).unsqueeze(0).to(device=device, dtype=model_dtype)
        past.append((k_cat, v_cat))
    return past


class FinalDecoder:
    """Minimal decoder wrapper to stream text and measure latency metrics."""

    def __init__(self, model_id: str, device: str):
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.mdl = AutoModelForCausalLM.from_pretrained(model_id).to(self.device).eval()

    def _stream_generate(self, gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Run `generate` on a background thread and consume a TextIteratorStreamer."""
        first_tok_t = None
        pieces: List[str] = []
        streamer = gen_kwargs["streamer"]

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

        num_toks = len(pieces)
        e2e = t1 - t0
        ttft = (first_tok_t - t0) if first_tok_t is not None else 0.0
        throughput = (num_toks / e2e) if e2e > 0 else 0.0
        tpot = (e2e / num_toks) if num_toks > 0 else 0.0

        return {
            "text": "".join(pieces),
            "ttft": ttft,
            "e2e_latency": e2e,
            "throughput": throughput,
            "tpot": tpot,
            "decoded_tokens": num_toks,
        }

    def decode_with_saved_kvs(
        self,
        question_text: str,
        past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Decode `Answer:` given a question suffix, optionally with past KV."""
        suffix = f"Question: {question_text}\nAnswer:"
        enc = self.tok([suffix], return_tensors="pt", padding=False, truncation=False)
        input_ids = enc["input_ids"].to(self.device)

        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs: Dict[str, Any] = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            streamer=streamer,
            return_dict_in_generate=True,
        )

        if past is not None and len(past) > 0 and past[0][0].shape[2] > 0:
            cached_len = past[0][0].shape[2]
            position_ids = torch.arange(cached_len, cached_len + input_ids.shape[1], device=self.device).unsqueeze(0)
            cache_position = torch.arange(cached_len, cached_len + input_ids.shape[1], device=self.device)
            gen_kwargs.update({
                "past_key_values": tuple((k.contiguous(), v.contiguous()) for k, v in past),
                "position_ids": position_ids,
                "cache_position": cache_position,
            })

        out = self._stream_generate(gen_kwargs)
        return {
            "answer": out["text"].strip(),
            "ttft": out["ttft"],
            "e2e_latency": out["e2e_latency"],
            "throughput": out["throughput"],
            "tpot": out["tpot"],
            "decoded_tokens": out["decoded_tokens"],
        }


# ---------------- main pipeline ----------------

def main():
    """Entry point: runs retrieval → KV build (top‑k only) → decode → schedule → decode."""
    ap = argparse.ArgumentParser("Full KV Reuse Pipeline WITH speculative + scheduler (fixed)")
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
    logger.info(f"[Input] Loaded {len(samples)} samples")

    # Output paths
    retrieval_json_path = _abs_path(args.output, cfg["paths"]["retrieval_json"])
    kv_summary_path = _abs_path(args.output, cfg["paths"]["kv_summary_json"])
    pred_out_path = _abs_path(args.output, cfg["speculative"].get("out_path", "speculative_next.json"))
    sched_out_path = _abs_path(args.output, cfg["scheduler"].get("out_path", "scheduler/decoder_trace.json"))
    final_results_path = os.path.join(args.output, "final_results.json")
    summary_path = os.path.join(args.output, "summary.json")

    # Retrieval (skip if JSON already exists)
    maybe_existing = _load_existing_retrieval(retrieval_json_path, logger)
    if maybe_existing is not None:
        samples = maybe_existing

    # Objects
    retriever = None
    if maybe_existing is None:
        rconf = RetrievalConfig(
            model_id=cfg["retrieval"]["model_id"],
            dataset_name=cfg["retrieval"]["dataset_name"],
            r_text_index_key=cfg["retrieval"]["r_text_index_key"],
            doc_key=cfg["retrieval"]["doc_key"],
            question_key=cfg["retrieval"]["question_key"],
            retrieved_key=cfg["retrieval"]["retrieved_key"],
            page_id_key=cfg["retrieval"]["page_id_key"],
            top_k=int(cfg["retrieval"]["top_k"]),
        )
        retriever = ColbertRetrieval(rconf)

    kv_builder = KVCachesBuilder()
    predictor = SpeculativeChunkPredictor()
    scheduler = TangoScheduler()
    decoder = FinalDecoder(model_id=model_id, device=device)

    aggregated_kv: List[Dict[str, Any]] = []
    pred_trace_all: List[Dict[str, Any]] = []
    pred_results_all: List[Dict[str, Any]] = []
    sched_trace_all: List[Dict[str, Any]] = []
    final_answers: List[Dict[str, Any]] = []

    kv_root = _abs_path(args.output, cfg["kv_builder"].get("save_cache_dir", "kv_caches"))

    logger.info("[Pipeline] Running full_kv_reuse + speculative + scheduler (fixed)")
    for si in tqdm(range(len(samples)), desc="Samples", unit="sample"):
        sample = samples[si]
        sample_id = str(sample.get("id", f"sample{si}"))

        # 1) Retrieval (if not preloaded) -> flush retrieval.json
        if maybe_existing is None:
            enriched = retriever.prepare([sample])
            enriched = retriever.retrieve(enriched)
            sample = enriched[0]
            samples[si] = sample
            _write_json(retrieval_json_path, samples)

        # 2) KV Cache Build: only top‑k keep real K/V; others placeholders
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
            save_placeholders=True,  # placeholders for non‑top‑k
            retrieval_json_path=retrieval_json_path if os.path.exists(retrieval_json_path) else "",
        )

        _restructure_kv_cache_dir_to_numeric(sample_dir, expected_sample_id=str(sample_id), logger=logger)

        # --- NEW: Cut non‑top‑k K/V files so only top‑k are materialized now ---
        topk_indices = _topk_indices_from_sample(sample, cfg)
        _prune_non_topk_kv_files(sample_dir, topk_indices, logger)

        aggregated_kv.append({"sample_index": si, **kv_payload})
        _write_json(kv_summary_path, {"per_sample": aggregated_kv, "kv_root": kv_root})

        # Start set on GPU = **top‑k only**
        gpu_indices_initial = sorted(set(int(i) for i in topk_indices))

        # 3) Initial decoding (with current GPU set)
        past_init = _load_and_merge_kvs(sample_dir, gpu_indices_initial, decoder.device, next(decoder.mdl.parameters()).dtype)
        decode_metrics_initial = decoder.decode_with_saved_kvs(
            question_text=sample.get("question", ""),
            past=past_init,
            max_new_tokens=cfg.get("generation", {}).get("max_new_tokens", 128),
        )

        # 4) Speculative prediction (propose future chunks needed)
        pred_payload = predictor.predict(
            samples=[sample],
            model_id=model_id,
            device=device,
            top_k=int(cfg["speculative"]["top_k"]),
            steps=int(cfg["speculative"]["steps"]),
            promote_per_step=int(cfg["speculative"]["promote_per_step"]),
            max_gpu=int(cfg["speculative"]["max_gpu"]),
            max_samples=1,
            enable_progress=bool(cfg["speculative"].get("enable_progress", False)),
            out_path=None,
        )
        for row in pred_payload.get("trace", []):
            row["sample_index"] = si
        for row in pred_payload.get("results", []):
            row["sample_index"] = si
        pred_trace_all.extend(pred_payload.get("trace", []))
        pred_results_all.extend(pred_payload.get("results", []))
        _write_json(pred_out_path, {"trace": pred_trace_all, "results": pred_results_all})

        # 5) Scheduler (promote/evict across CPU/GPU)
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
            cache_filter_prefix="",              # numeric subdirs
            load_initial_to_gpu=True,            # **ensure** initial top‑k stays on GPU
            out_path=None,
            enable_progress=bool(cfg["scheduler"].get("enable_progress", False)),
        )
        for row in sched_payload.get("trace", []):
            row["sample_index"] = si
        sched_trace_all.extend(sched_payload.get("trace", []))
        _write_json(sched_out_path, {"trace": sched_trace_all})

        # 6) Final decoding with scheduled GPU set (seeded by our initial top‑k)
        final_gpu_indices = _compute_final_gpu_indices(gpu_indices_initial, sched_payload)
        past_final = _load_and_merge_kvs(sample_dir, final_gpu_indices, decoder.device, next(decoder.mdl.parameters()).dtype)
        decode_metrics_final = decoder.decode_with_saved_kvs(
            question_text=sample.get("question", ""),
            past=past_final,
            max_new_tokens=cfg.get("generation", {}).get("max_new_tokens", 128),
        )

        # Record final row (initial metrics kept in case you wish to log separately)
        final_answers.append({
            "sample_index": si,
            "question": sample.get("question", ""),
            "mode": "full_kv_reuse",
            "gpu_indices_initial": gpu_indices_initial,
            "gpu_indices_final": final_gpu_indices,
            **decode_metrics_final,
        })

        # Flush rolling results + summary
        ttfts = [x["ttft"] for x in final_answers if x.get("ttft") is not None]
        e2es = [x["e2e_latency"] for x in final_answers if x.get("e2e_latency") is not None]
        thr = [x["throughput"] for x in final_answers if x.get("throughput") is not None]
        tpot = [x["tpot"] for x in final_answers if x.get("tpot") is not None]

        _write_json(final_results_path, {
            "per_sample": final_answers,
            "averages": {
                "ttft": (sum(ttfts) / len(ttfts)) if ttfts else None,
                "e2e_latency": (sum(e2es) / len(e2es)) if e2es else None,
                "throughput": (sum(thr) / len(thr)) if thr else None,
                "tpot": (sum(tpot) / len(tpot)) if tpot else None,
            },
            "count": len(final_answers)
        })

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
            "timestamps": {"last_update": time.strftime("%Y-%m-%d %H:%M:%S")},
        })

    # Finalize
    if maybe_existing is None:
        _write_json(retrieval_json_path, samples)
        logger.info(f"[Output] Retrieval JSON: {retrieval_json_path}")

    ttfts = [x["ttft"] for x in final_answers if x.get("ttft") is not None]
    e2es = [x["e2e_latency"] for x in final_answers if x.get("e2e_latency") is not None]
    thr = [x["throughput"] for x in final_answers if x.get("throughput") is not None]
    tpot = [x["tpot"] for x in final_answers if x.get("tpot") is not None]

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
        "averages": {
            "ttft": (sum(ttfts) / len(ttfts)) if ttfts else None,
            "e2e_latency": (sum(e2es) / len(e2es)) if e2es else None,
            "throughput": (sum(thr) / len(thr)) if thr else None,
            "tpot": (sum(tpot) / len(tpot)) if tpot else None,
        },
        "timestamps": {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    })
    logger.info(f"[pipeline] Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
