#!/usr/bin/env python3
import os
import sys
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
from prefill import CacheBlendFuser, build_chunk_ids  # vLLM fused prefill (CacheBlend-style)


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
            # Try parse from folder name
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


# ---------------- KV-aware decoding helpers ----------------

def _load_and_merge_kvs(
    sample_dir: str,
    indices: List[int],
    device: torch.device,
    model_dtype: torch.dtype,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Load keys/values for the selected chunk indices and merge them per layer.
    Expects per-chunk folders: {sample_dir}/{idx}/keys.pt, values.pt, valid_mask.pt
    Returns HF-style past_key_values: List[(k, v)] with shapes [1, H, S_total, D].
    """
    if not indices:
        return None

    # Accumulators per layer
    merged_k: List[List[torch.Tensor]] = []
    merged_v: List[List[torch.Tensor]] = []
    num_layers = None

    def _normalize_kv(k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Accepts k,v as 4D [L, H, S, D] or 5D [L, B, H, S, D]; returns 5D [L, 1, H, S, D]
        """
        if k.ndim == 4:
            k = k.unsqueeze(1)  # add batch=1 -> [L, 1, H, S, D]
        if v.ndim == 4:
            v = v.unsqueeze(1)
        return k, v

    for idx in indices:
        cdir = os.path.join(sample_dir, str(idx))
        k_path = os.path.join(cdir, "keys.pt")
        v_path = os.path.join(cdir, "values.pt")
        if not (os.path.isfile(k_path) and os.path.isfile(v_path)):
            continue

        k = torch.load(k_path, map_location="cpu")
        v = torch.load(v_path, map_location="cpu")
        k, v = _normalize_kv(k, v)  # [L, 1, H, S, D]
        if num_layers is None:
            num_layers = k.shape[0]
            merged_k = [[] for _ in range(num_layers)]
            merged_v = [[] for _ in range(num_layers)]
        assert k.shape[0] == num_layers and v.shape[0] == num_layers, "KV layer mismatch"

        for L in range(num_layers):
            # take [1, H, S, D]
            kL = k[L, 0]  # [H, S, D]
            vL = v[L, 0]  # [H, S, D]
            merged_k[L].append(kL)
            merged_v[L].append(vL)

    if num_layers is None:
        return None

    past: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for L in range(num_layers):
        if not merged_k[L]:
            # No KVs loaded for this layer
            return None
        # Concat along sequence length (dim=1 since shape is [H, S, D])
        k_cat = torch.cat(merged_k[L], dim=1).unsqueeze(0).to(device=device, dtype=model_dtype)  # [1,H,S,D]
        v_cat = torch.cat(merged_v[L], dim=1).unsqueeze(0).to(device=device, dtype=model_dtype)  # [1,H,S,D]
        past.append((k_cat, v_cat))
    return past

def _compute_final_gpu_indices(
    pred_payload: Dict[str, Any],
    sched_payload: Dict[str, Any],
    sample_id: str,
) -> List[int]:
    """
    Reconstruct final GPU-resident chunk indices for this sample by applying
    scheduler promotions/evictions over initial set.
    """
    init = []
    if pred_payload.get("results"):
        init = list(map(int, pred_payload["results"][0].get("initial_gpu", [])))
    gpu_set = set(init)

    for row in sched_payload.get("trace", []):
        # rows already correspond to this sample (we only passed [sample] to scheduler)
        for cid in row.get("promoted", []):
            if isinstance(cid, str) and "_chunk" in cid:
                try:
                    i = int(cid.split("_chunk")[1])
                    gpu_set.add(i)
                except Exception:
                    pass
        for cid in row.get("evicted", []):
            if isinstance(cid, str) and "_chunk" in cid:
                try:
                    i = int(cid.split("_chunk")[1])
                    gpu_set.discard(i)
                except Exception:
                    pass

    return sorted(gpu_set)

# ---------------- decoding helpers (baseline HF; plus a KV-aware method) ----------------

class FinalDecoder:
    def __init__(self, model_id: str, device: str):
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.mdl = AutoModelForCausalLM.from_pretrained(model_id).to(self.device).eval()

    def _stream_generate(self, gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
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

        ttft_s = (first_tok_t - t0) if first_tok_t is not None else None
        gen_time_s = t1 - (first_tok_t or t0)
        tps = (len(pieces) or 1) / max(1e-9, gen_time_s)
        return {
            "text": "".join(pieces),
            "ttft_s": ttft_s,
            "tokens_per_sec": tps,
            "decoded_tokens": len(pieces),
        }

    def decode(self, sample: Dict[str, Any], max_new_tokens: int = 128) -> Dict[str, Any]:
        # Baseline (no KV injection)
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
        out = self._stream_generate(gen_kwargs)
        return {
            "answer": out["text"].strip(),
            "ttft_s": out["ttft_s"],
            "tokens_per_sec": out["tokens_per_sec"],
            "decoded_tokens": out["decoded_tokens"],
        }

    def decode_with_saved_kvs(
        self,
        sample: Dict[str, Any],
        sample_dir: str,
        gpu_indices: List[int],
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        """
        Use saved KVs (merged across GPU-selected chunks) as past_key_values, then
        feed only the Question + 'Answer:' as the suffix to prefill.
        """
        past = _load_and_merge_kvs(
            sample_dir=sample_dir,
            indices=gpu_indices,
            device=self.device,
            model_dtype=next(self.mdl.parameters()).dtype,
        )
        if past is None:
            # Fallback
            return self.decode(sample, max_new_tokens=max_new_tokens)

        # Suffix = Question + "Answer:" to drive decoding
        suffix = f"Question: {sample.get('question','')}\nAnswer:"
        input_ids = self.tok([suffix], return_tensors="pt", padding=False, truncation=False)["input_ids"].to(self.device)

        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=input_ids,
            past_key_values=past,       # <-- inject prefilled memory
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            streamer=streamer,
            return_dict_in_generate=True,
        )
        out = self._stream_generate(gen_kwargs)
        return {
            "answer": out["text"].strip(),
            "ttft_s": out["ttft_s"],
            "tokens_per_sec": out["tokens_per_sec"],
            "decoded_tokens": out["decoded_tokens"],
        }


# ---------------- main pipeline ----------------

def main():
    ap = argparse.ArgumentParser("CacheBlend Pipeline v1 (sample-by-sample, streaming outputs, switchable prefill)")
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
    prefill_mode = (cfg.get("prefill", {}) or {}).get("mode", "full_kv_reuse")  # cb_fuse | full_kv_reuse | recompute
    logger.info(f"[Config] Model: {model_id} | Device: {device} | Prefill mode: {prefill_mode}")
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

    # Optional CacheBlend fuser (vLLM)
    fuser = None
    if prefill_mode == "cb_fuse":
        cb_cfg = cfg.get("prefill", {}).get("cb", {}) or {}
        fuser = CacheBlendFuser(
            model_id=model_id,
            gpu_mem_util=cfg["model"].get("gpu_mem_util", 0.5)
        )
        # store prompts in memory for convenience
        fuser._cb_prefix_prompt = cb_cfg.get("prefix_prompt", "")
        fuser._cb_query_prompt  = cb_cfg.get("query_prompt", "")

    # Aggregates (we will FLUSH after each sample)
    aggregated_kv: List[Dict[str, Any]] = []
    pred_trace_all: List[Dict[str, Any]] = []
    pred_results_all: List[Dict[str, Any]] = []
    sched_trace_all: List[Dict[str, Any]] = []
    final_answers: List[Dict[str, Any]] = []

    # KV root per requested structure (only used by full_kv_reuse)
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
            _write_json(retrieval_json_path, samples)  # flush

        # ----------------- MODE BRANCH -----------------
        if prefill_mode == "full_kv_reuse":
            # 2) KV Cache Build (per-sample)
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
            _write_json(kv_summary_path, {"per_sample": aggregated_kv, "kv_root": kv_root})  # flush

            # 3) Speculative Prediction (per-sample)
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
            _write_json(pred_out_path, {"trace": pred_trace_all, "results": pred_results_all})  # flush

            # 4) Scheduler (per-sample)
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
            _write_json(sched_out_path, {"trace": sched_trace_all})  # flush

            # -------- 5) Final Decoding (USE SAVED KVs) --------
            final_gpu_indices = _compute_final_gpu_indices(pred_payload, sched_payload, sample_id)
            decode_metrics = decoder.decode_with_saved_kvs(
                sample=sample,
                sample_dir=sample_dir,
                gpu_indices=final_gpu_indices,
                max_new_tokens=cfg.get("generation", {}).get("max_new_tokens", 128),
            )
            final_answers.append({
                "sample_index": si,
                "question": sample.get("question", ""),
                **decode_metrics,
                "mode": "full_kv_reuse",
                "gpu_indices_used": final_gpu_indices,
            })

        elif prefill_mode == "cb_fuse":
            # 2) CacheBlend-style fused prefill via vLLM (no disk KV build)
            topk = int(cfg["retrieval"]["top_k"])
            ridxs = [int(i) for i in (sample.get("retrieved_indices") or [])][:topk]
            ctxs = sample.get("ctxs", [])
            doc_prompts = []
            for i in ridxs:
                if 0 <= i < len(ctxs):
                    title = (ctxs[i].get("title") or "").strip()
                    text  = (ctxs[i].get("text")  or "").strip()
                    doc_prompts.append((f"{title}\n{text}".strip() if title else text) or "")

            prefix_prompt = getattr(fuser, "_cb_prefix_prompt", "")
            query_prompt  = getattr(fuser, "_cb_query_prompt", "")

            cb_save_dir = os.path.join(args.output, "results/pipeline/cb_kv_cache", f"sample{si+1}")
            os.makedirs(cb_save_dir, exist_ok=True)
            # (re)build fuser with save path (or set fuser.save_cache_dir = cb_save_dir)
            fuser = CacheBlendFuser(
                model_id=model_id,
                gpu_mem_util=cfg["model"].get("gpu_mem_util", 0.5),
                save_cache_dir=cb_save_dir,  # <-- this makes collect() persist per-chunk KVs
            )
            fuser._cb_prefix_prompt = prefix_prompt
            fuser._cb_query_prompt  = query_prompt

            # tokenize chunks/question with required sentinels
            doc_chunk_ids, q_ids, meta = build_chunk_ids(
                fuser.tokenizer,
                doc_prompts=doc_prompts,
                q_prompt=query_prompt,
                prefix_prompt=prefix_prompt,
            )

            # 3) Collect fused KVs (prefill w/o decoding) + SAVE per-chunk KVs to cb_save_dir
            fuser.collect(doc_chunk_ids, meta, sample_id=str(sample.get("id", f"sample{si}")))
            # 4) Decode with fused cache (optional, keeps current behavior)
            input_ids = fuser.stitch_input_ids(doc_chunk_ids, meta)
            out = fuser.decode_with_fused(
                input_ids,
                suffix_len=meta["last_len"],
                max_new_tokens=int(cfg.get("prefill", {}).get("cb", {}).get("max_new_tokens", 128)),
            )

            final_answers.append({
                "sample_index": si,
                "question": sample.get("question", ""),
                "answer": out["text"].strip(),
                "ttft_s": out["ttft_s"],
                "tokens_per_sec": out.get("tokens_per_sec"),
                "decoded_tokens": out.get("decoded_tokens"),
                "mode": "cb_fuse",
            })

            # write empty placeholders so downstream consumers find the files
            if not os.path.exists(kv_summary_path):
                _write_json(kv_summary_path, {"per_sample": [], "kv_root": None})
            if not os.path.exists(pred_out_path):
                _write_json(pred_out_path, {"trace": [], "results": []})
            if not os.path.exists(sched_out_path):
                _write_json(sched_out_path, {"trace": []})

        else:  # "recompute"
            # Skip KV builder, predictor, scheduler; just decode vanilla HF
            decode_metrics = decoder.decode(sample, max_new_tokens=cfg.get("generation", {}).get("max_new_tokens", 128))
            final_answers.append({
                "sample_index": si,
                "question": sample.get("question", ""),
                **decode_metrics,
                "mode": "recompute",
            })
            # write empty placeholders (first time only)
            if not os.path.exists(kv_summary_path):
                _write_json(kv_summary_path, {"per_sample": [], "kv_root": None})
            if not os.path.exists(pred_out_path):
                _write_json(pred_out_path, {"trace": [], "results": []})
            if not os.path.exists(sched_out_path):
                _write_json(sched_out_path, {"trace": []})

        # 6) Flush rolling final results + summary
        ttfts = [x["ttft_s"] for x in final_answers if x.get("ttft_s") is not None]
        avg_ttft = (sum(ttfts) / len(ttfts)) if ttfts else None
        _write_json(final_results_path, {
            "per_sample": final_answers,
            "avg_ttft_s": avg_ttft,
            "count": len(final_answers)
        })

        _write_json(summary_path, {
            "status": "running",
            "processed": si + 1,
            "total": len(samples),
            "model": model_id,
            "device": device,
            "prefill_mode": prefill_mode,
            "outputs": {
                "retrieval_json": retrieval_json_path,
                "kv_summary_json": kv_summary_path,
                "speculative_json": pred_out_path,
                "scheduler_json": sched_out_path,
                "final_results_json": final_results_path,
                "kv_root": kv_root if prefill_mode == "full_kv_reuse" else None,
            },
            "metrics": {"avg_ttft_s": avg_ttft},
            "timestamps": {"last_update": time.strftime("%Y-%m-%d %H:%M:%S")},
        })

    # -------- Finalize --------
    if maybe_existing is None:
        _write_json(retrieval_json_path, samples)
        logger.info(f"[Output] Retrieval JSON: {retrieval_json_path}")

    ttfts = [x["ttft_s"] for x in final_answers if x.get("ttft_s") is not None]
    avg_ttft = (sum(ttfts) / len(ttfts)) if ttfts else None
    _write_json(summary_path, {
        "status": "completed",
        "model": model_id,
        "device": device,
        "prefill_mode": prefill_mode,
        "counts": {"samples": len(samples)},
        "outputs": {
            "retrieval_json": retrieval_json_path,
            "kv_summary_json": kv_summary_path,
            "speculative_json": pred_out_path,
            "scheduler_json": sched_out_path,
            "final_results_json": final_results_path,
            "kv_root": kv_root if prefill_mode == "full_kv_reuse" else None,
        },
        "metrics": {"avg_ttft_s": avg_ttft},
        "timestamps": {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    })
    logger.info(f"[pipeline] Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
