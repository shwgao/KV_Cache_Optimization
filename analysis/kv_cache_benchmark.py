import argparse
import os
import time
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

try:
    from .configs import bytes_to_gib
except Exception:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from configs import bytes_to_gib


# ---------- Helpers ----------

def setup_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but device='cuda' was requested.")
    return torch.device(device)


def precision_to_bytes(precision: str) -> int:
    if precision == "fp16" or precision == "bf16":
        return 2
    elif precision == "int8":
        return 1
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def get_gpu_props(device: torch.device) -> Dict[str, Any]:
    if device.type != "cuda":
        return {"gpu_name": "cpu", "vram_bytes": 0, "vram_gib": 0.0}
    idx = device.index if device.index is not None else 0
    props = torch.cuda.get_device_properties(idx)
    total = props.total_memory
    return {
        "gpu_name": props.name,
        "vram_bytes": int(total),
        "vram_gib": bytes_to_gib(int(total)),
    }


def load_model(model_name: str, precision: str, device: torch.device):
    """
    Loads the model in the requested precision. For int8, uses bitsandbytes with device_map='auto'.
    Returns (model, tokenizer, primary_device) where primary_device is where inputs should be placed
    (for single-device models) or the requested device (best-effort) for multi-device maps.
    """
    dtype = None
    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
        ).to(device)
        primary_device = device
    elif precision == "int8":
        try:
            # Requires bitsandbytes
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
            )
        except Exception as e:
            raise RuntimeError("Failed to load model in 8-bit. Ensure bitsandbytes is installed.") from e
        # Heuristic: put inputs on cuda:0 when using auto device map
        primary_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, primary_device


def measure_peak_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        return int(peak)
    return 0


def estimate_weights_bytes(model) -> Optional[int]:
    """
    Try to estimate weights memory on device.
    - If the model exposes get_memory_footprint (works when fully on a single device), prefer it.
    - Fallback: sum of parameters * element_size (works for 8-bit and 16-bit tensors).
    """
    try:
        # If model is on CUDA and fully materialized, this is a decent estimate
        fp = int(model.get_memory_footprint())
        if fp > 0:
            return fp
    except Exception:
        pass

    try:
        total = 0
        for p in model.parameters():
            if hasattr(p, "data") and torch.is_tensor(p.data):
                total += p.numel() * p.element_size()
        return int(total)
    except Exception:
        return None


def get_layers_and_hidden(model) -> Dict[str, int]:
    """
    Robustly extract (layers, hidden_size) from HF config across families.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("Model has no config; cannot infer layers/hidden_size.")

    # Try the common names first
    layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    if layers is None or hidden is None:
        raise RuntimeError(f"Could not infer (num_hidden_layers, hidden_size) from config: {cfg}")
    return {"layers": int(layers), "hidden_size": int(hidden)}


def estimate_kv_bytes(model, seq_len: int, batch_size: int, precision: str) -> int:
    spec = get_layers_and_hidden(model)
    pbytes = precision_to_bytes(precision)
    kv = 2 * spec["layers"] * seq_len * spec["hidden_size"] * pbytes * batch_size
    return int(kv)


def make_inputs_exact_tokens(tokenizer, seq_len: int, batch_size: int, device: torch.device):
    """
    Build input_ids and attention_mask that are EXACTLY seq_len long per batch
    by repeating the EOS token id. This avoids tokenization variance.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        # Fallback to 0 if EOS unknown
        eos_id = 0
    input_ids = torch.full((batch_size, seq_len), fill_value=eos_id, dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    return input_ids, attention_mask


# ---------- Benchmark core ----------

def benchmark_once(model, tokenizer, primary_device: torch.device,
                   seq_len: int, batch_size: int, max_new_tokens: int = 1) -> Dict[str, Any]:
    # Reset memory stats for the specific device if CUDA
    if primary_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(primary_device)

    # Build deterministic inputs with exact seq_len
    input_ids, attention_mask = make_inputs_exact_tokens(tokenizer, seq_len, batch_size, primary_device)

    # Warmup (short)
    with torch.inference_mode():
        _ = model(input_ids[:, :8], attention_mask=attention_mask[:, :8])

    if primary_device.type == "cuda":
        torch.cuda.synchronize(primary_device)
        torch.cuda.reset_peak_memory_stats(primary_device)

    start = time.time()
    succeeded = True
    err_msg = ""
    try:
        with torch.inference_mode():
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if primary_device.type == "cuda":
            torch.cuda.synchronize(primary_device)
    except RuntimeError as e:
        succeeded = False
        err_msg = str(e)

    peak_bytes = measure_peak_bytes(primary_device)
    end = time.time()

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "succeeded": succeeded,
        "error": err_msg,
        "peak_bytes": peak_bytes,
        "peak_gib": bytes_to_gib(peak_bytes),
        "elapsed_s": end - start,
    }


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Benchmark KV/memory by running short generation at given sequence lengths.")
    parser.add_argument("--model", required=True, help="HF model id or local path (e.g., meta-llama/Llama-3-8B-Instruct)")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "int8"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[16384, 32768, 65536, 131072, 262144])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--data", default="../../inputs/musique_s.json", help="Path to data file")
    parser.add_argument("--out-csv", default="baselines/CacheBlend/analysis/results/kv_cache_benchmark.csv")
    args = parser.parse_args()

    device = setup_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    model, tokenizer, primary_device = load_model(args.model, args.precision, device)
    model.eval()

    # Make tokenizer permissive about long seqs (avoid internal truncation)
    try:
        tokenizer.model_max_length = int(10**9)
    except Exception:
        pass

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    gpu = get_gpu_props(primary_device)
    weights_bytes = estimate_weights_bytes(model)
    weights_gib = bytes_to_gib(weights_bytes) if weights_bytes is not None else None

    results: List[Dict[str, Any]] = []

    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            print(f"Running seq_len={seq_len}, batch_size={batch_size}...", flush=True)

            # Theory estimates for this (seq_len, batch_size)
            kv_bytes_theory = estimate_kv_bytes(model, seq_len, batch_size, args.precision)
            kv_gib_theory = bytes_to_gib(kv_bytes_theory)

            res = benchmark_once(
                model, tokenizer, primary_device,
                seq_len=seq_len, batch_size=batch_size, max_new_tokens=args.max_new_tokens
            )

            # Theory vs VRAM check (rough — excludes activations/optimizer/etc.)
            vram_bytes = gpu["vram_bytes"]
            theory_total = (weights_bytes or 0) + kv_bytes_theory
            predicted_oom = (vram_bytes > 0) and (theory_total > vram_bytes)

            out_row = {
                "model": args.model,
                "precision": args.precision,
                "device": str(primary_device),
                "gpu_name": gpu["gpu_name"],
                "vram_gib": gpu["vram_gib"],

                "seq_len": seq_len,
                "batch_size": batch_size,

                "kv_bytes_theory": kv_bytes_theory,
                "kv_gib_theory": kv_gib_theory,
                "weights_bytes_est": weights_bytes,
                "weights_gib_est": weights_gib,

                "predicted_oom_by_theory": bool(predicted_oom),

                "succeeded": res["succeeded"],
                "status": "Success" if res["succeeded"] else "OOM Error" if "out of memory" in res["error"].lower() else "Error",
                "error": res["error"],

                "peak_bytes": res["peak_bytes"],
                "peak_gib": res["peak_gib"],
                "elapsed_s": res["elapsed_s"],

                # Headroom using measured peak
                "headroom_gib_measured": (gpu["vram_gib"] - res["peak_gib"]) if gpu["vram_gib"] else None,
                # Simple pass/fail by measured peak vs VRAM
                "fits_by_peak": (res["peak_gib"] <= gpu["vram_gib"]) if gpu["vram_gib"] else None,
            }

            results.append(out_row)

            # Clear cache between runs to avoid accumulation effects
            if primary_device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(results)

    # Nice ordering
    cols = [
        "model", "precision", "gpu_name", "device", "vram_gib",
        "seq_len", "batch_size",
        "kv_gib_theory", "weights_gib_est", "predicted_oom_by_theory",
        "status", "succeeded", "peak_gib", "headroom_gib_measured", "elapsed_s",
        "kv_bytes_theory", "weights_bytes_est", "peak_bytes",
    ]
    cols = [c for c in cols if c in df.columns]  # guard
    df = df[cols]

    df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
    # Show the last few lines for convenience
    print(df.tail().to_string(index=False))


if __name__ == "__main__":
    main()
