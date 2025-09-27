#!/usr/bin/env python3

import json
import argparse
from typing import Any, Dict, List


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def average_metrics(results_path: str) -> Dict[str, float]:
    data = _load_json(results_path)
    results: List[Dict[str, Any]] = data["results"] if isinstance(data, dict) and "results" in data else data

    sums = {
        "ttft_s": 0.0,
        "tpot_s": 0.0,
        "e2e_s": 0.0,
        "throughput_tps": 0.0,
        "num_output_tokens": 0.0,
    }
    counts = {k: 0 for k in sums.keys()}

    for r in results:
        m = r.get("metrics", {}) or {}

        # TTFT seconds (fallback from ms)
        if "ttft_s" in m or "ttft_ms" in m:
            ttft_s = _to_float(m.get("ttft_s", _to_float(m.get("ttft_ms", 0.0)) / 1000.0))
            sums["ttft_s"] += ttft_s
            counts["ttft_s"] += 1

        # TPOT seconds (fallback from ms)
        if "tpot_s" in m or "tpot_ms" in m:
            tpot_s = _to_float(m.get("tpot_s", _to_float(m.get("tpot_ms", 0.0)) / 1000.0))
            sums["tpot_s"] += tpot_s
            counts["tpot_s"] += 1

        # E2E seconds (fallback from ms)
        if "e2e_s" in m or "e2e_ms" in m:
            e2e_s = _to_float(m.get("e2e_s", _to_float(m.get("e2e_ms", 0.0)) / 1000.0))
            sums["e2e_s"] += e2e_s
            counts["e2e_s"] += 1

        # Throughput (tok/s) supports multiple key names
        throughput = None
        for k in ("throughput_toks_per_s", "throughput_tps", "throughput_tok_per_s"):
            if k in m:
                throughput = _to_float(m.get(k))
                break
        if throughput is not None:
            sums["throughput_tps"] += throughput
            counts["throughput_tps"] += 1

        # Output tokens
        tokens = m.get("num_output_tokens", m.get("num_tokens"))
        if tokens is not None:
            sums["num_output_tokens"] += _to_float(tokens)
            counts["num_output_tokens"] += 1

    avgs = {k: (sums[k] / counts[k] if counts[k] > 0 else 0.0) for k in sums}
    # Convenience milliseconds
    avgs.update({
        "ttft_ms": avgs["ttft_s"] * 1000.0,
        "tpot_ms": avgs["tpot_s"] * 1000.0,
        "e2e_ms": avgs["e2e_s"] * 1000.0,
        "num_samples": len(results),
    })
    return avgs


def main():
    ap = argparse.ArgumentParser("Compute average metrics from results JSON")
    ap.add_argument("--results", required=True, help="Path to result.json")
    ap.add_argument("--output", default=None, help="Optional output JSON path for averages")
    args = ap.parse_args()

    avgs = average_metrics(args.results)

    summary = {
        "num_samples": avgs["num_samples"],
        "avg_ttft_s": avgs["ttft_s"],
        "avg_tpot_s": avgs["tpot_s"],
        "avg_e2e_s": avgs["e2e_s"],
        "avg_ttft_ms": avgs["ttft_ms"],
        "avg_tpot_ms": avgs["tpot_ms"],
        "avg_e2e_ms": avgs["e2e_ms"],
        "avg_throughput_tps": avgs["throughput_tps"],
        "avg_num_output_tokens": avgs["num_output_tokens"],
    }

    print(json.dumps(summary, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()