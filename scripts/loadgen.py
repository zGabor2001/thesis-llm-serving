#!/usr/bin/env python3
"""Concurrency load generator for the LLM gateway.

Prints a latency/throughput summary and appends a structured per-run record to
logs/loadgen.jsonl (override with --out, disable with --no-write).

Usage:
  python scripts/loadgen.py --url http://<gateway-host>:8080 \
      --requests 200 --concurrency 8 --tenant tenant-a --model base-model
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path

import requests

DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "logs" / "loadgen.jsonl"

# Short / medium / long prompts cycled by --variable-prompt to create
# heterogeneous per-request token costs for scheduler experiments.
VARIABLE_PROMPTS = [
    "Say hello in one sentence.",
    (
        "Describe the tradeoffs between batch size and latency when serving "
        "large language models, in three detailed sentences."
    ),
    (
        "Write a thorough technical explanation of how paged attention improves "
        "KV-cache memory utilization in vLLM, covering fragmentation, block "
        "tables, and how it enables denser continuous batching across many "
        "concurrent requests. Include several concrete examples and discuss the "
        "implications for prefill and decode latency."
    ),
]


def make_prompt(args: argparse.Namespace, i: int) -> str:
    if args.variable_prompt:
        return VARIABLE_PROMPTS[i % len(VARIABLE_PROMPTS)]
    return args.prompt


def send(args: argparse.Namespace, i: int) -> tuple[float, int]:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": make_prompt(args, i)}],
        "max_tokens": args.max_tokens,
        "tenant": args.tenant,
    }
    t0 = time.time()
    r = requests.post(
        f"{args.url}/v1/chat/completions", json=payload, timeout=args.timeout
    )
    return time.time() - t0, r.status_code


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, int(round(p * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--tenant", default="tenant-a")
    ap.add_argument("--model", default="base-model")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="Say hello in one sentence.")
    ap.add_argument(
        "--variable-prompt",
        action="store_true",
        help="cycle short/medium/long prompts to vary per-request token cost",
    )
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument(
        "--out",
        default=str(DEFAULT_RESULTS),
        help="append a JSONL result record to this path",
    )
    ap.add_argument(
        "--no-write",
        action="store_true",
        help="do not write a result record, only print the summary",
    )
    args = ap.parse_args()

    latencies: list[float] = []
    statuses: list[int] = []

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(send, args, i) for i in range(args.requests)]
        for f in concurrent.futures.as_completed(futs):
            dt, status = f.result()
            latencies.append(dt)
            statuses.append(status)
    wall = time.time() - t0

    ok = sum(1 for s in statuses if s == 200)
    latencies.sort()

    summary = {
        "ts": time.time(),
        "url": args.url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "tenant": args.tenant,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "variable_prompt": args.variable_prompt,
        "ok": ok,
        "failed": len(statuses) - ok,
        "wall_s": round(wall, 3),
        "throughput_rps": round(args.requests / wall, 2) if wall > 0 else 0.0,
        "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "latency_p50_s": round(percentile(latencies, 0.50), 3),
        "latency_p95_s": round(percentile(latencies, 0.95), 3),
        "latency_p99_s": round(percentile(latencies, 0.99), 3),
    }
    print(json.dumps(summary, indent=2))

    if not args.no_write:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
