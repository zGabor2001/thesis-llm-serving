#!/usr/bin/env python3
"""Concurrency load generator for the LLM gateway.

Two modes:
  * closed-loop  (default): N requests at fixed concurrency
  * open-loop    (--trace): replay an arrival-driven JSONL trace
      (see scripts/gen-trace.py)

Prints a summary (plus per-tenant breakdown) and appends a structured record to
logs/loadgen.jsonl (override with --out, disable with --no-write).
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

# Short / medium / long prompts (cycled by --variable-prompt or by trace prompt_idx).
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


def parse_tenant_mix(spec: str) -> list[tuple[str, int]]:
    """Parse ``'a:2,b:1'`` into ``[('a', 2), ('b', 1)]``."""
    mix: list[tuple[str, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            tenant, weight = part.split(":", 1)
            weight = int(weight.strip())
        else:
            tenant, weight = part, 1
        if weight < 1:
            raise ValueError(f"weight for tenant '{tenant}' must be >= 1")
        mix.append((tenant.strip(), weight))
    if not mix:
        raise ValueError("empty tenant mix")
    return mix


def build_schedule(mix: list[tuple[str, int]]) -> list[str]:
    """Expand a weighted tenant mix into a deterministic round-robin schedule."""
    schedule: list[str] = []
    for tenant, weight in mix:
        schedule.extend([tenant] * weight)
    return schedule


def load_trace(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def make_prompt(args: argparse.Namespace, i: int) -> str:
    if args.variable_prompt:
        return VARIABLE_PROMPTS[i % len(VARIABLE_PROMPTS)]
    return args.prompt


def send(args: argparse.Namespace, prompt_text: str, tenant: str) -> tuple[float, int, str]:
    payload = {
        "model": args.model or tenant,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": args.max_tokens,
        "tenant": tenant,
    }
    t0 = time.time()
    r = requests.post(
        f"{args.url}/v1/chat/completions", json=payload, timeout=args.timeout
    )
    return time.time() - t0, r.status_code, tenant


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, int(round(p * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def tenant_stats(latencies: list[float], statuses: list[int], wall: float) -> dict:
    ok = sum(1 for s in statuses if s == 200)
    lat = sorted(latencies)
    return {
        "requests": len(statuses),
        "ok": ok,
        "failed": len(statuses) - ok,
        "throughput_rps": round(len(statuses) / wall, 2) if wall > 0 else 0.0,
        "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "latency_p50_s": round(percentile(lat, 0.50), 3),
        "latency_p95_s": round(percentile(lat, 0.95), 3),
        "latency_p99_s": round(percentile(lat, 0.99), 3),
    }


def _run(executor, futs):
    latencies: list[float] = []
    statuses: list[int] = []
    by_tenant_lat: dict[str, list[float]] = {}
    by_tenant_status: dict[str, list[int]] = {}
    for f in concurrent.futures.as_completed(futs):
        dt, status, tenant = f.result()
        latencies.append(dt)
        statuses.append(status)
        by_tenant_lat.setdefault(tenant, []).append(dt)
        by_tenant_status.setdefault(tenant, []).append(status)
    return latencies, statuses, by_tenant_lat, by_tenant_status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="closed-loop concurrency, or max in-flight workers for --trace",
    )
    ap.add_argument("--tenant", default="tenant-a", help="single tenant (ignored with --tenants/--trace)")
    ap.add_argument("--tenants", default=None, help="weighted tenant mix, e.g. 'tenant-a:2,tenant-b:1'")
    ap.add_argument("--trace", default=None, help="JSONL trace to replay (open-loop)")
    ap.add_argument("--model", default=None, help="model name (defaults to the tenant)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="Say hello in one sentence.")
    ap.add_argument("--variable-prompt", action="store_true", help="cycle short/medium/long prompts")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--out", default=str(DEFAULT_RESULTS), help="append a JSONL result record here")
    ap.add_argument("--no-write", action="store_true", help="do not write a result record")
    args = ap.parse_args()

    t0 = time.time()

    if args.trace:
        trace = load_trace(args.trace)
        tenants = {
            t: sum(1 for r in trace if r.get("tenant", "tenant-a") == t)
            for t in sorted({r.get("tenant", "tenant-a") for r in trace})
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = []
            for rec in trace:
                target = t0 + rec.get("arrival_s", 0.0)
                delay = target - time.time()
                if delay > 0:
                    time.sleep(delay)
                tenant = rec.get("tenant", "tenant-a")
                prompt_idx = rec.get("prompt_idx", 0)
                prompt_text = VARIABLE_PROMPTS[prompt_idx % len(VARIABLE_PROMPTS)]
                futs.append(ex.submit(send, args, prompt_text, tenant))
            latencies, statuses, by_tenant_lat, by_tenant_status = _run(ex, futs)
    else:
        if args.tenants:
            mix = parse_tenant_mix(args.tenants)
        else:
            mix = [(args.tenant, 1)]
        tenants = dict(mix)
        schedule = build_schedule(mix)
        tenant_of = [schedule[i % len(schedule)] for i in range(args.requests)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(send, args, make_prompt(args, i), tenant_of[i]) for i in range(args.requests)]
            latencies, statuses, by_tenant_lat, by_tenant_status = _run(ex, futs)

    wall = time.time() - t0
    ok = sum(1 for s in statuses if s == 200)
    latencies.sort()

    summary = {
        "ts": time.time(),
        "url": args.url,
        "trace": args.trace,
        "requests": len(latencies),
        "concurrency": args.concurrency,
        "tenants": tenants,
        "model": args.model or "auto",
        "max_tokens": args.max_tokens,
        "variable_prompt": args.variable_prompt or bool(args.trace),
        "ok": ok,
        "failed": len(statuses) - ok,
        "wall_s": round(wall, 3),
        "throughput_rps": round(len(latencies) / wall, 2) if wall > 0 else 0.0,
        "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "latency_p50_s": round(percentile(latencies, 0.50), 3),
        "latency_p95_s": round(percentile(latencies, 0.95), 3),
        "latency_p99_s": round(percentile(latencies, 0.99), 3),
        "per_tenant": {
            t: tenant_stats(by_tenant_lat.get(t, []), by_tenant_status.get(t, []), wall)
            for t in tenants
        },
    }
    if not args.tenants and not args.trace:
        summary["tenant"] = args.tenant

    print(json.dumps(summary, indent=2))

    if not args.no_write:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
