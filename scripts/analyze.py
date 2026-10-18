#!/usr/bin/env python3
"""Analyze gateway request logs (logs/requests.jsonl).

Joins request_received / request_completed events, computes per-tenant and
per-scheduler latency, throughput, and fairness metrics, prints a report, and
writes structured JSON.

Usage:
  python scripts/analyze.py [--log logs/requests.jsonl] [--out logs/analysis.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "logs" / "requests.jsonl"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "logs" / "analysis.json"


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, int(round(p * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def jain_fairness(values: list[float]) -> float:
    """Jain's fairness index: (sum x)^2 / (n * sum x^2), in [1/n, 1]."""
    if not values:
        return float("nan")
    n = len(values)
    numerator = sum(values) ** 2
    denominator = n * sum(v * v for v in values)
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def latency_stats(values: list[float]) -> dict:
    if not values:
        return {"mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "p99_s": 0.0}
    s = sorted(values)
    return {
        "mean_s": round(statistics.mean(s), 4),
        "p50_s": round(percentile(s, 0.50), 4),
        "p95_s": round(percentile(s, 0.95), 4),
        "p99_s": round(percentile(s, 0.99), 4),
    }


def cold_switch_stats(items: list[dict]) -> dict:
    """Cold/warm split and adapter/model switch statistics.

    A request is 'cold' if it is the first request (by dispatch time) for its
    model. A 'switch' is a transition between two consecutively dispatched
    requests with different models; switch gap is the finish->dispatch interval.
    """
    ordered = sorted(
        (m for m in items if m["dispatch_ts"] is not None),
        key=lambda m: m["dispatch_ts"],
    )
    seen: set[str] = set()
    cold: list[dict] = []
    warm: list[dict] = []
    switches = 0
    switch_gaps: list[float] = []
    prev: dict | None = None

    for m in ordered:
        key = m["model"]
        if key not in seen:
            seen.add(key)
            cold.append(m)
        else:
            warm.append(m)

        if prev is not None and prev["model"] != m["model"]:
            switches += 1
            if prev["finish_ts"] is not None and m["dispatch_ts"] is not None:
                switch_gaps.append(max(0.0, m["dispatch_ts"] - prev["finish_ts"]))
        prev = m

    return {
        "cold_count": len(cold),
        "switch_count": switches,
        "cold_latency": latency_stats([m["latency"] for m in cold if m["latency"] is not None]),
        "warm_latency": latency_stats([m["latency"] for m in warm if m["latency"] is not None]),
        "cold_ttft": latency_stats([m["ttft"] for m in cold if m["ttft"] is not None]),
        "warm_ttft": latency_stats([m["ttft"] for m in warm if m["ttft"] is not None]),
        "switch_gap_mean_s": round(statistics.mean(switch_gaps), 4) if switch_gaps else 0.0,
    }


def load_events(path: str) -> list[dict]:
    events: list[dict] = []
    if not Path(path).is_file():
        return events
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def join_requests(events: list[dict]) -> tuple[list[dict], int, int]:
    received = {e["request_id"]: e for e in events if e.get("event_type") == "request_received"}
    completed = [e for e in events if e.get("event_type") == "request_completed"]
    joined = [{**received.get(c["request_id"], {}), **c} for c in completed]
    return joined, len(received), len(completed)


def request_metrics(rec: dict) -> dict:
    arrival = rec.get("arrival_ts")
    dispatch = rec.get("dispatch_ts")
    finish = rec.get("finish_ts")
    first = rec.get("first_token_ts")
    prompt = rec.get("prompt_tokens") or 0
    generated = rec.get("generated_tokens") or 0
    status = rec.get("status", "unknown")

    return {
        "tenant": rec.get("tenant", "unknown"),
        "scheduler": rec.get("scheduler", "unknown"),
        "status": status,
        "ok": status in ("completed", "success"),
        "arrival_ts": arrival,
        "finish_ts": finish,
        "dispatch_ts": dispatch,
        "model": rec.get("model", "unknown"),
        "prompt_tokens": prompt,
        "generated_tokens": generated,
        "total_tokens": prompt + generated,
        "latency": (finish - arrival) if (arrival is not None and finish is not None) else None,
        "queue_delay": (dispatch - arrival) if (arrival is not None and dispatch is not None) else None,
        "service_time": (finish - dispatch) if (dispatch is not None and finish is not None) else None,
        "ttft": (first - arrival) if (arrival is not None and first is not None) else None,
    }


def summarize(items: list[dict]) -> dict:
    latencies = [m["latency"] for m in items if m["latency"] is not None]
    ttfts = [m["ttft"] for m in items if m["ttft"] is not None]
    queue_delays = [m["queue_delay"] for m in items if m["queue_delay"] is not None]
    arrivals = [m["arrival_ts"] for m in items if m["arrival_ts"] is not None]
    finishes = [m["finish_ts"] for m in items if m["finish_ts"] is not None]
    window = (max(finishes) - min(arrivals)) if arrivals and finishes else 0.0

    ok = sum(1 for m in items if m["ok"])
    generated = sum(m["generated_tokens"] for m in items)
    prompt = sum(m["prompt_tokens"] for m in items)

    return {
        "requests": len(items),
        "ok": ok,
        "failed": len(items) - ok,
        "window_s": round(window, 3),
        "prompt_tokens": prompt,
        "generated_tokens": generated,
        "throughput_req_s": round(len(items) / window, 3) if window > 0 else 0.0,
        "throughput_tok_s": round(generated / window, 3) if window > 0 else 0.0,
        "latency": latency_stats(latencies),
        "ttft": latency_stats(ttfts),
        "queue_delay": latency_stats(queue_delays),
        "cold_switch": cold_switch_stats(items),
    }


def analyze(path: str, load_time_s: float | None = None) -> dict:
    events = load_events(path)
    joined, n_received, n_completed = join_requests(events)
    metrics = [request_metrics(r) for r in joined]

    by_scheduler: dict[str, list[dict]] = {}
    for m in metrics:
        by_scheduler.setdefault(m["scheduler"], []).append(m)

    schedulers: dict[str, dict] = {}
    for scheduler, items in by_scheduler.items():
        by_tenant: dict[str, list[dict]] = {}
        for m in items:
            by_tenant.setdefault(m["tenant"], []).append(m)

        per_tenant = {t: summarize(ms) for t, ms in sorted(by_tenant.items())}
        report = summarize(items)
        report["per_tenant"] = per_tenant
        report["jain_generated_tokens"] = round(
            jain_fairness([s["generated_tokens"] for s in per_tenant.values()]), 4
        )
        report["jain_tok_per_s"] = round(
            jain_fairness([s["throughput_tok_s"] for s in per_tenant.values()]), 4
        )
        if load_time_s is not None:
            report["cold_ttft_incl_load_mean_s"] = round(
                load_time_s + report["cold_switch"]["cold_ttft"]["mean_s"], 4
            )
        schedulers[scheduler] = report

    return {
        "events": {"received": n_received, "completed": n_completed},
        "schedulers": schedulers,
    }


def _fmt(stats: dict) -> str:
    return (
        f"mean={stats['mean_s']} p50={stats['p50_s']} "
        f"p95={stats['p95_s']} p99={stats['p99_s']}"
    )


def print_report(report: dict) -> None:
    ev = report["events"]
    print(f"Events: received={ev['received']} completed={ev['completed']}\n")

    for scheduler, s in report["schedulers"].items():
        print(f"Scheduler: {scheduler}")
        print(
            f"  requests={s['requests']} ok={s['ok']} failed={s['failed']} "
            f"window={s['window_s']}s"
        )
        print(
            f"  throughput: {s['throughput_req_s']} req/s, "
            f"{s['throughput_tok_s']} tok/s"
        )
        print(
            f"  tokens: prompt={s['prompt_tokens']} generated={s['generated_tokens']}"
        )
        print(f"  latency: {_fmt(s['latency'])}")
        print(f"  ttft:    {_fmt(s['ttft'])}")
        print(f"  queue:   {_fmt(s['queue_delay'])}")
        print(
            f"  Jain fairness: generated_tokens={s['jain_generated_tokens']} "
            f"tok/s={s['jain_tok_per_s']}"
        )
        cs = s["cold_switch"]
        print(
            f"  cold={cs['cold_count']} switches={cs['switch_count']} "
            f"switch_gap_mean={cs['switch_gap_mean_s']}s"
        )
        print(f"  cold ttft: {_fmt(cs['cold_ttft'])}")
        print(f"  warm ttft: {_fmt(cs['warm_ttft'])}")
        if "cold_ttft_incl_load_mean_s" in s:
            print(f"  cold ttft incl. load (mean): {s['cold_ttft_incl_load_mean_s']}s")
        for tenant, ts in sorted(s["per_tenant"].items()):
            print(
                f"    {tenant}: req={ts['requests']} ok={ts['ok']} "
                f"gen_tok={ts['generated_tokens']} tok/s={ts['throughput_tok_s']} "
                f"p50={ts['latency']['p50_s']} p95={ts['latency']['p95_s']}"
            )
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--load-time-s", type=float, default=None, help="model load time (s); adds cold-ttft-incl-load")
    ap.add_argument("--json", action="store_true", help="print the JSON report to stdout")
    ap.add_argument("--no-write", action="store_true", help="do not write the JSON report file")
    args = ap.parse_args()

    load_time_s = args.load_time_s
    if load_time_s is None:
        state = Path(__file__).resolve().parent.parent / "run" / "vllm.load_time_s"
        if state.is_file():
            try:
                load_time_s = float(state.read_text().strip())
            except ValueError:
                load_time_s = None

    report = analyze(args.log, load_time_s=load_time_s)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    if not args.no_write:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
