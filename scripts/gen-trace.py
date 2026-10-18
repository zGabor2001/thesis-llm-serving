#!/usr/bin/env python3
"""Generate arrival-driven (open-loop) request traces.

Four archetypes mirror the interactive traces used in Taha's baseline thesis:
sparse_tail, bursty_tail, popularity_shift, balanced_mix. Each output line is
JSONL:

    {"arrival_s": <float>, "tenant": <str>, "prompt_idx": <int>}

`arrival_s` is the offset (seconds) from trace start. `tenant` is the "hot"
model's tenant vs. the "tail" model's tenant (stand-ins for e.g. 8B vs. 70B).
`prompt_idx` indexes the short/medium/long prompt pool (0, 1, 2).

Usage:
  python scripts/gen-trace.py --type sparse_tail --requests 200 \
      --out traces/sparse_tail.jsonl
  # quick local run: scale inter-arrival times down
  python scripts/gen-trace.py --type bursty_tail --requests 60 \
      --interarrival-scale 0.05
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TRACE_TYPES = ("sparse_tail", "bursty_tail", "popularity_shift", "balanced_mix")

# Base inter-arrival means in seconds (magnitudes mirroring Taha's traces).
BASE_MEAN = {
    "sparse_tail": 1.95,
    "bursty_bg": 2.07,
    "bursty_burst": 0.27,
    "popularity_shift": 1.95,
    "balanced_mix": 1.0,
}

HOT_DEFAULT = "tenant-a"
TAIL_DEFAULT = "tenant-b"


def _exp(rng, mean: float) -> float:
    return rng.expovariate(1.0 / mean)


def gen_sparse_tail(n, rng, scale, hot, tail):
    rows = []
    t = 0.0
    mean = BASE_MEAN["sparse_tail"] * scale
    for i in range(n):
        rows.append({"arrival_s": round(t, 4), "tenant": hot if rng.random() < 0.8 else tail, "prompt_idx": i % 3})
        t += _exp(rng, mean)
    return rows


def gen_bursty_tail(n, rng, scale, hot, tail):
    rows = []
    t = 0.0
    i = 0
    burst_left = 0
    while i < n:
        if burst_left == 0 and rng.random() < 0.2:
            burst_left = rng.randint(2, 5)  # start a tail burst
        if burst_left > 0:
            tenant = tail
            burst_left -= 1
            mean = BASE_MEAN["bursty_burst"] * scale
        else:
            tenant = hot
            mean = BASE_MEAN["bursty_bg"] * scale
        rows.append({"arrival_s": round(t, 4), "tenant": tenant, "prompt_idx": i % 3})
        t += _exp(rng, mean)
        i += 1
    return rows


def gen_popularity_shift(n, rng, scale, hot, tail):
    rows = []
    t = 0.0
    mean = BASE_MEAN["popularity_shift"] * scale
    third = max(1, n // 3)
    p_hot_by_phase = (0.8, 0.5, 0.3)
    for i in range(n):
        phase = min(2, i // third)
        tenant = hot if rng.random() < p_hot_by_phase[phase] else tail
        rows.append({"arrival_s": round(t, 4), "tenant": tenant, "prompt_idx": i % 3})
        t += _exp(rng, mean)
    return rows


def gen_balanced_mix(n, rng, scale, hot, tail):
    rows = []
    t = 0.0
    mean = BASE_MEAN["balanced_mix"] * scale
    for i in range(n):
        rows.append({"arrival_s": round(t, 4), "tenant": hot if rng.random() < 0.5 else tail, "prompt_idx": i % 3})
        t += _exp(rng, mean)
    return rows


GENERATORS = {
    "sparse_tail": gen_sparse_tail,
    "bursty_tail": gen_bursty_tail,
    "popularity_shift": gen_popularity_shift,
    "balanced_mix": gen_balanced_mix,
}


def generate(trace_type, n, seed, scale, hot, tail):
    rng = random.Random(seed)
    return GENERATORS[trace_type](n, rng, scale, hot, tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=TRACE_TYPES)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--hot", default=HOT_DEFAULT)
    ap.add_argument("--tail", default=TAIL_DEFAULT)
    ap.add_argument("--interarrival-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "traces" / f"{args.type}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = generate(args.type, args.requests, args.seed, args.interarrival_scale, args.hot, args.tail)

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} requests to {out}")


if __name__ == "__main__":
    main()
