"""Tests for the log analysis script."""

import importlib.util
import json
from pathlib import Path

import pytest

_ANALYZE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "analyze.py"


def _load_analyze():
    spec = importlib.util.spec_from_file_location("analyze", _ANALYZE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze_mod = _load_analyze()


def test_jain_fairness():
    assert analyze_mod.jain_fairness([1, 1, 1]) == 1.0
    assert analyze_mod.jain_fairness([1, 0]) == 0.5
    assert analyze_mod.jain_fairness([24, 12]) == pytest.approx(0.9)


def test_analyze_joins_and_aggregates(tmp_path):
    log = tmp_path / "requests.jsonl"
    events = [
        {
            "event_type": "request_received",
            "request_id": "a1",
            "tenant": "tenant-a",
            "scheduler": "fifo",
            "model": "base-model",
            "arrival_ts": 0.0,
        },
        {
            "event_type": "request_completed",
            "request_id": "a1",
            "tenant": "tenant-a",
            "scheduler": "fifo",
            "model": "base-model",
            "arrival_ts": 0.0,
            "dispatch_ts": 0.001,
            "finish_ts": 0.2,
            "first_token_ts": 0.05,
            "status": "completed",
            "prompt_tokens": 6,
            "generated_tokens": 10,
        },
        {
            "event_type": "request_received",
            "request_id": "b1",
            "tenant": "tenant-b",
            "scheduler": "fifo",
            "model": "base-model",
            "arrival_ts": 0.0,
        },
        {
            "event_type": "request_completed",
            "request_id": "b1",
            "tenant": "tenant-b",
            "scheduler": "fifo",
            "model": "base-model",
            "arrival_ts": 0.0,
            "dispatch_ts": 0.001,
            "finish_ts": 0.2,
            "first_token_ts": 0.05,
            "status": "completed",
            "prompt_tokens": 6,
            "generated_tokens": 10,
        },
    ]
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    report = analyze_mod.analyze(str(log))

    assert report["events"] == {"received": 2, "completed": 2}

    fifo = report["schedulers"]["fifo"]
    assert fifo["requests"] == 2
    assert fifo["ok"] == 2
    assert fifo["generated_tokens"] == 20
    assert fifo["jain_generated_tokens"] == 1.0
    assert fifo["per_tenant"]["tenant-a"]["requests"] == 1
    assert fifo["per_tenant"]["tenant-b"]["requests"] == 1
    assert fifo["per_tenant"]["tenant-a"]["latency"]["p50_s"] > 0.0


def _metric(model, dispatch_ts, finish_ts, latency, ttft):
    return {
        "model": model,
        "dispatch_ts": dispatch_ts,
        "finish_ts": finish_ts,
        "latency": latency,
        "ttft": ttft,
    }


def test_cold_switch_stats():
    items = [
        _metric("tenant-a", 1.0, 1.2, 0.2, 0.05),
        _metric("tenant-a", 2.0, 2.2, 0.2, 0.05),
        _metric("tenant-b", 3.0, 3.3, 0.3, 0.08),
        _metric("tenant-a", 4.0, 4.2, 0.2, 0.05),
    ]

    cs = analyze_mod.cold_switch_stats(items)

    assert cs["cold_count"] == 2  # first tenant-a + first tenant-b
    assert cs["switch_count"] == 2  # a->b and b->a
    assert cs["switch_gap_mean_s"] == pytest.approx(0.75)  # (3.0-2.2) and (4.0-3.3)
