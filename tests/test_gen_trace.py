"""Tests for the trace generator."""

import importlib.util
from pathlib import Path

_TRACE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "gen-trace.py"


def _load():
    spec = importlib.util.spec_from_file_location("gen_trace", _TRACE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen_trace = _load()


def _rows(trace_type, n=60, scale=0.05):
    return gen_trace.generate(trace_type, n, seed=42, scale=scale, hot="tenant-a", tail="tenant-b")


def test_all_trace_types_generate_valid_rows():
    for trace_type in gen_trace.TRACE_TYPES:
        rows = _rows(trace_type)
        assert len(rows) == 60
        arrivals = [r["arrival_s"] for r in rows]
        assert arrivals == sorted(arrivals), f"{trace_type}: arrivals not monotonic"
        for r in rows:
            assert r["tenant"] in ("tenant-a", "tenant-b")
            assert r["prompt_idx"] in (0, 1, 2)


def test_popularity_shift_has_decreasing_hot_share():
    rows = _rows("popularity_shift", n=90)
    third = 30
    p_hot = [
        sum(1 for r in rows[i * third:(i + 1) * third] if r["tenant"] == "tenant-a") / third
        for i in range(3)
    ]
    assert p_hot[0] > 0.6
    assert p_hot[2] < 0.5
    assert p_hot[0] > p_hot[2]
