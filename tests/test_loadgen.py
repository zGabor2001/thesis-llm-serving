"""Unit tests for the load generator's workload helpers."""

import importlib.util
from pathlib import Path

import pytest

_LOADGEN_PATH = Path(__file__).resolve().parent.parent / "scripts" / "loadgen.py"


def _load_loadgen():
    spec = importlib.util.spec_from_file_location("loadgen", _LOADGEN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loadgen = _load_loadgen()


def test_parse_tenant_mix_weighted_and_unweighted():
    assert loadgen.parse_tenant_mix("a:2,b:1") == [("a", 2), ("b", 1)]
    assert loadgen.parse_tenant_mix("a,b,c") == [("a", 1), ("b", 1), ("c", 1)]


def test_parse_tenant_mix_rejects_zero_weight():
    with pytest.raises(ValueError, match="weight"):
        loadgen.parse_tenant_mix("a:0")


def test_parse_tenant_mix_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        loadgen.parse_tenant_mix("  ,  ")


def test_build_schedule_expands_weights():
    assert loadgen.build_schedule([("a", 2), ("b", 1)]) == ["a", "a", "b"]


def test_percentile_handles_empty_and_median():
    assert loadgen.percentile([], 0.5) == 0.0
    assert loadgen.percentile([10.0, 20.0, 30.0, 40.0, 50.0], 0.5) == 30.0
