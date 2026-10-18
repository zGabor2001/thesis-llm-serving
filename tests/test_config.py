"""Tests for gateway config parsing (tenant weights and quanta)."""

import pytest

from gateway.config import GatewayConfig, parse_weight_map


def test_parse_weight_map_weighted_and_unweighted():
    assert parse_weight_map("a:2,b:1") == {"a": 2, "b": 1}
    assert parse_weight_map("a,b,c") == {"a": 1, "b": 1, "c": 1}


def test_parse_weight_map_rejects_bad_input():
    with pytest.raises(ValueError, match=">= 1"):
        parse_weight_map("a:0")
    with pytest.raises(ValueError, match="empty"):
        parse_weight_map("  ,  ")


def test_from_env_parses_tenant_weights_and_quanta(monkeypatch):
    monkeypatch.setenv("TENANT_WEIGHTS", "tenant-a:2,tenant-b:1")
    monkeypatch.setenv("TENANT_QUANTA", "tenant-a:8,tenant-b:16")

    cfg = GatewayConfig.from_env()

    assert cfg.tenant_weights == {"tenant-a": 2, "tenant-b": 1}
    assert cfg.tenant_quanta == {"tenant-a": 8, "tenant-b": 16}


def test_from_env_defaults_weights_quanta_to_none(monkeypatch):
    monkeypatch.delenv("TENANT_WEIGHTS", raising=False)
    monkeypatch.delenv("TENANT_QUANTA", raising=False)

    cfg = GatewayConfig.from_env()

    assert cfg.tenant_weights is None
    assert cfg.tenant_quanta is None
