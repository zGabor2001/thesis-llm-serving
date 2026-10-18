"""Live-vLLM integration tests for the gateway.

These require a running vLLM server (see `make up-vllm`) and are excluded from
the default `make test` run. Run with:  make test-integration
"""

import os

import pytest
from fastapi.testclient import TestClient

from gateway.config import GatewayConfig
from gateway.proxy import build_app

pytestmark = pytest.mark.integration

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")


def _make_app():
    cfg = GatewayConfig(backend_mode="vllm", vllm_base_url=VLLM_BASE_URL)
    return build_app(config=cfg)


def test_gateway_health_against_live_vllm():
    client = TestClient(_make_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["backend_ok"] is True


def test_gateway_models_against_live_vllm():
    client = TestClient(_make_app())
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "base-model" in ids
    assert "tenant-a" in ids


def test_gateway_chat_completion_base_model():
    client = TestClient(_make_app())
    payload = {
        "model": "base-model",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"].strip()
    assert body["usage"]["completion_tokens"] > 0


def test_gateway_routes_tenant_to_lora():
    client = TestClient(_make_app())
    payload = {
        "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "tenant": "tenant-a",
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "tenant-a"
    assert body["choices"][0]["message"]["content"].strip()
