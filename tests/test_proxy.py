from fastapi.testclient import TestClient

from gateway.proxy import build_app
from gateway.vllm_client import MockBackendClient


def test_health_endpoint():
    app = build_app(backend_client=MockBackendClient(fixed_delay_s=0.0))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend_ok"] is True


def test_models_endpoint():
    app = build_app(backend_client=MockBackendClient(fixed_delay_s=0.0))
    client = TestClient(app)

    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) >= 1
    assert "id" in body["data"][0]


def test_chat_completions_endpoint():
    app = build_app(backend_client=MockBackendClient(fixed_delay_s=0.0))
    client = TestClient(app)

    payload = {
        "model": "base-model",
        "messages": [
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        "max_tokens": 32,
        "temperature": 0.0,
        "tenant": "tenant-a",
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "base-model"
    assert body["tenant"] == "tenant-a"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)


def test_chat_completions_uses_header_tenant():
    app = build_app(backend_client=MockBackendClient(fixed_delay_s=0.0))
    client = TestClient(app)

    payload = {
        "model": "base-model",
        "messages": [
            {"role": "user", "content": "Say hello in one sentence."}
        ],
    }

    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Tenant-Id": "tenant-from-header"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tenant"] == "tenant-from-header"