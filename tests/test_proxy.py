import json

from fastapi.testclient import TestClient

from gateway.proxy import build_app
from gateway.request_logger import RequestLogger
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


def test_chat_completions_endpoint(tmp_path):
    log_file = tmp_path / "requests.jsonl"

    app = build_app(
        backend_client=MockBackendClient(fixed_delay_s=0.0),
        logger=RequestLogger(log_file),
    )
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
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert len(body["choices"][0]["message"]["content"]) > 0

    assert log_file.exists()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2

    events = [json.loads(line) for line in lines]

    received_event = next(event for event in events if event["event_type"] == "request_received")
    completed_event = next(event for event in events if event["event_type"] == "request_completed")

    assert received_event["tenant"] == "tenant-a"
    assert received_event["model"] == "base-model"

    assert completed_event["tenant"] == "tenant-a"
    assert completed_event["model"] == "base-model"
    assert completed_event["status"] == "completed"
    assert "dispatch_ts" in completed_event
    assert completed_event["dispatch_ts"] >= completed_event["arrival_ts"]
    assert completed_event["finish_ts"] >= completed_event["dispatch_ts"]


def test_chat_completions_uses_header_tenant(tmp_path):
    log_file = tmp_path / "requests.jsonl"

    app = build_app(
        backend_client=MockBackendClient(fixed_delay_s=0.0),
        logger=RequestLogger(log_file),
    )
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

    assert log_file.exists()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]

    received_event = next(event for event in events if event["event_type"] == "request_received")
    completed_event = next(event for event in events if event["event_type"] == "request_completed")

    assert received_event["tenant"] == "tenant-from-header"
    assert completed_event["tenant"] == "tenant-from-header"


def test_chat_completions_endpoint_with_wrr_policy(tmp_path):
    import json

    log_file = tmp_path / "requests.jsonl"

    app = build_app(
        backend_client=MockBackendClient(fixed_delay_s=0.0),
        logger=RequestLogger(log_file),
        scheduler_policy="wrr",
        tenant_weights={"tenant-a": 2, "tenant-b": 1},
    )
    client = TestClient(app)

    payload = {
        "model": "base-model",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 32,
        "temperature": 0.0,
        "tenant": "tenant-a",
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["tenant"] == "tenant-a"

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    completed_event = next(event for event in events if event["event_type"] == "request_completed")

    assert completed_event["tenant"] == "tenant-a"
    assert "dispatch_ts" in completed_event