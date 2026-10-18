import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from gateway.config import GatewayConfig
from gateway.dispatcher.async_dispatcher import AsyncDispatcher
from gateway.dispatcher.dispatcher import FIFODispatcher
from gateway.dispatcher.dispatcher_drr import DRRDispatcher
from gateway.dispatcher.dispatcher_factory import build_dispatcher
from gateway.dispatcher.dispatcher_wrr import WRRDispatcher
from gateway.models import InferenceRequest
from gateway.proxy import build_app
from gateway.request_logger import RequestLogger
from gateway.vllm_client import MockBackendClient


def _req(request_id: str, tenant: str, prompt_tokens: int = 8) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        tenant=tenant,
        model="base-model",
        arrival_ts=0.0,
        messages=[{"role": "user", "content": f"Hello from {tenant}"}],
        max_tokens=16,
        adapter=None,
        prompt_text=f"Hello from {tenant}",
        prompt_tokens=prompt_tokens,
        temperature=0.0,
    )


def test_build_dispatcher_fifo_returns_async_with_fifo_policy():
    dispatcher = build_dispatcher(MockBackendClient(fixed_delay_s=0.0), policy="fifo")
    assert isinstance(dispatcher, AsyncDispatcher)
    assert isinstance(dispatcher.policy, FIFODispatcher)


def test_build_dispatcher_wrr_returns_async_with_wrr_policy():
    dispatcher = build_dispatcher(
        MockBackendClient(fixed_delay_s=0.0),
        policy="wrr",
        tenant_weights={"tenant-a": 2, "tenant-b": 1},
    )
    assert isinstance(dispatcher, AsyncDispatcher)
    assert isinstance(dispatcher.policy, WRRDispatcher)


def test_build_dispatcher_drr_returns_async_with_drr_policy():
    dispatcher = build_dispatcher(
        MockBackendClient(fixed_delay_s=0.0),
        policy="drr",
        tenant_quanta={"tenant-a": 8, "tenant-b": 8},
    )
    assert isinstance(dispatcher, AsyncDispatcher)
    assert isinstance(dispatcher.policy, DRRDispatcher)


def test_build_dispatcher_unknown_policy_raises():
    with pytest.raises(ValueError, match="Unsupported dispatcher policy"):
        build_dispatcher(MockBackendClient(fixed_delay_s=0.0), policy="unknown")


def test_async_dispatch_returns_result_and_dispatch_ts():
    dispatcher = build_dispatcher(MockBackendClient(fixed_delay_s=0.0), policy="fifo")

    out = dispatcher.dispatch(_req("req-1", "tenant-a"), timeout=5)

    assert out.result is not None
    assert out.result.request_id == "req-1"
    assert isinstance(out.dispatch_ts, float)
    assert out.dispatch_ts > 0


def test_build_app_uses_scheduler_mode_from_config():
    # Regression: SCHEDULER_MODE must actually select the dispatcher policy.
    cfg = GatewayConfig(scheduler_mode="drr")
    app = build_app(config=cfg, backend_client=MockBackendClient(fixed_delay_s=0.0))
    assert isinstance(app.state.dispatcher.policy, DRRDispatcher)


def test_build_app_uses_tenant_weights_from_config():
    cfg = GatewayConfig(scheduler_mode="wrr", tenant_weights={"tenant-a": 3, "tenant-b": 1})
    app = build_app(config=cfg, backend_client=MockBackendClient(fixed_delay_s=0.0))
    assert app.state.dispatcher.policy.tenant_weights == {"tenant-a": 3, "tenant-b": 1}


def test_build_app_uses_tenant_quanta_from_config():
    cfg = GatewayConfig(scheduler_mode="drr", tenant_quanta={"tenant-a": 8, "tenant-b": 16})
    app = build_app(config=cfg, backend_client=MockBackendClient(fixed_delay_s=0.0))
    assert app.state.dispatcher.policy.tenant_quanta == {"tenant-a": 8, "tenant-b": 16}


def test_tenant_routes_to_lora_adapter(tmp_path):
    cfg = GatewayConfig(lora_tenants=["tenant-a"])
    app = build_app(
        config=cfg,
        backend_client=MockBackendClient(fixed_delay_s=0.0),
        logger=RequestLogger(tmp_path / "requests.jsonl"),
    )
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "tenant": "tenant-a",
        },
    )

    assert r.status_code == 200
    assert r.json()["model"] == "tenant-a"


def test_gateway_logs_cold_and_switch(tmp_path):
    log_file = tmp_path / "requests.jsonl"
    app = build_app(
        backend_client=MockBackendClient(fixed_delay_s=0.0),
        logger=RequestLogger(log_file),
        scheduler_policy="fifo",
    )
    client = TestClient(app)

    for tenant in ["tenant-a", "tenant-b", "tenant-a"]:
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "tenant": tenant,
            },
        )
        assert r.status_code == 200

    events = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    completed = [e for e in events if e["event_type"] == "request_completed"]

    assert [e["cold"] for e in completed] == [True, True, False]
    assert [e["switch"] for e in completed] == [False, True, True]


def test_concurrent_dispatch_returns_each_caller_its_own_result():
    # Regression: a caller must never receive another request's result.
    dispatcher = build_dispatcher(
        MockBackendClient(fixed_delay_s=0.005),
        policy="fifo",
        max_concurrency=8,
    )
    requests = [_req(f"req-{i}", "tenant-a") for i in range(24)]

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(
            ex.map(lambda r: dispatcher.dispatch(r, timeout=5).result.request_id, requests)
        )

    assert sorted(results) == [r.request_id for r in sorted(requests, key=lambda r: r.request_id)]


def test_chat_completions_endpoint_with_wrr_policy(tmp_path):
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
    assert body["object"] == "chat.completion"
    assert body["model"] == "base-model"
    assert body["tenant"] == "tenant-a"

    assert log_file.exists()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]

    completed_event = next(event for event in events if event["event_type"] == "request_completed")

    assert completed_event["tenant"] == "tenant-a"
    assert completed_event["model"] == "base-model"
    assert "dispatch_ts" in completed_event
    assert "first_token_ts" in completed_event


def test_wrr_policy_handles_multiple_tenants(tmp_path):
    log_file = tmp_path / "requests.jsonl"

    app = build_app(
        backend_client=MockBackendClient(fixed_delay_s=0.0),
        logger=RequestLogger(log_file),
        scheduler_policy="wrr",
        tenant_weights={"tenant-a": 2, "tenant-b": 1},
    )
    client = TestClient(app)

    payload_a = {
        "model": "base-model",
        "messages": [{"role": "user", "content": "Hello from A"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "tenant": "tenant-a",
    }
    payload_b = {
        "model": "base-model",
        "messages": [{"role": "user", "content": "Hello from B"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "tenant": "tenant-b",
    }

    r1 = client.post("/v1/chat/completions", json=payload_a)
    r2 = client.post("/v1/chat/completions", json=payload_b)
    r3 = client.post("/v1/chat/completions", json=payload_a)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 200

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]

    completed_events = [e for e in events if e["event_type"] == "request_completed"]
    assert len(completed_events) == 3

    tenants = [e["tenant"] for e in completed_events]
    assert "tenant-a" in tenants
    assert "tenant-b" in tenants

    for event in completed_events:
        assert "dispatch_ts" in event
        assert event["dispatch_ts"] >= event["arrival_ts"]
        assert event["finish_ts"] >= event["dispatch_ts"]
