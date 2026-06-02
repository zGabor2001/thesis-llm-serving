import pytest
import json

from fastapi.testclient import TestClient

from gateway.models import InferenceRequest
from gateway.vllm_client import MockBackendClient
from gateway.proxy import build_app
from gateway.request_logger import RequestLogger
from gateway.dispatcher.dispatcher import FIFODispatcher
from gateway.dispatcher.dispatcher_wrr import WRRDispatcher
from gateway.dispatcher.dispatcher_drr import DRRDispatcher
from gateway.dispatcher.dispatcher_factory import build_dispatcher


def test_build_dispatcher_fifo_returns_fifo_dispatcher():
    backend = MockBackendClient(fixed_delay_s=0.0)
    dispatcher = build_dispatcher(backend, policy="fifo")
    assert isinstance(dispatcher, FIFODispatcher)


def test_build_dispatcher_wrr_returns_wrr_dispatcher():
    backend = MockBackendClient(fixed_delay_s=0.0)
    dispatcher = build_dispatcher(
        backend,
        policy="wrr",
        tenant_weights={"tenant-a": 2, "tenant-b": 1},
    )
    assert isinstance(dispatcher, WRRDispatcher)


def test_build_dispatcher_unknown_policy_raises():
    backend = MockBackendClient(fixed_delay_s=0.0)

    with pytest.raises(ValueError, match="Unsupported dispatcher policy"):
        build_dispatcher(backend, policy="unknown")


def test_fifo_dispatcher_returns_result_and_dispatch_ts():
    backend = MockBackendClient(fixed_delay_s=0.0)
    dispatcher = FIFODispatcher(backend)

    req = InferenceRequest(
        request_id="req-1",
        tenant="tenant-a",
        model="base-model",
        arrival_ts=0.0,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=16,
        adapter=None,
        prompt_text="Hello",
        prompt_tokens=5,
        temperature=0.0,
    )

    out = dispatcher.dispatch(req)

    assert out.result is not None
    assert out.result.request_id == "req-1"
    assert isinstance(out.dispatch_ts, float)
    assert out.dispatch_ts > 0


def test_wrr_dispatcher_enqueues_and_dispatches():
    backend = MockBackendClient(fixed_delay_s=0.0)
    dispatcher = WRRDispatcher(backend, tenant_weights={"tenant-a": 1, "tenant-b": 1})

    out = dispatcher.dispatch(_req("req-1", "tenant-a"))

    assert out.result is not None
    assert out.result.request_id == "req-1"
    assert out.dispatch_ts > 0


def test_wrr_dispatcher_respects_weights_structure():
    backend = MockBackendClient(fixed_delay_s=0.0)
    dispatcher = WRRDispatcher(backend, tenant_weights={"tenant-a": 2, "tenant-b": 1})

    for t in ["tenant-a", "tenant-b"]:
        req = InferenceRequest(
            request_id="req-t1",
            tenant=t,
            model="base-model",
            arrival_ts=0.0,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
            adapter=None,
            prompt_text="Hello",
            prompt_tokens=5,
            temperature=0.0,
        )
        out = dispatcher.dispatch(req)
        assert out.result is not None


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

    assert log_file.exists()

    import json
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]

    completed_event = next(event for event in events if event["event_type"] == "request_completed")

    assert completed_event["tenant"] == "tenant-a"
    assert completed_event["model"] == "base-model"
    assert "dispatch_ts" in completed_event


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


def _req(request_id: str, tenant: str, prompt_tokens: int = 8):
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


def test_wrr_dispatcher_handles_multiple_tenants():
    backend = MockBackendClient(fixed_delay_s=0.0)
    dispatcher = WRRDispatcher(backend, tenant_weights={"tenant-a": 2, "tenant-b": 1})

    out1 = dispatcher.dispatch(_req("req-a1", "tenant-a"))
    out2 = dispatcher.dispatch(_req("req-b1", "tenant-b"))
    out3 = dispatcher.dispatch(_req("req-a2", "tenant-a"))

    ids = {out1.result.request_id, out2.result.request_id, out3.result.request_id}
    assert ids == {"req-a1", "req-b1", "req-a2"}


def test_drr_dispatcher_dispatches_single_request():
    backend = MockBackendClient(fixed_delay_s=0.0)
    dispatcher = DRRDispatcher(backend, tenant_quanta={"tenant-a": 8})

    out = dispatcher.dispatch(_req("req-a1", "tenant-a", prompt_tokens=4))

    assert out.result.request_id == "req-a1"
    assert out.dispatch_ts > 0


def test_drr_dispatcher_handles_multiple_tenants_with_different_costs():
    backend = MockBackendClient(fixed_delay_s=0.0)
    dispatcher = DRRDispatcher(
        backend,
        tenant_quanta={"tenant-a": 8, "tenant-b": 8},
    )

    out1 = dispatcher.dispatch(_req("req-a1", "tenant-a", prompt_tokens=4))
    out2 = dispatcher.dispatch(_req("req-b1", "tenant-b", prompt_tokens=12))
    out3 = dispatcher.dispatch(_req("req-a2", "tenant-a", prompt_tokens=4))

    ids = {out1.result.request_id, out2.result.request_id, out3.result.request_id}
    assert ids == {"req-a1", "req-b1", "req-a2"}