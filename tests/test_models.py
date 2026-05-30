import pytest

from gateway.models import InferenceRequest, RequestResult


def test_inference_request_creation():
    req = InferenceRequest(
        request_id="r1",
        tenant="tenant-a",
        model="base-model",
        arrival_ts=1.0,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=32,
    )

    assert req.request_id == "r1"
    assert req.tenant == "tenant-a"
    assert req.model == "base-model"
    assert req.max_tokens == 32
    assert req.temperature == 0.0
    assert req.priority_class == "standard"


def test_inference_request_to_dict():
    req = InferenceRequest(
        request_id="r2",
        tenant="tenant-b",
        model="tenant-b",
        arrival_ts=2.0,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=16,
    )

    data = req.to_dict()

    assert data["request_id"] == "r2"
    assert data["tenant"] == "tenant-b"
    assert data["model"] == "tenant-b"
    assert data["max_tokens"] == 16


def test_request_result_metrics():
    result = RequestResult(
        request_id="r3",
        tenant="tenant-c",
        model="tenant-c",
        arrival_ts=10.0,
        dispatch_ts=12.0,
        finish_ts=18.0,
        first_token_ts=14.0,
        status="completed",
    )

    assert result.queue_delay == 2.0
    assert result.latency == 8.0
    assert result.service_time == 6.0
    assert result.ttft == 4.0


def test_request_result_ttft_none_when_missing():
    result = RequestResult(
        request_id="r4",
        tenant="tenant-a",
        model="base-model",
        arrival_ts=10.0,
        dispatch_ts=11.0,
        finish_ts=15.0,
        first_token_ts=None,
        status="failed",
    )

    assert result.ttft is None