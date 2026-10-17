"""Ordering tests proving FIFO, WRR, and DRR implement distinct policies."""

from gateway.dispatcher.dispatcher import FIFODispatcher
from gateway.dispatcher.dispatcher_drr import DRRDispatcher
from gateway.dispatcher.dispatcher_wrr import WRRDispatcher
from gateway.models import InferenceRequest


def _req(request_id: str, tenant: str, prompt_tokens: int = 4) -> InferenceRequest:
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


def _serve(dispatcher, requests: list[InferenceRequest]) -> list[str]:
    for request in requests:
        dispatcher.enqueue(request)
    return [dispatcher.select_next().tenant for _ in requests]


def test_fifo_serves_in_arrival_order():
    dispatcher = FIFODispatcher()
    requests = [_req("r1", "tenant-a"), _req("r2", "tenant-b"), _req("r3", "tenant-a")]

    assert _serve(dispatcher, requests) == ["tenant-a", "tenant-b", "tenant-a"]


def test_wrr_serves_in_proportion_to_weights():
    dispatcher = WRRDispatcher(tenant_weights={"tenant-a": 2, "tenant-b": 1})
    requests = [
        _req("a1", "tenant-a"),
        _req("b1", "tenant-b"),
        _req("a2", "tenant-a"),
        _req("b2", "tenant-b"),
        _req("a3", "tenant-a"),
        _req("a4", "tenant-a"),
    ]

    order = _serve(dispatcher, requests)

    assert order.count("tenant-a") == 4
    assert order.count("tenant-b") == 2
    assert order == ["tenant-a", "tenant-b", "tenant-a", "tenant-a", "tenant-b", "tenant-a"]


def test_drr_serves_small_request_before_large_one():
    dispatcher = DRRDispatcher(tenant_quanta={"tenant-a": 8, "tenant-b": 8})
    requests = [
        _req("a1", "tenant-a", prompt_tokens=4),
        _req("b1", "tenant-b", prompt_tokens=12),
        _req("a2", "tenant-a", prompt_tokens=4),
    ]

    # b1 costs more than one quantum, so a2 overtakes it.
    assert _serve(dispatcher, requests) == ["tenant-a", "tenant-a", "tenant-b"]


def test_three_policies_produce_distinct_orders():
    requests = [
        _req("a1", "tenant-a", prompt_tokens=4),
        _req("b1", "tenant-b", prompt_tokens=12),
        _req("a2", "tenant-a", prompt_tokens=4),
        _req("b2", "tenant-b", prompt_tokens=4),
        _req("a3", "tenant-a", prompt_tokens=4),
        _req("b3", "tenant-b", prompt_tokens=4),
    ]

    fifo = _serve(FIFODispatcher(), requests)
    wrr = _serve(WRRDispatcher(tenant_weights={"tenant-a": 2, "tenant-b": 1}), requests)
    drr = _serve(DRRDispatcher(tenant_quanta={"tenant-a": 8, "tenant-b": 8}), requests)

    assert fifo == ["tenant-a", "tenant-b", "tenant-a", "tenant-b", "tenant-a", "tenant-b"]
    assert wrr == ["tenant-a", "tenant-b", "tenant-a", "tenant-a", "tenant-b", "tenant-b"]
    assert drr == ["tenant-a", "tenant-a", "tenant-b", "tenant-a", "tenant-b", "tenant-b"]

    assert len({tuple(fifo), tuple(wrr), tuple(drr)}) == 3
