from __future__ import annotations

from gateway.dispatcher.async_dispatcher import AsyncDispatcher
from gateway.dispatcher.dispatcher import FIFODispatcher
from gateway.dispatcher.dispatcher_drr import DRRDispatcher
from gateway.dispatcher.dispatcher_wrr import WRRDispatcher


def build_dispatcher(
    backend_client,
    policy: str = "fifo",
    tenant_weights: dict[str, int] | None = None,
    tenant_quanta: dict[str, int] | None = None,
    max_concurrency: int = 32,
):
    policy = policy.lower()

    if policy == "fifo":
        selector = FIFODispatcher()
    elif policy == "wrr":
        selector = WRRDispatcher(tenant_weights=tenant_weights)
    elif policy == "drr":
        selector = DRRDispatcher(tenant_quanta=tenant_quanta)
    else:
        raise ValueError(f"Unsupported dispatcher policy: {policy}")

    return AsyncDispatcher(selector, backend_client, max_concurrency=max_concurrency)
