from __future__ import annotations

from gateway.dispatcher.dispatcher import FIFODispatcher
from gateway.dispatcher.dispatcher_drr import DRRDispatcher
from gateway.dispatcher.dispatcher_wrr import WRRDispatcher


def build_dispatcher(
    backend_client,
    policy: str = "fifo",
    tenant_weights: dict[str, int] | None = None,
    tenant_quanta: dict[str, int] | None = None,
):
    policy = policy.lower()

    if policy == "fifo":
        return FIFODispatcher(backend_client)

    if policy == "wrr":
        return WRRDispatcher(backend_client, tenant_weights=tenant_weights)

    if policy == "drr":
        return DRRDispatcher(backend_client, tenant_quanta=tenant_quanta)

    raise ValueError(f"Unsupported dispatcher policy: {policy}")
