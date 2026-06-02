from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from gateway.models import InferenceRequest, RequestResult


@dataclass
class DispatchResult:
    result: RequestResult
    dispatch_ts: float


@dataclass
class TenantQueue:
    tenant: str
    weight: int
    queue: deque = field(default_factory=deque)
    deficit: int = 0


class WRRDispatcher:
    def __init__(self, backend_client, tenant_weights: dict[str, int] | None = None):
        self.backend_client = backend_client
        self.tenant_weights = tenant_weights or {
            "tenant-a": 1,
            "tenant-b": 1,
            "tenant-c": 1,
            "base-model": 1,
        }
        self.queues: dict[str, TenantQueue] = {}
        for tenant, weight in self.tenant_weights.items():
            self.queues[tenant] = TenantQueue(tenant=tenant, weight=weight)

    def enqueue(self, request: InferenceRequest) -> None:
        tenant = request.tenant
        if tenant not in self.queues:
            weight = self.tenant_weights.get(tenant, 1)
            self.queues[tenant] = TenantQueue(tenant=tenant, weight=weight)
        self.queues[tenant].queue.append(request)

    def _next_tenant(self) -> str | None:
        active_tenants = [t for t, q in self.queues.items() if len(q.queue) > 0]
        if not active_tenants:
            return None

        while True:
            for tenant in active_tenants:
                q = self.queues[tenant]
                if len(q.queue) > 0:
                    return tenant

    def dispatch(self, request: InferenceRequest) -> DispatchResult:
        self.enqueue(request)
        tenant = self._next_tenant()
        if tenant is None:
            raise RuntimeError("No tenant available, but queue was non-empty")

        q = self.queues[tenant]
        req = q.queue.popleft()

        dispatch_ts = time.time()
        result = self.backend_client.complete(req)
        return DispatchResult(result=result, dispatch_ts=dispatch_ts)