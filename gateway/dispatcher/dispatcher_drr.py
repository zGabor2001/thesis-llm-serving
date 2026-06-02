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
class TenantState:
    tenant: str
    quantum: int
    deficit: int = 0
    queue: deque = field(default_factory=deque)


class DRRDispatcher:
    def __init__(self, backend_client, tenant_quanta: dict[str, int] | None = None):
        self.backend_client = backend_client
        self.tenant_quanta = tenant_quanta or {
            "tenant-a": 16,
            "tenant-b": 16,
            "tenant-c": 16,
        }
        self.tenants: dict[str, TenantState] = {
            tenant: TenantState(tenant=tenant, quantum=quantum)
            for tenant, quantum in self.tenant_quanta.items()
        }
        self._rr_order = list(self.tenants.keys())
        self._rr_index = 0

    def _ensure_tenant(self, tenant: str) -> None:
        if tenant not in self.tenants:
            quantum = self.tenant_quanta.get(tenant, 16)
            self.tenants[tenant] = TenantState(tenant=tenant, quantum=quantum)
            self._rr_order.append(tenant)

    def enqueue(self, request: InferenceRequest) -> None:
        self._ensure_tenant(request.tenant)
        self.tenants[request.tenant].queue.append(request)

    def _request_cost(self, request: InferenceRequest) -> int:
        if request.prompt_tokens is not None:
            return max(1, request.prompt_tokens)
        return 1

    def _pick_next_request(self) -> InferenceRequest:
        if not any(state.queue for state in self.tenants.values()):
            raise RuntimeError("No queued requests available")

        checked = 0
        while checked < max(1, len(self._rr_order) * 4):
            tenant = self._rr_order[self._rr_index]
            state = self.tenants[tenant]

            if state.queue:
                req = state.queue[0]
                cost = self._request_cost(req)
                state.deficit += state.quantum

                if state.deficit >= cost:
                    state.deficit -= cost
                    self._rr_index = (self._rr_index + 1) % len(self._rr_order)
                    return state.queue.popleft()

            self._rr_index = (self._rr_index + 1) % len(self._rr_order)
            checked += 1

        for tenant in self._rr_order:
            state = self.tenants[tenant]
            if state.queue:
                req = state.queue[0]
                cost = self._request_cost(req)
                while state.deficit < cost:
                    state.deficit += state.quantum
                state.deficit -= cost
                return state.queue.popleft()

        raise RuntimeError("Failed to select request in DRR dispatcher")

    def dispatch(self, request: InferenceRequest) -> DispatchResult:
        self.enqueue(request)
        req = self._pick_next_request()

        dispatch_ts = time.time()
        result = self.backend_client.complete(req)
        return DispatchResult(result=result, dispatch_ts=dispatch_ts)
