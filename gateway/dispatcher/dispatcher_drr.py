from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from gateway.models import InferenceRequest


@dataclass
class TenantState:
    tenant: str
    quantum: int
    deficit: int = 0
    queue: deque = field(default_factory=deque)


class DRRDispatcher:
    """Deficit round-robin admission policy with token-aware request costs.

    One full pass over tenants is a round: each tenant's deficit grows by its
    ``quantum`` (a token budget) every round, and its head request is served
    once the deficit covers that request's estimated token cost. Unused deficit
    carries forward, which bounds the head-of-line delay a large request can
    impose on smaller ones.
    """

    DEFAULT_QUANTA = {
        "tenant-a": 16,
        "tenant-b": 16,
        "tenant-c": 16,
    }

    def __init__(self, tenant_quanta: dict[str, int] | None = None):
        quanta = tenant_quanta if tenant_quanta else self.DEFAULT_QUANTA
        self.tenant_quanta = {tenant: max(1, int(q)) for tenant, q in quanta.items()}
        self.tenants: dict[str, TenantState] = {
            tenant: TenantState(tenant=tenant, quantum=quantum)
            for tenant, quantum in self.tenant_quanta.items()
        }
        self._rr_order = list(self.tenants.keys())
        self._rr_index = 0

    def _ensure_tenant(self, tenant: str) -> None:
        if tenant not in self.tenants:
            quantum = max(1, int(self.tenant_quanta.get(tenant, 16)))
            self.tenants[tenant] = TenantState(tenant=tenant, quantum=quantum)
            self._rr_order.append(tenant)

    def enqueue(self, request: InferenceRequest) -> None:
        self._ensure_tenant(request.tenant)
        self.tenants[request.tenant].queue.append(request)

    def _request_cost(self, request: InferenceRequest) -> int:
        if request.prompt_tokens is not None:
            return max(1, int(request.prompt_tokens))
        return 1

    def select_next(self) -> InferenceRequest:
        if not any(state.queue for state in self.tenants.values()):
            raise RuntimeError("No queued requests available")

        while True:
            tenant = self._rr_order[self._rr_index]
            state = self.tenants[tenant]
            if state.queue:
                state.deficit += state.quantum
                req = state.queue[0]
                cost = self._request_cost(req)
                if state.deficit >= cost:
                    state.deficit -= cost
                    self._rr_index = (self._rr_index + 1) % len(self._rr_order)
                    return state.queue.popleft()
            self._rr_index = (self._rr_index + 1) % len(self._rr_order)
