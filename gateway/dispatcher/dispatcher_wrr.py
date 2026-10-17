from __future__ import annotations

from collections import deque

from gateway.models import InferenceRequest


class WRRDispatcher:
    """Weighted round-robin admission policy (request-based accounting).

    Each request costs exactly one unit, regardless of its token count. At
    every step the tenant with the lowest ``served / weight`` ratio is picked,
    which yields a deterministic weighted round-robin schedule. Ties are broken
    by tenant order (first declared wins).
    """

    DEFAULT_WEIGHTS = {
        "tenant-a": 1,
        "tenant-b": 1,
        "tenant-c": 1,
        "base-model": 1,
    }

    def __init__(self, tenant_weights: dict[str, int] | None = None):
        self.tenant_weights = dict(tenant_weights) if tenant_weights else dict(self.DEFAULT_WEIGHTS)
        self.queues: dict[str, deque] = {t: deque() for t in self.tenant_weights}
        self.served: dict[str, int] = {t: 0 for t in self.tenant_weights}
        self._order = list(self.tenant_weights.keys())

    def _ensure_tenant(self, tenant: str) -> None:
        if tenant not in self.queues:
            self.tenant_weights[tenant] = self.tenant_weights.get(tenant, 1)
            self.queues[tenant] = deque()
            self.served[tenant] = 0
            self._order.append(tenant)

    def enqueue(self, request: InferenceRequest) -> None:
        self._ensure_tenant(request.tenant)
        self.queues[request.tenant].append(request)

    def select_next(self) -> InferenceRequest:
        best: str | None = None
        best_ratio: float | None = None

        for tenant in self._order:
            if not self.queues[tenant]:
                continue
            ratio = self.served[tenant] / self.tenant_weights[tenant]
            if best is None or ratio < best_ratio:
                best = tenant
                best_ratio = ratio

        if best is None:
            raise RuntimeError("No queued requests available")

        self.served[best] += 1
        return self.queues[best].popleft()
