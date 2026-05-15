from __future__ import annotations

from collections import deque

from .config import ExperimentConfig
from .models import Request


class Scheduler:
    def __init__(self, cfg: ExperimentConfig, workload: list[Request]) -> None:
        self.cfg = cfg
        self.queues: dict[str, deque[Request]] = {t: deque() for t in cfg.tenant_weights}
        for req in workload:
            self.queues[req.tenant].append(req)
        self.order = list(cfg.tenant_weights.keys())
        self.rr_index = 0
        self.deficits = {t: 0 for t in cfg.tenant_weights}

    def remaining(self) -> int:
        return sum(len(q) for q in self.queues.values())

    def pop_next(self) -> Request | None:
        if self.cfg.scheduler == "DRR":
            return self._pop_drr()
        return self._pop_wrr()

    def _pop_wrr(self) -> Request | None:
        for _ in range(len(self.order) * 8):
            tenant = self.order[self.rr_index % len(self.order)]
            self.rr_index += 1
            if not self.queues[tenant]:
                continue
            weight = self.cfg.tenant_weights[tenant]
            if (_ % max(1, round(4 / max(1, weight)))) == 0:
                return self.queues[tenant].popleft()
        for tenant in self.order:
            if self.queues[tenant]:
                return self.queues[tenant].popleft()
        return None

    def _pop_drr(self) -> Request | None:
        for _ in range(len(self.order) * 6):
            tenant = self.order[self.rr_index % len(self.order)]
            self.rr_index += 1
            if not self.queues[tenant]:
                continue
            self.deficits[tenant] += self.cfg.quantum * self.cfg.tenant_weights[tenant]
            head = self.queues[tenant][0]
            if self.deficits[tenant] >= head.total_tokens:
                self.deficits[tenant] -= head.total_tokens
                return self.queues[tenant].popleft()
        for tenant in self.order:
            if self.queues[tenant]:
                head = self.queues[tenant][0]
                self.deficits[tenant] = max(0, self.deficits[tenant] - head.total_tokens)
                return self.queues[tenant].popleft()
        return None
