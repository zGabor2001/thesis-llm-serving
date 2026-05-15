from __future__ import annotations

from dataclasses import dataclass

from .config import ExperimentConfig
from .models import CompletedRequest, Request
from .scheduler import Scheduler


@dataclass
class SimulationResult:
    completed_requests: list[CompletedRequest]
    served_tokens_by_tenant: dict[str, int]
    cold_starts: int
    total_runtime_ms: int


def _service_model(cfg: ExperimentConfig, req: Request, current_adapter: str | None, current_time: int) -> tuple[int, int]:
    wait_time = max(0, current_time - req.arrival_time_ms)
    if current_adapter == req.adapter:
        switch_penalty = 18
    else:
        switch_penalty = 140 if cfg.residency == "warm" else 900

    prefill_cost = req.prompt_tokens * (0.75 if req.whale else 0.42)
    decode_cost = req.output_tokens * 0.9
    interference = 260 if req.whale else (90 if wait_time > 300 else 35)
    ttft = round(wait_time + switch_penalty + prefill_cost + interference)
    service_time = round(switch_penalty + prefill_cost + decode_cost + (420 if req.whale else 80))
    return ttft, service_time


def run_simulation(cfg: ExperimentConfig, workload: list[Request]) -> SimulationResult:
    scheduler = Scheduler(cfg, workload)
    completed: list[CompletedRequest] = []
    served_tokens = {t: 0 for t in cfg.tenant_weights}
    current_time = 0
    current_adapter: str | None = None
    cold_starts = 0

    while scheduler.remaining() > 0:
        req = scheduler.pop_next()
        if req is None:
            break

        start_time = max(current_time, req.arrival_time_ms)
        ttft, service_time = _service_model(cfg, req, current_adapter, current_time)
        switch_penalty = 18 if current_adapter == req.adapter else (140 if cfg.residency == "warm" else 900)
        if current_adapter != req.adapter:
            cold_starts += 1
        current_adapter = req.adapter
        finish_time = start_time + service_time
        wait_time = max(0, current_time - req.arrival_time_ms)
        current_time = finish_time
        served_tokens[req.tenant] += req.total_tokens

        completed.append(
            CompletedRequest(
                request_id=req.request_id,
                tenant=req.tenant,
                adapter=req.adapter,
                whale=req.whale,
                arrival_time_ms=req.arrival_time_ms,
                start_time_ms=start_time,
                finish_time_ms=finish_time,
                wait_time_ms=wait_time,
                ttft_ms=ttft,
                service_time_ms=service_time,
                prompt_tokens=req.prompt_tokens,
                output_tokens=req.output_tokens,
                total_tokens=req.total_tokens,
                switch_penalty_ms=switch_penalty,
            )
        )

    return SimulationResult(
        completed_requests=completed,
        served_tokens_by_tenant=served_tokens,
        cold_starts=cold_starts,
        total_runtime_ms=current_time,
    )
