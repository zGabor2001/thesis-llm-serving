from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .simulator import SimulationResult


def jains_index(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    numerator = arr.sum() ** 2
    denominator = len(arr) * np.square(arr).sum()
    return 0.0 if denominator == 0 else float(numerator / denominator)


def build_run_summary(cfg: ExperimentConfig, sim: SimulationResult) -> dict:
    ttft = np.array([r.ttft_ms for r in sim.completed_requests], dtype=float)
    total_tokens = int(sum(r.total_tokens for r in sim.completed_requests))
    throughput = total_tokens / max(1e-9, sim.total_runtime_ms / 1000)

    return {
        "scheduler": cfg.scheduler,
        "residency": cfg.residency,
        "request_count": len(sim.completed_requests),
        "burstiness": cfg.burstiness,
        "whale_rate": cfg.whale_rate,
        "quantum": cfg.quantum,
        "cold_starts": sim.cold_starts,
        "total_runtime_ms": sim.total_runtime_ms,
        "throughput_tokens_per_sec": round(float(throughput), 3),
        "avg_ttft_ms": round(float(ttft.mean()), 3),
        "p95_ttft_ms": round(float(np.percentile(ttft, 95)), 3),
        "p99_ttft_ms": round(float(np.percentile(ttft, 99)), 3),
        "jains_fairness_index": round(jains_index(list(sim.served_tokens_by_tenant.values())), 6),
        "served_tokens_by_tenant": sim.served_tokens_by_tenant,
    }


def build_tenant_summary(cfg: ExperimentConfig, sim: SimulationResult) -> pd.DataFrame:
    rows = []
    total_tokens = sum(sim.served_tokens_by_tenant.values()) or 1
    for tenant, weight in cfg.tenant_weights.items():
        subset = [r for r in sim.completed_requests if r.tenant == tenant]
        avg_ttft = float(np.mean([r.ttft_ms for r in subset])) if subset else 0.0
        p95_ttft = float(np.percentile([r.ttft_ms for r in subset], 95)) if subset else 0.0
        served = sim.served_tokens_by_tenant[tenant]
        rows.append({
            "tenant": tenant,
            "weight": weight,
            "requests_served": len(subset),
            "served_tokens": served,
            "served_share": round(served / total_tokens, 6),
            "avg_ttft_ms": round(avg_ttft, 3),
            "p95_ttft_ms": round(p95_ttft, 3),
        })
    return pd.DataFrame(rows)
