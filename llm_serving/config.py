from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    scheduler: str = "DRR"
    residency: str = "warm"
    request_count: int = 600
    burstiness: float = 0.45
    whale_rate: float = 0.12
    quantum: int = 256
    seed: int = 42
    tenant_weights: dict[str, int] = field(default_factory=lambda: {"A": 3, "B": 2, "C": 1, "D": 1})
    tenant_adapters: dict[str, str] = field(default_factory=lambda: {
        "A": "adapter-chat",
        "B": "adapter-code",
        "C": "adapter-doc",
        "D": "adapter-research",
    })
