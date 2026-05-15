from __future__ import annotations

import random

from .config import ExperimentConfig
from .models import Request


def generate_workload(cfg: ExperimentConfig) -> list[Request]:
    rng = random.Random(cfg.seed)
    tenants = list(cfg.tenant_weights.keys())
    requests: list[Request] = []
    arrival = 0

    for i in range(cfg.request_count):
        if rng.random() < cfg.burstiness:
            x = rng.random()
            if x < 0.55:
                tenant = "A"
            elif x < 0.80:
                tenant = "B"
            elif x < 0.93:
                tenant = "C"
            else:
                tenant = "D"
        else:
            tenant = rng.choice(tenants)

        whale = rng.random() < cfg.whale_rate or (tenant == "C" and rng.random() < 0.35)
        prompt_tokens = rng.randint(1800, 4200) if whale else rng.randint(40, 320)
        output_tokens = rng.randint(400, 1300) if whale else rng.randint(30, 150)
        gap = rng.randint(1, 4) if cfg.burstiness > 0.3 else rng.randint(1, 8)
        arrival += gap

        requests.append(
            Request(
                request_id=i + 1,
                tenant=tenant,
                adapter=cfg.tenant_adapters[tenant],
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                arrival_time_ms=arrival,
                whale=whale,
            )
        )

    return requests
