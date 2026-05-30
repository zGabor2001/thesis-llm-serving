from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class GatewayConfig:
    backend_mode: str = "mock"
    vllm_base_url: str = "http://localhost:8000"
    default_model: str = "base-model"
    host: str = "0.0.0.0"
    port: int = 8080
    request_timeout_s: float = 60.0
    scheduler_mode: str = "fifo"
    max_queue_size: int = 1024
    backend_ids: list[str] = field(default_factory=lambda: ["gpu-0"])

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        return cls(
            backend_mode=os.getenv("BACKEND_MODE", "mock"),
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
            default_model=os.getenv("DEFAULT_MODEL", "base-model"),
            host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
            port=int(os.getenv("GATEWAY_PORT", "8080")),
            request_timeout_s=float(os.getenv("REQUEST_TIMEOUT_S", "60.0")),
            scheduler_mode=os.getenv("SCHEDULER_MODE", "fifo"),
            max_queue_size=int(os.getenv("MAX_QUEUE_SIZE", "1024")),
            backend_ids=os.getenv("BACKEND_IDS", "gpu-0").split(","),
        )