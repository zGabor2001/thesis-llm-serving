from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def parse_weight_map(spec: str) -> dict[str, int]:
    """Parse ``'a:2,b:1'`` into ``{'a': 2, 'b': 1}``.

    A key without an explicit value gets 1. Used for tenant weights and quanta.
    """
    result: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, value = part.split(":", 1)
        else:
            key, value = part, "1"
        weight = int(value.strip())
        if weight < 1:
            raise ValueError(f"value for '{key}' must be >= 1")
        result[key.strip()] = weight
    if not result:
        raise ValueError("empty value map")
    return result


def _parse_optional_map(value: str | None) -> dict[str, int] | None:
    if not value:
        return None
    return parse_weight_map(value)


@dataclass
class GatewayConfig:
    backend_mode: str = "mock"
    vllm_base_url: str = "http://localhost:8000"
    default_model: str = "base-model"
    host: str = "0.0.0.0"
    port: int = 8080
    request_timeout_s: float = 60.0
    scheduler_mode: str = "fifo"
    max_concurrency: int = 32
    tenant_weights: dict[str, int] | None = None
    tenant_quanta: dict[str, int] | None = None
    lora_tenants: list[str] = field(default_factory=lambda: ["tenant-a", "tenant-b", "tenant-c"])

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
            max_concurrency=int(os.getenv("MAX_CONCURRENCY", "32")),
            tenant_weights=_parse_optional_map(os.getenv("TENANT_WEIGHTS")),
            tenant_quanta=_parse_optional_map(os.getenv("TENANT_QUANTA")),
            lora_tenants=[t.strip() for t in os.getenv("LORA_TENANTS", "tenant-a,tenant-b,tenant-c").split(",") if t.strip()],
        )
