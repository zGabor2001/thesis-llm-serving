from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Request:
    request_id: int
    tenant: str
    adapter: str
    prompt_tokens: int
    output_tokens: int
    arrival_time_ms: int
    whale: bool

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


@dataclass
class CompletedRequest:
    request_id: int
    tenant: str
    adapter: str
    whale: bool
    arrival_time_ms: int
    start_time_ms: int
    finish_time_ms: int
    wait_time_ms: int
    ttft_ms: int
    service_time_ms: int
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    switch_penalty_ms: int

    def to_dict(self) -> dict:
        return asdict(self)
