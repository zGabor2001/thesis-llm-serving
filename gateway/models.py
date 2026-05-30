from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


PriorityClass = Literal["interactive", "standard", "background", "batch"]
RequestStatus = Literal["queued", "dispatched", "completed", "failed"]


@dataclass
class InferenceRequest:
    request_id: str
    tenant: str
    model: str
    arrival_ts: float
    messages: list[dict[str, str]]
    max_tokens: int

    adapter: str | None = None
    prompt_text: str | None = None
    prompt_tokens: int | None = None
    temperature: float = 0.0
    priority_class: PriorityClass = "standard"
    metadata: dict[str, Any] = field(default_factory=dict)

    enqueue_ts: float | None = None
    dispatch_ts: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RequestResult:
    request_id: str
    tenant: str
    model: str
    arrival_ts: float
    dispatch_ts: float
    finish_ts: float
    status: RequestStatus

    adapter: str | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    first_token_ts: float | None = None
    temperature: float = 0.0
    error: str | None = None
    response_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def queue_delay(self) -> float:
        return self.dispatch_ts - self.arrival_ts

    @property
    def latency(self) -> float:
        return self.finish_ts - self.arrival_ts

    @property
    def service_time(self) -> float:
        return self.finish_ts - self.dispatch_ts

    @property
    def ttft(self) -> float | None:
        if self.first_token_ts is None:
            return None
        return self.first_token_ts - self.arrival_ts

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Request = InferenceRequest
CompletedRequest = RequestResult