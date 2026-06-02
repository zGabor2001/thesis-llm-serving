from __future__ import annotations

import time
from dataclasses import dataclass

from gateway.models import InferenceRequest, RequestResult


@dataclass
class DispatchResult:
    result: RequestResult
    dispatch_ts: float


class FIFODispatcher:
    def __init__(self, backend_client):
        self.backend_client = backend_client

    def dispatch(self, request: InferenceRequest) -> DispatchResult:
        dispatch_ts = time.time()
        result = self.backend_client.complete(request)
        return DispatchResult(result=result, dispatch_ts=dispatch_ts)
