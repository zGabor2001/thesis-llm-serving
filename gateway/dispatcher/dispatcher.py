from __future__ import annotations

from collections import deque

from gateway.models import InferenceRequest


class FIFODispatcher:
    """FIFO admission policy: requests are served in arrival order."""

    def __init__(self) -> None:
        self._queue: deque[InferenceRequest] = deque()

    def enqueue(self, request: InferenceRequest) -> None:
        self._queue.append(request)

    def select_next(self) -> InferenceRequest:
        if not self._queue:
            raise RuntimeError("No queued requests available")
        return self._queue.popleft()
