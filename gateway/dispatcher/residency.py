from __future__ import annotations

import threading


class ResidencyTracker:
    """Tracks model residency for cold/switch classification.

    A request is ``cold`` if it is the first request served for its model, and
    a ``switch`` if its model differs from the previously served request's
    model. Thread-safe: called from the dispatcher worker threads in serve
    order, so the classification reflects the actual dispatch sequence.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._last: str | None = None

    def mark(self, model: str) -> dict[str, bool]:
        with self._lock:
            cold = model not in self._seen
            switch = self._last is not None and self._last != model
            self._seen.add(model)
            self._last = model
            return {"cold": cold, "switch": switch}
