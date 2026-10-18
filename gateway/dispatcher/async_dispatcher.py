from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from gateway.dispatcher.residency import ResidencyTracker
from gateway.models import InferenceRequest, RequestResult


@dataclass
class DispatchResult:
    result: RequestResult
    dispatch_ts: float


class AsyncDispatcher:
    """Admission-controlled serving on top of a scheduling policy.

    ``dispatch()`` enqueues a request, lets a background scheduler pick requests
    in the order the wrapped policy dictates, and blocks the caller only until
    *its own* request completes. A bounded worker pool keeps multiple requests
    in flight, so the scheduler controls admission order without serializing
    backend work.
    """

    def __init__(self, policy, backend_client, max_concurrency: int = 32):
        self.policy = policy
        self.backend_client = backend_client
        self.tracker = ResidencyTracker()

        self._condition = threading.Condition()
        self._slots = threading.Semaphore(max_concurrency)
        self._futures: dict[str, Future] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency, thread_name_prefix="gw-serve"
        )
        self._stopped = threading.Event()
        self._scheduler = threading.Thread(
            target=self._run, name="gw-scheduler", daemon=True
        )
        self._scheduler.start()

    def dispatch(self, request: InferenceRequest, timeout: float | None = None) -> DispatchResult:
        future: Future = Future()
        with self._condition:
            self.policy.enqueue(request)
            self._futures[request.request_id] = future
            self._condition.notify()

        result = future.result(timeout=timeout)
        return DispatchResult(result=result, dispatch_ts=result.dispatch_ts)

    def _run(self) -> None:
        while not self._stopped.is_set():
            self._slots.acquire()
            request = None
            with self._condition:
                try:
                    request = self.policy.select_next()
                except RuntimeError:
                    request = None

            if request is None:
                self._slots.release()
                with self._condition:
                    self._condition.wait(timeout=0.5)
                continue

            self._executor.submit(self._serve, request)

    def _serve(self, request: InferenceRequest) -> None:
        cold_switch = self.tracker.mark(request.model or "unknown")
        try:
            result = self.backend_client.complete(request)
        except Exception as exc:  # noqa: BLE001 — keep the failure inside the result
            result = self._failure(request, exc)
        finally:
            self._slots.release()

        result.metadata = {**result.metadata, **cold_switch}

        with self._condition:
            future = self._futures.pop(request.request_id, None)
        if future is not None and not future.done():
            future.set_result(result)

    @staticmethod
    def _failure(request: InferenceRequest, exc: Exception) -> RequestResult:
        ts = time.time()
        return RequestResult(
            request_id=request.request_id,
            tenant=request.tenant,
            model=request.model,
            adapter=request.adapter,
            arrival_ts=request.arrival_ts,
            dispatch_ts=ts,
            finish_ts=ts,
            first_token_ts=None,
            prompt_tokens=request.prompt_tokens,
            generated_tokens=None,
            temperature=request.temperature,
            status="failed",
            error=str(exc),
            response_text=None,
            metadata={"exception_type": type(exc).__name__},
        )

    def close(self, wait: bool = True) -> None:
        self._stopped.set()
        with self._condition:
            self._condition.notify_all()
        self._executor.shutdown(wait=wait, cancel_futures=True)
