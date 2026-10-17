from __future__ import annotations

import json
import time
from typing import Any

import requests

from .models import InferenceRequest, RequestResult


class VLLMClientError(Exception):
    pass


def parse_sse_line(line: str) -> dict | None:
    """Parse one Server-Sent-Events ``data:`` line into a dict, or None."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return {"done": True}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


class VLLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.default_headers: dict[str, str] = {}
        if api_key:
            self.default_headers["Authorization"] = f"Bearer {api_key}"

    def close(self) -> None:
        self.session.close()

    def _get(self, path: str) -> requests.Response:
        url = f"{self.base_url}{path}"
        return self.session.get(url, headers=self.default_headers, timeout=self.timeout)

    def _post(self, path: str, payload: dict[str, Any]) -> requests.Response:
        url = f"{self.base_url}{path}"
        return self.session.post(
            url,
            json=payload,
            headers=self.default_headers,
            timeout=self.timeout,
        )

    def _post_stream(self, path: str, payload: dict[str, Any]) -> requests.Response:
        url = f"{self.base_url}{path}"
        return self.session.post(
            url,
            json=payload,
            headers=self.default_headers,
            timeout=self.timeout,
            stream=True,
        )

    def health(self) -> bool:
        r = self._get("/health")
        return r.status_code == 200

    def ping(self) -> bool:
        r = self._get("/ping")
        return r.status_code == 200

    def version(self) -> str:
        r = self._get("/version")
        r.raise_for_status()
        return r.text.strip()

    def list_models(self) -> list[str]:
        r = self._get("/v1/models")
        r.raise_for_status()
        body = r.json()
        return [item["id"] for item in body.get("data", [])]

    def wait_until_ready(self, max_wait_s: float = 180.0, poll_s: float = 2.0) -> bool:
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            try:
                if self.health():
                    return True
            except requests.RequestException:
                pass
            time.sleep(poll_s)
        return False

    def build_payload(self, req: InferenceRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if "stream" in req.metadata:
            payload["stream"] = req.metadata["stream"]
        if "top_p" in req.metadata:
            payload["top_p"] = req.metadata["top_p"]
        if "extra_body" in req.metadata and isinstance(req.metadata["extra_body"], dict):
            payload.update(req.metadata["extra_body"])
        return payload

    def complete(self, req: InferenceRequest) -> RequestResult:
        dispatch_ts = time.time()
        payload = self.build_payload(req)
        payload["stream"] = True
        payload.setdefault("stream_options", {"include_usage": True})

        try:
            response = self._post_stream("/v1/chat/completions", payload)
        except requests.RequestException as e:
            return self._error_result(req, dispatch_ts, e)

        if response.status_code >= 400:
            return self._error_result(
                req,
                dispatch_ts,
                VLLMClientError(f"status {response.status_code}: {response.text}"),
            )

        try:
            text, first_token_ts, usage = self._consume_stream(response)
        except Exception as e:  # noqa: BLE001
            return self._error_result(req, dispatch_ts, e)

        finish_ts = time.time()

        return RequestResult(
            request_id=req.request_id,
            tenant=req.tenant,
            model=req.model,
            adapter=req.adapter,
            arrival_ts=req.arrival_ts,
            dispatch_ts=dispatch_ts,
            finish_ts=finish_ts,
            first_token_ts=first_token_ts,
            prompt_tokens=usage.get("prompt_tokens", req.prompt_tokens),
            generated_tokens=usage.get("completion_tokens"),
            temperature=req.temperature,
            status="completed",
            error=None,
            response_text=text,
            metadata={
                "status_code": response.status_code,
                "streamed": True,
                "usage": usage,
            },
        )

    def _consume_stream(self, response: requests.Response) -> tuple[str, float | None, dict]:
        text_parts: list[str] = []
        first_token_ts: float | None = None
        usage: dict[str, Any] = {}

        for raw in response.iter_lines(decode_unicode=True):
            if not raw:
                continue
            chunk = parse_sse_line(raw)
            if chunk is None or chunk.get("done"):
                continue

            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    if first_token_ts is None:
                        first_token_ts = time.time()
                    text_parts.append(content)

            if chunk.get("usage"):
                usage = chunk["usage"]

        return "".join(text_parts), first_token_ts, usage

    def _error_result(
        self, req: InferenceRequest, dispatch_ts: float, exc: Exception
    ) -> RequestResult:
        finish_ts = time.time()
        return RequestResult(
            request_id=req.request_id,
            tenant=req.tenant,
            model=req.model,
            adapter=req.adapter,
            arrival_ts=req.arrival_ts,
            dispatch_ts=dispatch_ts,
            finish_ts=finish_ts,
            first_token_ts=None,
            prompt_tokens=req.prompt_tokens,
            generated_tokens=None,
            temperature=req.temperature,
            status="failed",
            error=str(exc),
            response_text=None,
            metadata={"exception_type": type(exc).__name__},
        )


class MockBackendClient:
    def __init__(self, fixed_delay_s: float = 0.2) -> None:
        self.fixed_delay_s = fixed_delay_s

    def health(self) -> bool:
        return True

    def ping(self) -> bool:
        return True

    def version(self) -> str:
        return "mock-backend"

    def list_models(self) -> list[str]:
        return ["base-model", "tenant-a", "tenant-b", "tenant-c"]

    def wait_until_ready(self, max_wait_s: float = 1.0, poll_s: float = 0.1) -> bool:
        return True

    def complete(self, req: InferenceRequest) -> RequestResult:
        dispatch_ts = time.time()
        time.sleep(self.fixed_delay_s)
        finish_ts = time.time()
        first_token_ts = dispatch_ts + (finish_ts - dispatch_ts) * 0.25
        prompt = req.prompt_text or ""
        text = f"Mock response for tenant={req.tenant}, model={req.model}. Prompt={prompt[:80]}"

        return RequestResult(
            request_id=req.request_id,
            tenant=req.tenant,
            model=req.model,
            adapter=req.adapter,
            arrival_ts=req.arrival_ts,
            dispatch_ts=dispatch_ts,
            finish_ts=finish_ts,
            first_token_ts=first_token_ts,
            prompt_tokens=req.prompt_tokens,
            generated_tokens=min(req.max_tokens, 24),
            temperature=req.temperature,
            status="completed",
            error=None,
            response_text=text,
            metadata={"mock": True},
        )
