from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import requests

from .models import InferenceRequest, RequestResult


class VLLMClientError(Exception):
    pass


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

        try:
            response = self._post("/v1/chat/completions", payload)
            finish_ts = time.time()
        except requests.RequestException as e:
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
                error=str(e),
                response_text=None,
                metadata={"exception_type": type(e).__name__},
            )

        if response.status_code >= 400:
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
                error=response.text,
                response_text=None,
                metadata={"status_code": response.status_code},
            )

        body = response.json()
        choices = body.get("choices", [])
        if not choices:
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
                error="Missing choices in response",
                response_text=None,
                metadata={"raw_response": body},
            )

        message = choices[0].get("message", {})
        text = message.get("content", "")

        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", req.prompt_tokens)
        generated_tokens = usage.get("completion_tokens")

        return RequestResult(
            request_id=req.request_id,
            tenant=req.tenant,
            model=req.model,
            adapter=req.adapter,
            arrival_ts=req.arrival_ts,
            dispatch_ts=dispatch_ts,
            finish_ts=finish_ts,
            first_token_ts=finish_ts,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            temperature=req.temperature,
            status="completed",
            error=None,
            response_text=text,
            metadata={
                "status_code": response.status_code,
                "response_id": body.get("id"),
                "object": body.get("object"),
                "usage": usage,
            },
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
            first_token_ts=finish_ts,
            prompt_tokens=req.prompt_tokens,
            generated_tokens=min(req.max_tokens, 24),
            temperature=req.temperature,
            status="completed",
            error=None,
            response_text=text,
            metadata={"mock": True},
        )