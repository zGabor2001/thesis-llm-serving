from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from gateway.config import GatewayConfig
from gateway.models import InferenceRequest
from gateway.request_logger import RequestLogger
from gateway.vllm_client import MockBackendClient, VLLMClient
from gateway.dispatcher.dispatcher_factory import build_dispatcher


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.0
    tenant: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def make_backend_client(cfg: GatewayConfig) -> Any:
    if cfg.backend_mode.lower() == "vllm":
        return VLLMClient(
            base_url=cfg.vllm_base_url,
            timeout_s=cfg.request_timeout_s,
        )
    return MockBackendClient()


def build_prompt_text(messages: list[ChatMessage]) -> str:
    user_parts = [m.content for m in messages if m.role == "user"]
    if user_parts:
        return "\n".join(user_parts)
    return "\n".join(m.content for m in messages)


def to_openai_like_response(
    result: Any,
    *,
    tenant: str,
    model: str,
    created_ts: float,
) -> dict[str, Any]:
    prompt_tokens = getattr(result, "prompt_tokens", 0) or 0
    completion_tokens = getattr(result, "generated_tokens", 0) or 0

    return {
        "id": getattr(result, "request_id", str(uuid.uuid4())),
        "object": "chat.completion",
        "created": int(created_ts),
        "model": getattr(result, "model", model) or model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": getattr(result, "response_text", "") or "",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "tenant": tenant,
    }


def build_app(
    config: GatewayConfig | None = None,
    backend_client: Any | None = None,
    logger: RequestLogger | None = None,
    tenant_weights: dict[str, int] | None = None,
    scheduler_policy: str = "fifo",
    tenant_quanta: dict[str, int] | None = None,
) -> FastAPI:
    cfg = config or GatewayConfig.from_env()
    client = backend_client or make_backend_client(cfg)
    if logger is None:
        logger = RequestLogger()
    dispatcher = build_dispatcher(
        backend_client,
        policy=scheduler_policy,
        tenant_weights=tenant_weights,
        tenant_quanta=tenant_quanta,
    )

    app = FastAPI(title="LLM Gateway Proxy", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            ok = bool(client.health())
        except Exception:
            ok = False

        return {
            "status": "ok" if ok else "degraded",
            "backend_ok": ok,
            "backend_mode": cfg.backend_mode,
            "default_model": cfg.default_model,
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        try:
            models = client.list_models()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Backend list_models failed: {exc}") from exc

        return {
            "object": "list",
            "data": [{"id": m, "object": "model"} for m in models],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(
        req: ChatCompletionRequest,
        x_tenant_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        tenant = req.tenant or x_tenant_id or "default"
        model = req.model or cfg.default_model
        arrival_ts = time.time()

        inference_request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            tenant=tenant,
            model=model,
            arrival_ts=arrival_ts,
            messages=[m.model_dump() for m in req.messages],
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            metadata=req.metadata,
            prompt_text=build_prompt_text(req.messages),
        )

        logger.log(
            "request_received",
            request_id=inference_request.request_id,
            tenant=tenant,
            model=model,
            arrival_ts=arrival_ts,
            max_tokens=req.max_tokens,
        )

        try:
            dispatch_out = dispatcher.dispatch(inference_request)
            result = dispatch_out.result
            dispatch_ts = dispatch_out.dispatch_ts
            finish_ts = time.time()

            logger.log(
                "request_completed",
                request_id=inference_request.request_id,
                tenant=tenant,
                model=model,
                arrival_ts=arrival_ts,
                dispatch_ts=dispatch_ts,
                finish_ts=finish_ts,
                status=getattr(result, "status", "completed"),
                prompt_tokens=getattr(result, "prompt_tokens", 0),
                generated_tokens=getattr(result, "generated_tokens", 0),
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Backend completion failed: {exc}") from exc

        status = getattr(result, "status", "completed")
        if status not in ("completed", "success"):
            raise HTTPException(status_code=502, detail="Backend completion returned non-success status")

        return to_openai_like_response(
            result,
            tenant=tenant,
            model=model,
            created_ts=arrival_ts,
        )

    return app


app = build_app()
