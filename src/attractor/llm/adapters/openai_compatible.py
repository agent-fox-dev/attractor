"""OpenAI-compatible adapter for third-party services.

Targets services that implement the OpenAI Chat Completions API
(vLLM, Ollama, Together AI, Groq, etc.) but may not support the
newer Responses API.

Uses ``/v1/chat/completions`` instead of ``/v1/responses``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from attractor.llm.adapters.base import ProviderAdapter
from attractor.llm.types import (
    AccessDeniedError,
    AuthenticationError,
    ContentFilterError,
    ContentKind,
    ContentPart,
    ContextLengthError,
    FinishReason,
    InvalidRequestError,
    Message,
    NetworkError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    Request,
    RequestTimeoutError,
    Response,
    Role,
    ServerError,
    StreamEvent,
    StreamEventKind,
    ToolCallData,
    ToolDefinition,
    Usage,
)

logger = logging.getLogger(__name__)


def _map_status_to_error(status_code: int, body: str, provider: str) -> ProviderError:
    msg = f"OpenAI-compatible API error {status_code}: {body}"
    kw: dict = {"provider": provider, "raw": body}
    if status_code == 401:
        return AuthenticationError(msg, **kw)
    if status_code == 403:
        return AccessDeniedError(msg, **kw)
    if status_code == 404:
        return NotFoundError(msg, **kw)
    if status_code == 408:
        return RequestTimeoutError(msg, **kw)
    if status_code in (400, 422):
        return InvalidRequestError(msg, **kw)
    if status_code == 413:
        return ContextLengthError(msg, **kw)
    if status_code == 429:
        return RateLimitError(msg, **kw)
    if status_code >= 500:
        return ServerError(msg, status_code=status_code, **kw)
    # Body-based classification for ambiguous status codes (spec Section 6.5)
    lower = body.lower()
    if "not found" in lower or "does not exist" in lower:
        return NotFoundError(msg, **kw)
    if "unauthorized" in lower or "invalid key" in lower:
        return AuthenticationError(msg, **kw)
    if "context length" in lower or "too many tokens" in lower:
        return ContextLengthError(msg, **kw)
    if "content filter" in lower or "safety" in lower:
        return ContentFilterError(msg, **kw)
    return ProviderError(msg, status_code=status_code, **kw)


class OpenAICompatibleAdapter(ProviderAdapter):
    """Adapter for OpenAI-compatible Chat Completions API services.

    Parameters
    ----------
    api_key:
        API key for the service.
    base_url:
        Base URL (e.g. ``http://localhost:11434/v1`` for Ollama).
    provider_name:
        Name to use in error messages and logging.
    default_headers:
        Extra headers to include in every request.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str,
        provider_name: str = "openai_compatible",
        default_headers: dict[str, str] | None = None,
        timeout: float = 300.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._provider_name = provider_name
        self._default_headers = default_headers or {}
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return self._provider_name

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json", **self._default_headers}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(self._timeout),
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_payload(self, request: Request, stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": request.model, "stream": stream}

        # Convert messages
        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            role = msg.role.value
            if role == "developer":
                role = "system"
            text = msg.text
            messages.append({"role": role, "content": text})
        payload["messages"] = messages

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        if request.response_format:
            fmt = request.response_format
            if hasattr(fmt, 'type'):
                fmt_type = fmt.type
                fmt_schema = getattr(fmt, 'json_schema', None)
            else:
                fmt_type = fmt.get("type", "")
                fmt_schema = fmt.get("json_schema")
            if fmt_type == "json_object":
                payload["response_format"] = {"type": "json_object"}
            elif fmt_type == "json_schema" and fmt_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": fmt_schema,
                }

        # Tools
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.parameters or {},
                    },
                }
                for t in request.tools
            ]
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        return payload

    async def complete(self, request: Request) -> Response:
        payload = self._build_payload(request)
        client = self._ensure_client()
        url = f"{self._base_url}/chat/completions"

        try:
            resp = await client.post(url, json=payload)
        except httpx.TransportError as exc:
            raise NetworkError(str(exc), provider=self._provider_name) from exc

        if resp.status_code != 200:
            raise _map_status_to_error(
                resp.status_code, resp.text[:500], self._provider_name,
            )

        data = resp.json()
        return self._parse_response(data)

    def _parse_response(self, data: dict[str, Any]) -> Response:
        choices = data.get("choices", [])
        if not choices:
            return Response(model=data.get("model", ""))

        choice = choices[0]
        msg = choice.get("message", {})
        content_text = msg.get("content", "") or ""
        parts: list[ContentPart] = []

        if content_text:
            parts.append(ContentPart(kind=ContentKind.TEXT, text=content_text))

        # Tool calls
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                parsed_args = {"raw": args}
            parts.append(ContentPart(
                kind=ContentKind.TOOL_CALL,
                tool_call=ToolCallData(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=parsed_args,
                ),
            ))

        # Usage
        usage_data = data.get("usage", {})
        cached_tokens = usage_data.get("prompt_tokens_details", {}).get("cached_tokens") if isinstance(usage_data.get("prompt_tokens_details"), dict) else None
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            cache_read_tokens=cached_tokens,
            raw=usage_data,
        )

        # Finish reason
        fr_str = choice.get("finish_reason", "stop")
        fr_map = {"stop": FinishReason.STOP, "tool_calls": FinishReason.TOOL_CALLS,
                   "length": FinishReason.LENGTH, "content_filter": FinishReason.CONTENT_FILTER}
        finish_reason = fr_map.get(fr_str, FinishReason.STOP)

        return Response(
            id=data.get("id", ""),
            model=data.get("model", ""),
            provider=self._provider_name,
            content=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw_finish_reason=fr_str,
            raw=data,
        )

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        payload = self._build_payload(request, stream=True)
        client = self._ensure_client()
        url = f"{self._base_url}/chat/completions"

        try:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise _map_status_to_error(
                        resp.status_code, body.decode()[:500], self._provider_name,
                    )

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield StreamEvent(
                            kind=StreamEventKind.TEXT_DELTA,
                            data={"text": content},
                        )

                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        fr_map = {"stop": FinishReason.STOP, "tool_calls": FinishReason.TOOL_CALLS,
                                  "length": FinishReason.LENGTH}
                        yield StreamEvent(
                            kind=StreamEventKind.TEXT_END,
                            finish_reason=fr_map.get(finish_reason, FinishReason.STOP),
                        )

                    if "usage" in chunk:
                        u = chunk["usage"]
                        yield StreamEvent(
                            kind=StreamEventKind.USAGE,
                            usage=Usage(
                                input_tokens=u.get("prompt_tokens", 0),
                                output_tokens=u.get("completion_tokens", 0),
                                total_tokens=u.get("total_tokens", 0),
                            ),
                        )

        except httpx.TransportError as exc:
            raise NetworkError(str(exc), provider=self._provider_name) from exc
