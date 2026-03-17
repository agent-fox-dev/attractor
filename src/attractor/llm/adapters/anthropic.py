"""Anthropic Messages API adapter.

Translates unified :class:`Request` / :class:`Response` types to and from
the Anthropic ``/v1/messages`` wire format, including streaming via SSE.
"""

from __future__ import annotations

import json
import logging
import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx

from attractor.llm.adapters.base import ProviderAdapter
from attractor.llm.types import (
    AuthenticationError,
    ContentFilterError,
    ContentKind,
    ContentPart,
    ContextLengthError,
    FinishReason,
    InvalidRequestError,
    Message,
    NetworkError,
    ProviderError,
    RateLimitError,
    Request,
    Response,
    Role,
    ServerError,
    StreamError,
    StreamEvent,
    StreamEventKind,
    ThinkingData,
    ToolCallData,
    ToolDefinition,
    Usage,
)

logger = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503, 529}
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds


def _map_status_to_error(status_code: int, body: str, provider: str) -> ProviderError:
    """Map an HTTP status code to the appropriate error subclass."""
    msg = f"Anthropic API error {status_code}: {body}"
    kw: dict = {"provider": provider, "raw": body}
    if status_code == 401:
        return AuthenticationError(msg, **kw)
    if status_code == 403:
        from attractor.llm.types import AccessDeniedError
        return AccessDeniedError(msg, **kw)
    if status_code == 404:
        from attractor.llm.types import NotFoundError
        return NotFoundError(msg, **kw)
    if status_code in (400, 422):
        return InvalidRequestError(msg, **kw)
    if status_code == 413:
        return ContextLengthError(msg, **kw)
    if status_code == 429:
        # Try to parse retry-after from body
        retry_after = None
        try:
            import json as _json
            data = _json.loads(body)
            retry_after = data.get("error", {}).get("retry_after")
        except Exception:
            pass
        return RateLimitError(msg, retry_after=retry_after, **kw)
    if status_code >= 500:
        return ServerError(msg, status_code=status_code, **kw)
    return ProviderError(msg, status_code=status_code, **kw)


class AnthropicAdapter(ProviderAdapter):
    """Adapter for the Anthropic Messages API (``/v1/messages``)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        default_headers: dict[str, str] | None = None,
        timeout: float = 300.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_headers = default_headers or {}
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    # -- ProviderAdapter interface ------------------------------------------

    @property
    def name(self) -> str:
        return "anthropic"

    async def initialize(self) -> None:
        self._client = self._make_client()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(self, request: Request) -> Response:
        payload = self._build_payload(request, stream=False)
        headers = self._extra_headers(request)
        raw = await self._post(payload, headers)
        return self._parse_response(raw)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        payload = self._build_payload(request, stream=True)
        headers = self._extra_headers(request)
        client = self._ensure_client()

        async with client.stream(
            "POST",
            f"{self._base_url}/v1/messages",
            json=payload,
            headers=headers,
            timeout=self._timeout,
        ) as http_resp:
            if http_resp.status_code >= 400:
                body = await http_resp.aread()
                raise _map_status_to_error(
                    http_resp.status_code, body.decode(errors='replace'), self.name,
                )
            async for event in self._parse_sse_stream(http_resp):
                yield event

    def supports_tool_choice(self, mode: str) -> bool:
        return mode in {"auto", "none", "any", "required"}

    # -- HTTP helpers -------------------------------------------------------

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                **self._default_headers,
            },
            timeout=self._timeout,
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _extra_headers(self, request: Request) -> dict[str, str]:
        headers: dict[str, str] = {}
        opts = request.provider_options or {}
        if beta := opts.get("beta_headers"):
            if isinstance(beta, list):
                headers["anthropic-beta"] = ",".join(beta)
            else:
                headers["anthropic-beta"] = str(beta)
        return headers

    async def _post(
        self,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        client = self._ensure_client()
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await client.post(
                    f"{self._base_url}/v1/messages",
                    json=payload,
                    headers=extra_headers or {},
                )
                if resp.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                    delay = _BACKOFF_BASE * (2 ** attempt)
                    retry_after = resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    logger.warning(
                        "Anthropic %s (attempt %d/%d), retrying in %.1fs",
                        resp.status_code, attempt + 1, _MAX_RETRIES + 1, delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code >= 400:
                    raise _map_status_to_error(
                        resp.status_code, resp.text, self.name,
                    )
                return resp.json()

            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "Anthropic transport error (attempt %d/%d): %s",
                        attempt + 1, _MAX_RETRIES + 1, exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise NetworkError(
                    f"Anthropic transport error: {exc}",
                    provider=self.name,
                ) from exc

        raise NetworkError(
            f"Anthropic request failed after {_MAX_RETRIES + 1} attempts",
            provider=self.name,
        ) from last_exc

    # -- Payload construction -----------------------------------------------

    def _build_payload(
        self,
        request: Request,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        system_parts: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []

        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                block: dict[str, Any] = {"type": "text", "text": msg.text}
                # Inject cache_control on system blocks if requested.
                opts = request.provider_options or {}
                if opts.get("cache_system"):
                    block["cache_control"] = {"type": "ephemeral"}
                system_parts.append(block)
            else:
                messages.append(self._convert_message(msg, request))

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }

        if system_parts:
            payload["system"] = system_parts

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        else:
            payload["max_tokens"] = 8192  # Anthropic requires this field.

        if request.temperature is not None:
            payload["temperature"] = request.temperature

        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences

        if request.tools:
            payload["tools"] = [self._convert_tool(t) for t in request.tools]

        if request.tool_choice:
            payload["tool_choice"] = self._convert_tool_choice(request.tool_choice)

        if request.reasoning_effort:
            thinking = {"type": "enabled", "budget_tokens": self._effort_to_budget(request)}
            payload["thinking"] = thinking

        if stream:
            payload["stream"] = True

        return payload

    @staticmethod
    def _effort_to_budget(request: Request) -> int:
        """Map reasoning_effort string to a token budget."""
        mapping = {"low": 2048, "medium": 8192, "high": 32000}
        return mapping.get(request.reasoning_effort or "medium", 8192)

    def _convert_message(
        self, msg: Message, request: Request,
    ) -> dict[str, Any]:
        role = "user" if msg.role in (Role.USER, Role.TOOL) else "assistant"
        content: list[dict[str, Any]] = []

        for part in msg.content:
            if part.kind == ContentKind.TEXT and part.text is not None:
                content.append({"type": "text", "text": part.text})

            elif part.kind == ContentKind.IMAGE and part.image is not None:
                img = part.image
                if img.data is not None:
                    import base64
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.media_type or "image/png",
                            "data": base64.b64encode(img.data).decode(),
                        },
                    })
                elif img.url is not None:
                    content.append({
                        "type": "image",
                        "source": {"type": "url", "url": img.url},
                    })

            elif part.kind == ContentKind.TOOL_CALL and part.tool_call is not None:
                tc = part.tool_call
                args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": args,
                })

            elif part.kind == ContentKind.TOOL_RESULT and part.tool_result is not None:
                tr = part.tool_result
                result_content: str | list[dict[str, Any]]
                if isinstance(tr.content, dict):
                    result_content = json.dumps(tr.content)
                else:
                    result_content = str(tr.content)
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": result_content,
                }
                if tr.is_error:
                    block["is_error"] = True
                content.append(block)

            elif part.kind == ContentKind.THINKING and part.thinking is not None:
                block_t: dict[str, Any] = {
                    "type": "thinking",
                    "thinking": part.thinking.text,
                }
                if part.thinking.signature:
                    block_t["signature"] = part.thinking.signature
                content.append(block_t)

            elif part.kind == ContentKind.REDACTED_THINKING:
                content.append({"type": "redacted_thinking"})

        # Inject cache_control on the last user content block if requested.
        opts = request.provider_options or {}
        if opts.get("cache_messages") and content and role == "user":
            content[-1]["cache_control"] = {"type": "ephemeral"}

        return {"role": role, "content": content}

    @staticmethod
    def _convert_tool(tool: ToolDefinition) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    @staticmethod
    def _convert_tool_choice(choice: str) -> dict[str, Any]:
        if choice == "auto":
            return {"type": "auto"}
        if choice in ("none",):
            return {"type": "none"}
        if choice in ("required", "any"):
            return {"type": "any"}
        # Specific tool name.
        return {"type": "tool", "name": choice}

    # -- Response parsing ---------------------------------------------------

    def _parse_response(self, raw: dict[str, Any]) -> Response:
        content: list[ContentPart] = []

        for block in raw.get("content", []):
            btype = block.get("type")
            if btype == "text":
                content.append(ContentPart(kind=ContentKind.TEXT, text=block["text"]))
            elif btype == "tool_use":
                content.append(ContentPart(
                    kind=ContentKind.TOOL_CALL,
                    tool_call=ToolCallData(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    ),
                ))
            elif btype == "thinking":
                content.append(ContentPart(
                    kind=ContentKind.THINKING,
                    thinking=ThinkingData(
                        text=block.get("thinking", ""),
                        signature=block.get("signature"),
                    ),
                ))
            elif btype == "redacted_thinking":
                content.append(ContentPart(
                    kind=ContentKind.REDACTED_THINKING,
                    thinking=ThinkingData(redacted=True),
                ))

        usage_raw = raw.get("usage", {})
        usage = Usage(
            input_tokens=usage_raw.get("input_tokens", 0),
            output_tokens=usage_raw.get("output_tokens", 0),
            total_tokens=usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
            cache_read_tokens=usage_raw.get("cache_read_input_tokens"),
            cache_write_tokens=usage_raw.get("cache_creation_input_tokens"),
        )

        stop_reason = raw.get("stop_reason", "end_turn")
        finish = self._map_stop_reason(stop_reason)

        return Response(
            id=raw.get("id", ""),
            model=raw.get("model", ""),
            content=content,
            usage=usage,
            finish_reason=finish,
            provider_data=raw,
        )

    @staticmethod
    def _map_stop_reason(reason: str) -> FinishReason:
        mapping = {
            "end_turn": FinishReason.STOP,
            "stop_sequence": FinishReason.STOP,
            "tool_use": FinishReason.TOOL_CALLS,
            "max_tokens": FinishReason.LENGTH,
        }
        return mapping.get(reason, FinishReason.STOP)

    # -- SSE streaming ------------------------------------------------------

    async def _parse_sse_stream(
        self,
        http_resp: httpx.Response,
    ) -> AsyncIterator[StreamEvent]:
        """Parse Anthropic SSE events line-by-line."""
        event_type: str | None = None
        data_lines: list[str] = []

        async for line_bytes in http_resp.aiter_lines():
            line = line_bytes  # httpx aiter_lines yields str

            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
                continue

            if line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
                continue

            if line.strip() == "" and data_lines:
                # Event boundary.
                data_str = "\n".join(data_lines)
                data_lines = []
                if data_str == "[DONE]":
                    yield StreamEvent(kind=StreamEventKind.DONE)
                    return
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                async for evt in self._handle_sse_event(event_type, data):
                    yield evt
                event_type = None

        # Flush any remaining buffered data.
        if data_lines:
            data_str = "\n".join(data_lines)
            if data_str != "[DONE]":
                try:
                    data = json.loads(data_str)
                    async for evt in self._handle_sse_event(event_type, data):
                        yield evt
                except json.JSONDecodeError:
                    pass

    async def _handle_sse_event(
        self,
        event_type: str | None,
        data: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        etype = event_type or data.get("type", "")

        if etype == "message_start":
            # Contains initial usage.
            msg = data.get("message", {})
            usage_raw = msg.get("usage", {})
            if usage_raw:
                yield StreamEvent(
                    kind=StreamEventKind.USAGE,
                    usage=Usage(
                        input_tokens=usage_raw.get("input_tokens", 0),
                        output_tokens=usage_raw.get("output_tokens", 0),
                        total_tokens=usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
                        cache_read_tokens=usage_raw.get("cache_read_input_tokens"),
                        cache_write_tokens=usage_raw.get("cache_creation_input_tokens"),
                    ),
                )

        elif etype == "content_block_start":
            block = data.get("content_block", {})
            btype = block.get("type", "")
            if btype == "text":
                yield StreamEvent(kind=StreamEventKind.CONTENT_START)
            elif btype == "tool_use":
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_CALL_START,
                    data={"id": block.get("id", ""), "name": block.get("name", "")},
                )
            elif btype == "thinking":
                yield StreamEvent(kind=StreamEventKind.THINKING_START)

        elif etype == "content_block_delta":
            delta = data.get("delta", {})
            dtype = delta.get("type", "")
            if dtype == "text_delta":
                yield StreamEvent(
                    kind=StreamEventKind.CONTENT_DELTA,
                    content_part=ContentPart(kind=ContentKind.TEXT, text=delta.get("text", "")),
                )
            elif dtype == "input_json_delta":
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_CALL_DELTA,
                    data={"partial_json": delta.get("partial_json", "")},
                )
            elif dtype == "thinking_delta":
                yield StreamEvent(
                    kind=StreamEventKind.THINKING_DELTA,
                    content_part=ContentPart(
                        kind=ContentKind.THINKING,
                        thinking=ThinkingData(text=delta.get("thinking", "")),
                    ),
                )
            elif dtype == "signature_delta":
                yield StreamEvent(
                    kind=StreamEventKind.THINKING_DELTA,
                    data={"signature": delta.get("signature", "")},
                )

        elif etype == "content_block_stop":
            # We infer the end kind from the index but for simplicity emit
            # a generic CONTENT_END.  Callers track which block is active.
            yield StreamEvent(kind=StreamEventKind.CONTENT_END)

        elif etype == "message_delta":
            delta = data.get("delta", {})
            usage_raw = data.get("usage", {})
            finish = self._map_stop_reason(delta.get("stop_reason", "end_turn"))
            yield StreamEvent(
                kind=StreamEventKind.USAGE,
                usage=Usage(
                    output_tokens=usage_raw.get("output_tokens", 0),
                    total_tokens=usage_raw.get("output_tokens", 0),
                ),
                finish_reason=finish,
            )

        elif etype == "message_stop":
            yield StreamEvent(kind=StreamEventKind.DONE)

        elif etype == "error":
            yield StreamEvent(
                kind=StreamEventKind.ERROR,
                data=data.get("error", data),
            )
