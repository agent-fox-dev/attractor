"""OpenAI Responses API adapter.

Translates unified :class:`Request` / :class:`Response` types to and from
the OpenAI ``/v1/responses`` wire format, including streaming via SSE.
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
    AccessDeniedError,
    AuthenticationError,
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
    Response,
    Role,
    ServerError,
    StreamEvent,
    StreamEventKind,
    ThinkingData,
    ToolCallData,
    ToolDefinition,
    Usage,
)

logger = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503}
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0


def _map_status_to_error(status_code: int, body: str, provider: str) -> ProviderError:
    msg = f"OpenAI API error {status_code}: {body}"
    kw: dict = {"provider": provider, "raw": body}
    if status_code == 401:
        return AuthenticationError(msg, **kw)
    if status_code == 403:
        return AccessDeniedError(msg, **kw)
    if status_code == 404:
        return NotFoundError(msg, **kw)
    if status_code in (400, 422):
        return InvalidRequestError(msg, **kw)
    if status_code == 413:
        return ContextLengthError(msg, **kw)
    if status_code == 429:
        return RateLimitError(msg, **kw)
    if status_code >= 500:
        return ServerError(msg, status_code=status_code, **kw)
    return ProviderError(msg, status_code=status_code, **kw)


class OpenAIAdapter(ProviderAdapter):
    """Adapter for the OpenAI Responses API (``/v1/responses``)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com",
        default_headers: dict[str, str] | None = None,
        timeout: float = 300.0,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_headers = default_headers or {}
        if organization:
            self._default_headers["OpenAI-Organization"] = organization
        if project:
            self._default_headers["OpenAI-Project"] = project
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    # -- ProviderAdapter interface ------------------------------------------

    @property
    def name(self) -> str:
        return "openai"

    async def initialize(self) -> None:
        self._client = self._make_client()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(self, request: Request) -> Response:
        payload = self._build_payload(request, stream=False)
        raw = await self._post(payload)
        return self._parse_response(raw)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        payload = self._build_payload(request, stream=True)
        client = self._ensure_client()

        async with client.stream(
            "POST",
            f"{self._base_url}/v1/responses",
            json=payload,
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
        return mode in {"auto", "none", "required"}

    # -- HTTP helpers -------------------------------------------------------

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                **self._default_headers,
            },
            timeout=self._timeout,
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._ensure_client()
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await client.post(
                    f"{self._base_url}/v1/responses",
                    json=payload,
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
                        "OpenAI %s (attempt %d/%d), retrying in %.1fs",
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
                        "OpenAI transport error (attempt %d/%d): %s",
                        attempt + 1, _MAX_RETRIES + 1, exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise NetworkError(
                    f"OpenAI transport error: {exc}",
                    provider=self.name,
                ) from exc

        raise NetworkError(
            f"OpenAI request failed after {_MAX_RETRIES + 1} attempts",
            provider=self.name,
        ) from last_exc

    # -- Payload construction -----------------------------------------------

    def _build_payload(
        self,
        request: Request,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        input_items: list[dict[str, Any]] = []
        instructions: str | None = None

        for msg in request.messages:
            if msg.role == Role.SYSTEM or msg.role == Role.DEVELOPER:
                # Responses API uses "instructions" for system-level prompts.
                instructions = msg.text
                continue
            input_items.append(self._convert_message(msg))

        payload: dict[str, Any] = {
            "model": request.model,
            "input": input_items,
        }

        if instructions:
            payload["instructions"] = instructions

        if request.max_tokens is not None:
            payload["max_output_tokens"] = request.max_tokens

        if request.temperature is not None:
            payload["temperature"] = request.temperature

        if request.tools:
            payload["tools"] = [self._convert_tool(t) for t in request.tools]

        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        if request.reasoning_effort:
            payload["reasoning"] = {"effort": request.reasoning_effort}

        if stream:
            payload["stream"] = True

        return payload

    @staticmethod
    def _convert_message(msg: Message) -> dict[str, Any]:
        role = msg.role.value
        if role == "tool":
            role = "user"

        # Build content for the Responses API input items.
        # For simple text messages, we can use the shorthand.
        parts: list[dict[str, Any]] = []

        for part in msg.content:
            if part.kind == ContentKind.TEXT and part.text is not None:
                parts.append({"type": "input_text", "text": part.text})

            elif part.kind == ContentKind.IMAGE and part.image is not None:
                img = part.image
                if img.url:
                    parts.append({
                        "type": "input_image",
                        "image_url": img.url,
                        "detail": img.detail or "auto",
                    })
                elif img.data is not None:
                    import base64
                    data_url = f"data:{img.media_type or 'image/png'};base64,{base64.b64encode(img.data).decode()}"
                    parts.append({
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": img.detail or "auto",
                    })

            elif part.kind == ContentKind.TOOL_CALL and part.tool_call is not None:
                tc = part.tool_call
                args = tc.arguments if isinstance(tc.arguments, str) else json.dumps(tc.arguments)
                parts.append({
                    "type": "function_call",
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": args,
                })

            elif part.kind == ContentKind.TOOL_RESULT and part.tool_result is not None:
                tr = part.tool_result
                output = tr.content if isinstance(tr.content, str) else json.dumps(tr.content)
                parts.append({
                    "type": "function_call_output",
                    "call_id": tr.tool_call_id,
                    "output": output,
                })

        # Responses API items use "role" at the item level.
        if role == "assistant":
            # Assistant messages may contain function_call items which are
            # top-level items in Responses API, not nested.
            result: list[dict[str, Any]] = []
            text_parts = [p for p in parts if p["type"] == "input_text"]
            call_parts = [p for p in parts if p["type"] == "function_call"]
            if text_parts:
                result.append({"type": "message", "role": "assistant", "content": text_parts})
            for cp in call_parts:
                result.append(cp)
            return result[0] if len(result) == 1 else result  # type: ignore[return-value]

        if any(p["type"] == "function_call_output" for p in parts):
            # function_call_output is a top-level item.
            result_items: list[dict[str, Any]] = []
            for p in parts:
                if p["type"] == "function_call_output":
                    result_items.append(p)
                else:
                    result_items.append({"type": "message", "role": role, "content": [p]})
            return result_items[0] if len(result_items) == 1 else result_items  # type: ignore[return-value]

        return {"type": "message", "role": role, "content": parts}

    @staticmethod
    def _convert_tool(tool: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }

    # -- Response parsing ---------------------------------------------------

    def _parse_response(self, raw: dict[str, Any]) -> Response:
        content: list[ContentPart] = []

        # Responses API returns an "output" list of items.
        for item in raw.get("output", []):
            item_type = item.get("type", "")

            if item_type == "message":
                for part in item.get("content", []):
                    ptype = part.get("type", "")
                    if ptype == "output_text":
                        content.append(ContentPart(kind=ContentKind.TEXT, text=part.get("text", "")))

            elif item_type == "function_call":
                args_str = item.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = args_str
                content.append(ContentPart(
                    kind=ContentKind.TOOL_CALL,
                    tool_call=ToolCallData(
                        id=item.get("call_id", item.get("id", "")),
                        name=item.get("name", ""),
                        arguments=args,
                    ),
                ))

            elif item_type == "reasoning":
                for summary in item.get("summary", []):
                    if summary.get("type") == "summary_text":
                        content.append(ContentPart(
                            kind=ContentKind.THINKING,
                            thinking=ThinkingData(text=summary.get("text", "")),
                        ))

        usage_raw = raw.get("usage", {})
        reasoning_tokens = usage_raw.get("output_tokens_details", {}).get("reasoning_tokens")
        usage = Usage(
            input_tokens=usage_raw.get("input_tokens", 0),
            output_tokens=usage_raw.get("output_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
            reasoning_tokens=reasoning_tokens,
        )

        # Determine finish reason.
        status = raw.get("status", "completed")
        finish = self._map_status(status, content)

        return Response(
            id=raw.get("id", ""),
            model=raw.get("model", ""),
            content=content,
            usage=usage,
            finish_reason=finish,
            provider_data=raw,
        )

    @staticmethod
    def _map_status(status: str, content: list[ContentPart]) -> FinishReason:
        if any(p.kind == ContentKind.TOOL_CALL for p in content):
            return FinishReason.TOOL_CALLS
        mapping = {
            "completed": FinishReason.STOP,
            "failed": FinishReason.ERROR,
            "incomplete": FinishReason.LENGTH,
        }
        return mapping.get(status, FinishReason.STOP)

    # -- SSE streaming ------------------------------------------------------

    async def _parse_sse_stream(
        self,
        http_resp: httpx.Response,
    ) -> AsyncIterator[StreamEvent]:
        """Parse OpenAI Responses SSE events."""
        event_type: str | None = None
        data_lines: list[str] = []

        async for line in http_resp.aiter_lines():
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
                continue

            if line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
                continue

            if line.strip() == "" and data_lines:
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

        # Flush remaining.
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

        if etype == "response.output_text.delta":
            yield StreamEvent(
                kind=StreamEventKind.CONTENT_DELTA,
                content_part=ContentPart(
                    kind=ContentKind.TEXT,
                    text=data.get("delta", ""),
                ),
            )

        elif etype == "response.output_text.done":
            yield StreamEvent(kind=StreamEventKind.CONTENT_END)

        elif etype == "response.function_call_arguments.delta":
            yield StreamEvent(
                kind=StreamEventKind.TOOL_CALL_DELTA,
                data={
                    "partial_json": data.get("delta", ""),
                    "call_id": data.get("call_id", ""),
                    "name": data.get("name", ""),
                },
            )

        elif etype == "response.function_call_arguments.done":
            # Emit a completed tool call event.
            args_str = data.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = args_str
            yield StreamEvent(
                kind=StreamEventKind.TOOL_CALL_END,
                content_part=ContentPart(
                    kind=ContentKind.TOOL_CALL,
                    tool_call=ToolCallData(
                        id=data.get("call_id", data.get("item_id", "")),
                        name=data.get("name", ""),
                        arguments=args,
                    ),
                ),
            )

        elif etype == "response.content_part.added":
            part = data.get("part", {})
            if part.get("type") == "output_text":
                yield StreamEvent(kind=StreamEventKind.CONTENT_START)
            elif part.get("type") == "function_call":
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_CALL_START,
                    data={"id": part.get("call_id", ""), "name": part.get("name", "")},
                )

        elif etype == "response.reasoning_summary_text.delta":
            yield StreamEvent(
                kind=StreamEventKind.THINKING_DELTA,
                content_part=ContentPart(
                    kind=ContentKind.THINKING,
                    thinking=ThinkingData(text=data.get("delta", "")),
                ),
            )

        elif etype == "response.completed":
            resp_data = data.get("response", {})
            usage_raw = resp_data.get("usage", {})
            reasoning_tokens = usage_raw.get("output_tokens_details", {}).get("reasoning_tokens")
            yield StreamEvent(
                kind=StreamEventKind.USAGE,
                usage=Usage(
                    input_tokens=usage_raw.get("input_tokens", 0),
                    output_tokens=usage_raw.get("output_tokens", 0),
                    total_tokens=usage_raw.get("total_tokens", 0),
                    reasoning_tokens=reasoning_tokens,
                ),
            )
            yield StreamEvent(kind=StreamEventKind.DONE)

        elif etype in ("response.failed", "error"):
            yield StreamEvent(
                kind=StreamEventKind.ERROR,
                data=data,
            )
