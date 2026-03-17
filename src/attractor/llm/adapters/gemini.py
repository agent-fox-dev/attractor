"""Google Gemini API adapter.

Translates unified :class:`Request` / :class:`Response` types to and from
the Gemini ``/v1beta/models/*/generateContent`` wire format, including
streaming via SSE.
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
    msg = f"Gemini API error {status_code}: {body}"
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


class GeminiAdapter(ProviderAdapter):
    """Adapter for the Google Gemini API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout: float = 300.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    # -- ProviderAdapter interface ------------------------------------------

    @property
    def name(self) -> str:
        return "gemini"

    async def initialize(self) -> None:
        self._client = self._make_client()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(self, request: Request) -> Response:
        payload = self._build_payload(request)
        url = self._url(request.model, stream=False)
        raw = await self._post(url, payload)
        return self._parse_response(raw)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        payload = self._build_payload(request)
        url = self._url(request.model, stream=True)
        client = self._ensure_client()

        async with client.stream(
            "POST",
            url,
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
        return mode in {"auto", "none"}

    # -- HTTP helpers -------------------------------------------------------

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            timeout=self._timeout,
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _url(self, model: str, *, stream: bool) -> str:
        action = "streamGenerateContent" if stream else "generateContent"
        url = f"{self._base_url}/v1beta/models/{model}:{action}?key={self._api_key}"
        if stream:
            url += "&alt=sse"
        return url

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._ensure_client()
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await client.post(url, json=payload)

                if resp.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                    delay = _BACKOFF_BASE * (2 ** attempt)
                    retry_after = resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    logger.warning(
                        "Gemini %s (attempt %d/%d), retrying in %.1fs",
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
                        "Gemini transport error (attempt %d/%d): %s",
                        attempt + 1, _MAX_RETRIES + 1, exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise NetworkError(
                    f"Gemini transport error: {exc}",
                    provider=self.name,
                ) from exc

        raise NetworkError(
            f"Gemini request failed after {_MAX_RETRIES + 1} attempts",
            provider=self.name,
        ) from last_exc

    # -- Payload construction -----------------------------------------------

    def _build_payload(self, request: Request) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        system_instruction: dict[str, Any] | None = None

        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                system_instruction = {
                    "parts": [{"text": msg.text}],
                }
                continue
            contents.append(self._convert_message(msg))

        payload: dict[str, Any] = {"contents": contents}

        if system_instruction:
            payload["systemInstruction"] = system_instruction

        # Generation config.
        gen_config: dict[str, Any] = {}
        if request.max_tokens is not None:
            gen_config["maxOutputTokens"] = request.max_tokens
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.top_p is not None:
            gen_config["topP"] = request.top_p
        if request.stop_sequences:
            gen_config["stopSequences"] = request.stop_sequences
        if request.response_format:
            # Handle both dict and ResponseFormat model
            fmt = request.response_format
            if hasattr(fmt, 'type'):
                fmt_type = fmt.type
                fmt_schema = getattr(fmt, 'json_schema', None)
            else:
                fmt_type = fmt.get("type", "")
                fmt_schema = fmt.get("json_schema")
            if fmt_type == "json_object":
                gen_config["responseMimeType"] = "application/json"
            elif fmt_type == "json_schema":
                gen_config["responseMimeType"] = "application/json"
                if isinstance(fmt_schema, dict):
                    schema = fmt_schema.get("schema", fmt_schema)
                    gen_config["responseSchema"] = schema
        if request.reasoning_effort:
            thinking = {"thinkingBudget": self._effort_to_budget(request)}
            gen_config["thinkingConfig"] = thinking
        if gen_config:
            payload["generationConfig"] = gen_config

        # Tools.
        if request.tools:
            payload["tools"] = [{
                "functionDeclarations": [
                    self._convert_tool(t) for t in request.tools
                ],
            }]
            if request.tool_choice:
                payload["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": self._map_tool_choice(request.tool_choice),
                    },
                }

        return payload

    @staticmethod
    def _effort_to_budget(request: Request) -> int:
        mapping = {"low": 1024, "medium": 4096, "high": 16384}
        return mapping.get(request.reasoning_effort or "medium", 4096)

    @staticmethod
    def _convert_message(msg: Message) -> dict[str, Any]:
        role = "user" if msg.role in (Role.USER, Role.TOOL) else "model"
        parts: list[dict[str, Any]] = []

        for part in msg.content:
            if part.kind == ContentKind.TEXT and part.text is not None:
                parts.append({"text": part.text})

            elif part.kind == ContentKind.IMAGE and part.image is not None:
                img = part.image
                if img.data is not None:
                    import base64
                    parts.append({
                        "inlineData": {
                            "mimeType": img.media_type or "image/png",
                            "data": base64.b64encode(img.data).decode(),
                        },
                    })
                elif img.url:
                    parts.append({
                        "fileData": {
                            "mimeType": img.media_type or "image/png",
                            "fileUri": img.url,
                        },
                    })

            elif part.kind == ContentKind.TOOL_CALL and part.tool_call is not None:
                tc = part.tool_call
                args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
                parts.append({
                    "functionCall": {
                        "name": tc.name,
                        "args": args,
                    },
                })

            elif part.kind == ContentKind.TOOL_RESULT and part.tool_result is not None:
                tr = part.tool_result
                response_content: dict[str, Any]
                if isinstance(tr.content, dict):
                    response_content = tr.content
                else:
                    response_content = {"result": str(tr.content)}
                parts.append({
                    "functionResponse": {
                        "name": "",  # Gemini requires name but we may not have it.
                        "response": response_content,
                    },
                })

            elif part.kind == ContentKind.THINKING and part.thinking is not None:
                parts.append({
                    "thought": True,
                    "text": part.thinking.text,
                })

        return {"role": role, "parts": parts}

    @staticmethod
    def _convert_tool(tool: ToolDefinition) -> dict[str, Any]:
        decl: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
        }
        if tool.parameters:
            decl["parameters"] = tool.parameters
        return decl

    @staticmethod
    def _map_tool_choice(choice: str) -> str:
        mapping = {
            "auto": "AUTO",
            "none": "NONE",
            "required": "ANY",
            "any": "ANY",
        }
        return mapping.get(choice, "AUTO")

    # -- Response parsing ---------------------------------------------------

    def _parse_response(self, raw: dict[str, Any]) -> Response:
        content: list[ContentPart] = []

        candidates = raw.get("candidates", [])
        finish_reason_str = "STOP"

        if candidates:
            candidate = candidates[0]
            finish_reason_str = candidate.get("finishReason", "STOP")
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part and not part.get("thought"):
                    content.append(ContentPart(kind=ContentKind.TEXT, text=part["text"]))
                elif "text" in part and part.get("thought"):
                    content.append(ContentPart(
                        kind=ContentKind.THINKING,
                        thinking=ThinkingData(text=part["text"]),
                    ))
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    content.append(ContentPart(
                        kind=ContentKind.TOOL_CALL,
                        tool_call=ToolCallData(
                            id=fc.get("name", ""),  # Gemini uses name as id
                            name=fc.get("name", ""),
                            arguments=fc.get("args", {}),
                        ),
                    ))

        usage_raw = raw.get("usageMetadata", {})
        usage = Usage(
            input_tokens=usage_raw.get("promptTokenCount", 0),
            output_tokens=usage_raw.get("candidatesTokenCount", 0),
            total_tokens=usage_raw.get("totalTokenCount", 0),
            reasoning_tokens=usage_raw.get("thoughtsTokenCount"),
        )

        finish = self._map_finish_reason(finish_reason_str, content)

        return Response(
            id="",  # Gemini does not return a response id.
            model="",
            provider=self.name,
            content=content,
            usage=usage,
            finish_reason=finish,
            provider_data=raw,
        )

    @staticmethod
    def _map_finish_reason(
        reason: str, content: list[ContentPart],
    ) -> FinishReason:
        if any(p.kind == ContentKind.TOOL_CALL for p in content):
            return FinishReason.TOOL_CALLS
        mapping = {
            "STOP": FinishReason.STOP,
            "MAX_TOKENS": FinishReason.LENGTH,
            "SAFETY": FinishReason.CONTENT_FILTER,
            "RECITATION": FinishReason.CONTENT_FILTER,
        }
        return mapping.get(reason, FinishReason.STOP)

    # -- SSE streaming ------------------------------------------------------

    async def _parse_sse_stream(
        self,
        http_resp: httpx.Response,
    ) -> AsyncIterator[StreamEvent]:
        """Parse Gemini SSE stream (``alt=sse`` format).

        Gemini streams one JSON object per SSE ``data:`` line, where each
        object is a complete ``GenerateContentResponse`` chunk.
        """
        data_lines: list[str] = []
        started = False

        async for line in http_resp.aiter_lines():
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

                async for evt in self._handle_chunk(data, started):
                    yield evt
                started = True

        # Flush remaining.
        if data_lines:
            data_str = "\n".join(data_lines)
            if data_str != "[DONE]":
                try:
                    data = json.loads(data_str)
                    async for evt in self._handle_chunk(data, started):
                        yield evt
                except json.JSONDecodeError:
                    pass

        yield StreamEvent(kind=StreamEventKind.DONE)

    async def _handle_chunk(
        self,
        data: dict[str, Any],
        started: bool,
    ) -> AsyncIterator[StreamEvent]:
        candidates = data.get("candidates", [])
        if not candidates:
            # May be a usage-only chunk.
            usage_raw = data.get("usageMetadata", {})
            if usage_raw:
                yield StreamEvent(
                    kind=StreamEventKind.USAGE,
                    usage=Usage(
                        input_tokens=usage_raw.get("promptTokenCount", 0),
                        output_tokens=usage_raw.get("candidatesTokenCount", 0),
                        total_tokens=usage_raw.get("totalTokenCount", 0),
                        reasoning_tokens=usage_raw.get("thoughtsTokenCount"),
                    ),
                )
            return

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])

        for part in parts:
            if "text" in part and not part.get("thought"):
                if not started:
                    yield StreamEvent(kind=StreamEventKind.CONTENT_START)
                text = part["text"]
                yield StreamEvent(
                    kind=StreamEventKind.CONTENT_DELTA,
                    data={"text": text},
                    content_part=ContentPart(kind=ContentKind.TEXT, text=text),
                )

            elif "text" in part and part.get("thought"):
                yield StreamEvent(
                    kind=StreamEventKind.THINKING_DELTA,
                    content_part=ContentPart(
                        kind=ContentKind.THINKING,
                        thinking=ThinkingData(text=part["text"]),
                    ),
                )

            elif "functionCall" in part:
                fc = part["functionCall"]
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_CALL_START,
                    data={"id": fc.get("name", ""), "name": fc.get("name", "")},
                )
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_CALL_END,
                    content_part=ContentPart(
                        kind=ContentKind.TOOL_CALL,
                        tool_call=ToolCallData(
                            id=fc.get("name", ""),
                            name=fc.get("name", ""),
                            arguments=fc.get("args", {}),
                        ),
                    ),
                )

        # Check for finish reason.
        finish_str = candidate.get("finishReason")
        if finish_str:
            content_parts = [
                ContentPart(
                    kind=ContentKind.TOOL_CALL,
                    tool_call=ToolCallData(
                        id=p["functionCall"].get("name", ""),
                        name=p["functionCall"].get("name", ""),
                        arguments=p["functionCall"].get("args", {}),
                    ),
                )
                for p in parts
                if "functionCall" in p
            ]
            finish = self._map_finish_reason(finish_str, content_parts)
            yield StreamEvent(
                kind=StreamEventKind.CONTENT_END,
                finish_reason=finish,
            )

        # Usage metadata in this chunk.
        usage_raw = data.get("usageMetadata", {})
        if usage_raw:
            yield StreamEvent(
                kind=StreamEventKind.USAGE,
                usage=Usage(
                    input_tokens=usage_raw.get("promptTokenCount", 0),
                    output_tokens=usage_raw.get("candidatesTokenCount", 0),
                    total_tokens=usage_raw.get("totalTokenCount", 0),
                    reasoning_tokens=usage_raw.get("thoughtsTokenCount"),
                ),
            )
