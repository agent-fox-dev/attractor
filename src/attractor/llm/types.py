"""Unified LLM Client data model.

Implements the core types from the Unified LLM Client Specification.
All provider adapters translate to and from these canonical types.
"""

from __future__ import annotations

import threading
from enum import StrEnum
from typing import Any, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Role(StrEnum):
    """Who produced a message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class ContentKind(StrEnum):
    """Discriminator for ContentPart tagged union."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"


class FinishReason(StrEnum):
    """Why generation stopped."""

    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    OTHER = "other"


class StreamEventKind(StrEnum):
    """Discriminator for stream events."""

    CONTENT_START = "content_start"
    CONTENT_DELTA = "content_delta"
    CONTENT_END = "content_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_END = "thinking_end"
    STEP_FINISH = "step_finish"
    USAGE = "usage"
    DONE = "done"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Content data records
# ---------------------------------------------------------------------------


_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
}


class ImageData(BaseModel):
    """Image content — exactly one of *url* or *data* must be set."""

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    detail: str | None = None

    @classmethod
    def from_path(cls, path: str) -> "ImageData":
        """Load an image from a local file path and infer its MIME type."""
        import base64
        from pathlib import Path as _Path

        p = _Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        raw = p.read_bytes()
        ext = p.suffix.lower()
        mime = _MIME_BY_EXT.get(ext, "application/octet-stream")
        b64 = base64.b64encode(raw).decode("ascii")
        return cls(
            url=f"data:{mime};base64,{b64}",
            data=raw,
            media_type=mime,
        )


class AudioData(BaseModel):
    """Audio content."""

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None


class DocumentData(BaseModel):
    """Document content (PDF, etc.)."""

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    file_name: str | None = None


class ToolCallData(BaseModel):
    """A model-initiated tool invocation."""

    id: str
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)
    raw_arguments: str | None = None
    type: str = "function"


class ToolResultData(BaseModel):
    """Result of executing a tool call."""

    tool_call_id: str
    content: str | dict[str, Any] = ""
    is_error: bool = False
    image_data: bytes | None = None
    image_media_type: str | None = None


class ThinkingData(BaseModel):
    """Model reasoning / thinking content."""

    text: str = ""
    signature: str | None = None
    redacted: bool = False


# ---------------------------------------------------------------------------
# ContentPart — tagged union on ``kind``
# ---------------------------------------------------------------------------


class ContentPart(BaseModel):
    """A single part of a multimodal message, discriminated by *kind*."""

    kind: ContentKind | str
    text: str | None = None
    image: ImageData | None = None
    audio: AudioData | None = None
    document: DocumentData | None = None
    tool_call: ToolCallData | None = None
    tool_result: ToolResultData | None = None
    thinking: ThinkingData | None = None


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """The fundamental unit of conversation."""

    role: Role
    content: list[ContentPart] = Field(default_factory=list)
    name: str | None = None
    tool_call_id: str | None = None

    # -- convenience constructors ----------------------------------------

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(
            role=Role.SYSTEM,
            content=[ContentPart(kind=ContentKind.TEXT, text=text)],
        )

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(
            role=Role.USER,
            content=[ContentPart(kind=ContentKind.TEXT, text=text)],
        )

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(
            role=Role.ASSISTANT,
            content=[ContentPart(kind=ContentKind.TEXT, text=text)],
        )

    @classmethod
    def developer(cls, text: str) -> Message:
        """Create a developer-role message (used by some providers for system-like instructions)."""
        return cls(
            role=Role.DEVELOPER,
            content=[ContentPart(kind=ContentKind.TEXT, text=text)],
        )

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        content: str | dict[str, Any] = "",
        *,
        is_error: bool = False,
    ) -> Message:
        return cls(
            role=Role.TOOL,
            tool_call_id=tool_call_id,
            content=[
                ContentPart(
                    kind=ContentKind.TOOL_RESULT,
                    tool_result=ToolResultData(
                        tool_call_id=tool_call_id,
                        content=content,
                        is_error=is_error,
                    ),
                )
            ],
        )

    # -- convenience properties ------------------------------------------

    @property
    def text(self) -> str:
        """Concatenate all TEXT content parts."""
        return "".join(
            part.text for part in self.content
            if part.kind == ContentKind.TEXT and part.text is not None
        )


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


class ToolDefinition(BaseModel):
    """Schema describing a tool the model may call."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


# Public alias used in high-level API signatures.
Tool = ToolDefinition


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class Usage(BaseModel):
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    raw: dict[str, Any] | None = None

    def __add__(self, other: Usage) -> Usage:
        def _sum_optional(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=_sum_optional(self.reasoning_tokens, other.reasoning_tokens),
            cache_read_tokens=_sum_optional(self.cache_read_tokens, other.cache_read_tokens),
            cache_write_tokens=_sum_optional(self.cache_write_tokens, other.cache_write_tokens),
            raw=other.raw or self.raw,
        )


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class AbortSignal:
    """Cooperative cancellation signal for LLM requests.

    Usage::

        controller = AbortController()
        # Pass controller.signal to generate() or stream()
        # Later: controller.abort() to cancel
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    def _abort(self) -> None:
        self._event.set()

    def check(self) -> None:
        """Raise AbortError if the signal has been triggered."""
        if self._event.is_set():
            raise AbortError("Request aborted via signal")


class AbortController:
    """Creates and controls an AbortSignal."""

    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self) -> None:
        """Signal cancellation."""
        self.signal._abort()


class Request(BaseModel):
    """Input for both ``complete()`` and ``stream()``."""

    model: str
    messages: list[Message] = Field(default_factory=list)
    tools: list[ToolDefinition] | None = None
    tool_choice: str | ToolChoice | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None
    provider: str | None = None
    provider_options: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None
    response_format: ResponseFormat | dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    abort_signal: AbortSignal | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class Response(BaseModel):
    """Result of a blocking ``complete()`` call."""

    id: str = ""
    model: str = ""
    provider: str = ""
    content: list[ContentPart] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    finish_reason: FinishReason = FinishReason.STOP
    raw: dict[str, Any] | None = None
    provider_data: dict[str, Any] | None = None  # Alias for raw (backwards compat)
    warnings: list[ResponseWarning] = Field(default_factory=list)
    rate_limit: RateLimitInfo | None = None

    # -- convenience properties ------------------------------------------

    @property
    def text(self) -> str:
        """Concatenate all TEXT content parts."""
        return "".join(
            part.text for part in self.content
            if part.kind == ContentKind.TEXT and part.text is not None
        )

    @property
    def tool_calls(self) -> list[ToolCallData]:
        """Extract all tool-call parts."""
        return [
            part.tool_call for part in self.content
            if part.kind == ContentKind.TOOL_CALL and part.tool_call is not None
        ]

    @property
    def reasoning(self) -> str:
        """Concatenate all THINKING content parts."""
        return "".join(
            part.thinking.text for part in self.content
            if part.kind == ContentKind.THINKING and part.thinking and part.thinking.text
        )

    @property
    def message(self) -> "Message":
        """Return the assistant's response as a Message object."""
        return Message(role=Role.ASSISTANT, content=self.content)


# ---------------------------------------------------------------------------
# StreamEvent
# ---------------------------------------------------------------------------


class StreamEvent(BaseModel):
    """A single event from a streaming response."""

    kind: StreamEventKind | str
    data: dict[str, Any] | None = None
    content_part: ContentPart | None = None
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
    delta: str | None = None
    reasoning_delta: str | None = None
    error: str | None = None
    raw: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    """Catalog entry describing a known model."""

    id: str
    provider: str
    display_name: str = ""
    context_window: int = 0
    max_output: int | None = None
    supports_tools: bool = False
    supports_vision: bool = False
    supports_reasoning: bool = False
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    aliases: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# TimeoutConfig
# ---------------------------------------------------------------------------


class TimeoutConfig(BaseModel):
    """Timeout settings for LLM calls."""

    total: float | None = None
    per_step: float | None = None


class AdapterTimeout(BaseModel):
    """Granular timeout settings for provider adapters (Section 4.7)."""

    connect: float = 10.0
    request: float = 300.0
    stream_read: float = 60.0


# ---------------------------------------------------------------------------
# ProviderError
# ---------------------------------------------------------------------------


class ResponseWarning(BaseModel):
    """A non-fatal issue reported in a response."""

    code: str = ""
    message: str = ""


class RateLimitInfo(BaseModel):
    """Rate-limit information parsed from response headers."""

    requests_remaining: int | None = None
    requests_limit: int | None = None
    tokens_remaining: int | None = None
    tokens_limit: int | None = None
    reset_at: str | None = None


class ResponseFormat(BaseModel):
    """Requested response format."""

    type: str = "text"
    json_schema: dict[str, Any] | None = None
    strict: bool = False


class RetryPolicy(BaseModel):
    """Retry configuration for high-level API calls."""

    max_retries: int = 2
    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True


class ToolChoice(BaseModel):
    """Structured tool-choice specification."""

    mode: str = "auto"
    tool_name: str | None = None


# ---------------------------------------------------------------------------
# Error hierarchy (Section 6.1)
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base error raised by a provider adapter."""

    retryable: bool = False
    retry_after: float | None = None

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider: str | None = None,
        raw: Any | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider
        self.raw = raw
        self.retryable = retryable
        self.retry_after = retry_after
        self.error_code = error_code


class AuthenticationError(ProviderError):
    """401 — invalid or missing API key."""

    def __init__(self, message: str = "Authentication failed", **kw: Any) -> None:
        super().__init__(message, status_code=401, retryable=False, **kw)


class AccessDeniedError(ProviderError):
    """403 — valid key but insufficient permissions."""

    def __init__(self, message: str = "Access denied", **kw: Any) -> None:
        super().__init__(message, status_code=403, retryable=False, **kw)


class NotFoundError(ProviderError):
    """404 — model or resource not found."""

    def __init__(self, message: str = "Not found", **kw: Any) -> None:
        super().__init__(message, status_code=404, retryable=False, **kw)


class InvalidRequestError(ProviderError):
    """400/422 — malformed request."""

    def __init__(self, message: str = "Invalid request", **kw: Any) -> None:
        super().__init__(message, status_code=400, retryable=False, **kw)


class RateLimitError(ProviderError):
    """429 — rate limited."""

    def __init__(self, message: str = "Rate limited", retry_after: float | None = None, **kw: Any) -> None:
        super().__init__(message, status_code=429, retryable=True, retry_after=retry_after, **kw)


class ServerError(ProviderError):
    """5xx — provider server error."""

    def __init__(self, message: str = "Server error", status_code: int = 500, **kw: Any) -> None:
        super().__init__(message, status_code=status_code, retryable=True, **kw)


class ContentFilterError(ProviderError):
    """Content was blocked by safety filters."""

    def __init__(self, message: str = "Content filtered", **kw: Any) -> None:
        super().__init__(message, retryable=False, **kw)


class ContextLengthError(ProviderError):
    """413 — input exceeds model context window."""

    def __init__(self, message: str = "Context length exceeded", **kw: Any) -> None:
        super().__init__(message, status_code=413, retryable=False, **kw)


class QuotaExceededError(ProviderError):
    """Billing quota or usage limit exceeded."""

    def __init__(self, message: str = "Quota exceeded", **kw: Any) -> None:
        super().__init__(message, retryable=False, **kw)


class RequestTimeoutError(ProviderError):
    """408 — request timed out."""

    def __init__(self, message: str = "Request timed out", **kw: Any) -> None:
        super().__init__(message, status_code=408, retryable=True, **kw)


class AbortError(ProviderError):
    """Request was cancelled via abort signal."""

    def __init__(self, message: str = "Request aborted", **kw: Any) -> None:
        super().__init__(message, retryable=False, **kw)


class NetworkError(ProviderError):
    """Connection or DNS failure."""

    def __init__(self, message: str = "Network error", **kw: Any) -> None:
        super().__init__(message, retryable=True, **kw)


class StreamError(ProviderError):
    """Error during SSE stream processing."""

    def __init__(self, message: str = "Stream error", **kw: Any) -> None:
        super().__init__(message, retryable=True, **kw)


class InvalidToolCallError(ProviderError):
    """Model produced an invalid tool call (bad JSON, unknown tool, etc.)."""

    def __init__(self, message: str = "Invalid tool call", **kw: Any) -> None:
        super().__init__(message, retryable=False, **kw)


class UnsupportedToolChoiceError(ProviderError):
    """Adapter does not support the requested tool_choice mode."""

    def __init__(self, message: str = "Unsupported tool choice mode", **kw: Any) -> None:
        super().__init__(message, retryable=False, **kw)


class NoObjectGeneratedError(ProviderError):
    """generate_object() failed to produce a valid object."""

    def __init__(self, message: str = "No object generated", **kw: Any) -> None:
        super().__init__(message, retryable=False, **kw)


class ConfigurationError(ProviderError):
    """Client misconfiguration (missing keys, bad adapter setup, etc.)."""

    def __init__(self, message: str = "Configuration error", **kw: Any) -> None:
        super().__init__(message, retryable=False, **kw)
