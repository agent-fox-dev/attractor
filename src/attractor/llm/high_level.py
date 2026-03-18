"""High-level convenience API for the Unified LLM Client.

Provides module-level functions that lazily initialise a default
:class:`Client` from environment variables, making it easy to get
started without explicit wiring::

    from attractor.llm.high_level import generate

    response = await generate("claude-sonnet-4-5", "Explain monads in one sentence.")
    print(response.text)
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from attractor.llm.client import Client
from attractor.llm.types import (
    AbortSignal,
    ContentKind,
    ContentPart,
    FinishReason,
    Message,
    NoObjectGeneratedError,
    ProviderError,
    Request,
    Response,
    ResponseFormat,
    RetryPolicy,
    Role,
    StreamEvent,
    StreamEventKind,
    TimeoutConfig,
    ToolCallData,
    ToolDefinition,
    ToolResultData,
    Usage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level default client (lazy, thread-safe init)
# ---------------------------------------------------------------------------

_default_client: Client | None = None
_lock = threading.Lock()


def _get_default_client() -> Client:
    global _default_client
    if _default_client is None:
        with _lock:
            if _default_client is None:
                _default_client = Client.from_env()
    return _default_client


def set_default_client(client: Client) -> None:
    """Replace the module-level default client."""
    global _default_client
    with _lock:
        _default_client = client


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result of a single LLM call within a multi-step generate()."""

    response: Response
    tool_calls: list[ToolCallData] = field(default_factory=list)
    tool_results: list[ToolResultData] = field(default_factory=list)


@dataclass
class GenerateResult:
    """Wraps the final response with aggregated metadata from multi-step execution."""

    response: Response
    steps: list[StepResult] = field(default_factory=list)
    total_usage: Usage = field(default_factory=Usage)

    @property
    def text(self) -> str:
        return self.response.text

    @property
    def tool_calls(self) -> list[ToolCallData]:
        return self.response.tool_calls

    @property
    def reasoning(self) -> str:
        return self.response.reasoning

    @property
    def tool_results(self) -> list[ToolResultData]:
        """All tool results across all steps."""
        results: list[ToolResultData] = []
        for step in self.steps:
            results.extend(step.tool_results)
        return results

    @property
    def finish_reason(self) -> FinishReason:
        return self.response.finish_reason


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------

StopCondition = Callable[[Response, list[StepResult]], bool]


def stop_on_text(response: Response, steps: list[StepResult]) -> bool:
    """Default stop condition: stop when there are no tool calls."""
    return len(response.tool_calls) == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_messages(
    prompt_or_messages: str | list[Message],
) -> list[Message]:
    """Normalise a bare string into a single-user-message list."""
    if isinstance(prompt_or_messages, str):
        return [Message.user(prompt_or_messages)]
    return prompt_or_messages


def _coerce_tools(
    tools: list[ToolDefinition | dict[str, Any]] | None,
) -> list[ToolDefinition] | None:
    if tools is None:
        return None
    result: list[ToolDefinition] = []
    for t in tools:
        if isinstance(t, ToolDefinition):
            result.append(t)
        else:
            result.append(ToolDefinition(**t))
    return result


async def _retry_call(
    fn: Callable[[], Any],
    policy: RetryPolicy,
) -> Any:
    """Execute *fn* with retry logic per the RetryPolicy."""
    delay = policy.initial_delay
    last_error: Exception | None = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await fn()
        except ProviderError as exc:
            last_error = exc
            if not exc.retryable or attempt >= policy.max_retries:
                raise

            # Use retry_after from error if available
            wait = exc.retry_after if exc.retry_after else delay
            if policy.jitter:
                wait *= 0.5 + random.random()
            wait = min(wait, policy.max_delay)

            logger.info(
                "Retry %d/%d after %.1fs: %s",
                attempt + 1, policy.max_retries, wait, exc,
            )
            await asyncio.sleep(wait)
            delay = min(delay * policy.multiplier, policy.max_delay)

    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


ToolExecutor = Callable[[ToolCallData], ToolResultData | str]


async def generate(
    model: str,
    prompt_or_messages: str | list[Message],
    *,
    system: str | None = None,
    tools: list[ToolDefinition | dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stop_sequences: list[str] | None = None,
    response_format: ResponseFormat | dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    client: Client | None = None,
    max_tool_rounds: int = 1,
    tool_executor: ToolExecutor | None = None,
    stop_when: StopCondition | None = None,
    retry_policy: RetryPolicy | None = None,
    max_retries: int | None = None,
    timeout: float | TimeoutConfig | None = None,
    abort_signal: AbortSignal | None = None,
) -> GenerateResult:
    """Generate an LLM response, optionally with automatic tool execution.

    When *max_tool_rounds* > 0 and a *tool_executor* is provided, the
    function will automatically execute tool calls and feed results back
    to the model, looping up to *max_tool_rounds* times.

    Returns a :class:`GenerateResult` with the final response and all
    intermediate steps.
    """
    effective_client = client or _get_default_client()
    messages = _coerce_messages(prompt_or_messages)
    if system is not None:
        messages = [Message.system(system)] + messages
    coerced_tools = _coerce_tools(tools)
    if retry_policy is not None:
        policy = retry_policy
    elif max_retries is not None:
        policy = RetryPolicy(max_retries=max_retries)
    else:
        policy = RetryPolicy(max_retries=0)
    stop_fn = stop_when or stop_on_text

    steps: list[StepResult] = []
    total_usage = Usage()

    for _round in range(max(max_tool_rounds, 0) + 1):
        request = Request(
            model=model,
            messages=messages,
            tools=coerced_tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            provider=provider,
            provider_options=provider_options,
            abort_signal=abort_signal,
        )

        response = await _retry_call(
            lambda: effective_client.complete(request),
            policy,
        )
        total_usage = total_usage + response.usage

        step = StepResult(response=response, tool_calls=response.tool_calls)

        # Check stop condition
        if stop_fn(response, steps) or not response.tool_calls:
            steps.append(step)
            break

        # Execute tools if we have an executor and rounds remaining
        if tool_executor is None or _round >= max_tool_rounds:
            steps.append(step)
            break

        # Execute tools concurrently
        async def _exec_tool(tc: ToolCallData) -> ToolResultData:
            try:
                result = tool_executor(tc)
                if isinstance(result, str):
                    return ToolResultData(tool_call_id=tc.id, content=result)
                return result
            except Exception as exc:
                return ToolResultData(
                    tool_call_id=tc.id,
                    content=f"Tool error: {exc}",
                    is_error=True,
                )

        if len(response.tool_calls) > 1:
            import asyncio
            tool_results = list(await asyncio.gather(
                *[_exec_tool(tc) for tc in response.tool_calls]
            ))
        else:
            tool_results = [await _exec_tool(tc) for tc in response.tool_calls]

        step.tool_results = tool_results
        steps.append(step)

        # Append assistant message + tool results to messages for next round
        parts: list[ContentPart] = []
        if response.text:
            parts.append(ContentPart(kind=ContentKind.TEXT, text=response.text))
        for tc in response.tool_calls:
            parts.append(ContentPart(kind=ContentKind.TOOL_CALL, tool_call=tc))
        messages.append(Message(role=Message.assistant("").role, content=parts))

        for tr in tool_results:
            messages.append(Message.tool_result(
                tool_call_id=tr.tool_call_id,
                content=tr.content,
                is_error=tr.is_error,
            ))

    final_response = steps[-1].response if steps else Response()
    return GenerateResult(
        response=final_response,
        steps=steps,
        total_usage=total_usage,
    )


# ---------------------------------------------------------------------------
# generate_object()
# ---------------------------------------------------------------------------


async def generate_object(
    model: str,
    prompt_or_messages: str | list[Message],
    *,
    schema: dict[str, Any],
    max_tokens: int | None = None,
    temperature: float | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    client: Client | None = None,
    retry_policy: RetryPolicy | None = None,
    abort_signal: AbortSignal | None = None,
) -> dict[str, Any]:
    """Generate a structured object matching *schema*.

    Uses the provider's native JSON mode or response_format when available.
    For providers that lack native structured output (e.g. Anthropic), falls
    back to tool-based extraction: defines a tool whose input_schema matches
    *schema* and forces the model to call it.

    Raises :class:`NoObjectGeneratedError` if the response cannot be
    parsed as valid JSON.
    """
    effective_client = client or _get_default_client()
    messages = _coerce_messages(prompt_or_messages)
    policy = retry_policy or RetryPolicy(max_retries=0)

    # Determine if we should use tool-based extraction fallback.
    use_tool_fallback = (provider or "").startswith("anthropic") or (
        not provider
        and hasattr(effective_client, "_default_provider")
        and (effective_client._default_provider or "").startswith("anthropic")
    )

    if use_tool_fallback:
        return await _generate_object_via_tool(
            effective_client, model, messages, schema=schema,
            max_tokens=max_tokens, temperature=temperature,
            provider=provider, provider_options=provider_options,
            policy=policy, abort_signal=abort_signal,
        )

    request = Request(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        provider=provider,
        provider_options=provider_options,
        response_format={"type": "json_schema", "json_schema": schema, "strict": True},
        abort_signal=abort_signal,
    )

    response = await _retry_call(
        lambda: effective_client.complete(request),
        policy,
    )

    text = response.text.strip()
    if not text:
        raise NoObjectGeneratedError("Model returned empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise NoObjectGeneratedError(
            f"Failed to parse model output as JSON: {exc}"
        ) from exc


async def _generate_object_via_tool(
    client: Client,
    model: str,
    messages: list[Message],
    *,
    schema: dict[str, Any],
    max_tokens: int | None,
    temperature: float | None,
    provider: str | None,
    provider_options: dict[str, Any] | None,
    policy: RetryPolicy,
    abort_signal: AbortSignal | None,
) -> dict[str, Any]:
    """Fallback: use a tool call to extract structured output."""
    tool = ToolDefinition(
        name="_extract_object",
        description="Extract the structured object matching the requested schema.",
        parameters=schema,
    )
    request = Request(
        model=model,
        messages=messages,
        tools=[tool],
        tool_choice="_extract_object",
        max_tokens=max_tokens,
        temperature=temperature,
        provider=provider,
        provider_options=provider_options,
        abort_signal=abort_signal,
    )

    response = await _retry_call(
        lambda: client.complete(request),
        policy,
    )

    # Extract the tool call arguments
    tool_calls = response.tool_calls
    if tool_calls:
        args = tool_calls[0].arguments
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                return json.loads(args)
            except json.JSONDecodeError as exc:
                raise NoObjectGeneratedError(
                    f"Failed to parse tool call arguments as JSON: {exc}"
                ) from exc

    # Fallback: try parsing response text
    text = response.text.strip()
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    raise NoObjectGeneratedError("Model did not produce a valid structured object")


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


async def stream(
    model: str,
    prompt_or_messages: str | list[Message],
    *,
    system: str | None = None,
    tools: list[ToolDefinition | dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stop_sequences: list[str] | None = None,
    response_format: ResponseFormat | dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    client: Client | None = None,
    max_tool_rounds: int = 1,
    tool_executor: ToolExecutor | None = None,
    timeout: float | TimeoutConfig | None = None,
    abort_signal: AbortSignal | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream LLM response events.

    When *max_tool_rounds* > 0 and a *tool_executor* is provided, the
    stream pauses on tool calls, executes them, emits a STEP_FINISH event,
    then resumes streaming the model's next response.
    """
    effective_client = client or _get_default_client()
    messages = _coerce_messages(prompt_or_messages)
    if system is not None:
        messages = [Message.system(system)] + messages
    coerced_tools = _coerce_tools(tools)

    for _round in range(max(max_tool_rounds, 0) + 1):
        request = Request(
            model=model,
            messages=messages,
            tools=coerced_tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            provider=provider,
            provider_options=provider_options,
            abort_signal=abort_signal,
        )

        # Accumulate text and tool calls from the stream
        text_parts: list[str] = []
        tool_calls: list[ToolCallData] = []

        async for event in effective_client.stream(request):
            if event.kind == StreamEventKind.CONTENT_DELTA:
                delta = (event.data or {}).get("text", "")
                if delta:
                    text_parts.append(delta)
            elif event.kind == StreamEventKind.TOOL_CALL_END:
                if event.content_part and event.content_part.tool_call:
                    tool_calls.append(event.content_part.tool_call)
            yield event

        # If no tool calls or no executor, we're done
        if not tool_calls or tool_executor is None or _round >= max_tool_rounds:
            break

        # Execute tools
        tool_results: list[ToolResultData] = []
        for tc in tool_calls:
            try:
                result = tool_executor(tc)
                if isinstance(result, str):
                    tool_results.append(ToolResultData(tool_call_id=tc.id, content=result))
                else:
                    tool_results.append(result)
            except Exception as exc:
                tool_results.append(ToolResultData(
                    tool_call_id=tc.id,
                    content=f"Tool error: {exc}",
                    is_error=True,
                ))

        # Emit step_finish event
        yield StreamEvent(
            kind=StreamEventKind.STEP_FINISH,
            data={"tool_calls": len(tool_calls), "tool_results": len(tool_results)},
        )

        # Reconstruct messages for next round
        parts: list[ContentPart] = []
        full_text = "".join(text_parts)
        if full_text:
            parts.append(ContentPart(kind=ContentKind.TEXT, text=full_text))
        for tc in tool_calls:
            parts.append(ContentPart(kind=ContentKind.TOOL_CALL, tool_call=tc))
        messages.append(Message(role=Role.ASSISTANT, content=parts))

        for tr in tool_results:
            messages.append(Message.tool_result(
                tool_call_id=tr.tool_call_id,
                content=tr.content,
                is_error=tr.is_error,
            ))


# ---------------------------------------------------------------------------
# StreamResult
# ---------------------------------------------------------------------------


class StreamResult:
    """Wrapper around a streaming response that accumulates text and tool calls.

    Iterating yields ``StreamEvent`` items.  After iteration completes,
    ``response()`` returns the fully assembled ``Response``.
    """

    def __init__(self, events: AsyncIterator[StreamEvent]) -> None:
        self._events = events
        self._text_parts: list[str] = []
        self._tool_calls: list[ToolCallData] = []
        self._usage = Usage()
        self._finish_reason: FinishReason | None = None
        self._done = False

    async def __aiter__(self):
        async for event in self._events:
            if event.kind == StreamEventKind.CONTENT_DELTA:
                delta = (event.data or {}).get("text", "")
                if delta:
                    self._text_parts.append(delta)
            elif event.kind == StreamEventKind.TOOL_CALL_END:
                if event.content_part and event.content_part.tool_call:
                    self._tool_calls.append(event.content_part.tool_call)
            if event.finish_reason:
                self._finish_reason = event.finish_reason
            if event.usage:
                self._usage = event.usage
            yield event
        self._done = True

    @property
    def text_stream(self) -> AsyncIterator[str]:
        """Yield only text deltas."""
        return self._text_stream_gen()

    async def _text_stream_gen(self) -> AsyncIterator[str]:
        async for event in self._events:
            if event.kind == StreamEventKind.CONTENT_DELTA:
                delta = (event.data or {}).get("text", "")
                if delta:
                    self._text_parts.append(delta)
                    yield delta
            elif event.kind == StreamEventKind.TOOL_CALL_END:
                if event.content_part and event.content_part.tool_call:
                    self._tool_calls.append(event.content_part.tool_call)
            if event.finish_reason:
                self._finish_reason = event.finish_reason
            if event.usage:
                self._usage = event.usage
        self._done = True

    @property
    def partial_text(self) -> str:
        """Return text accumulated so far."""
        return "".join(self._text_parts)

    @property
    def partial_response(self) -> Response | None:
        """Return the accumulated response state at any point during streaming."""
        if not self._text_parts and not self._tool_calls:
            return None
        parts: list[ContentPart] = []
        text = "".join(self._text_parts)
        if text:
            parts.append(ContentPart(kind=ContentKind.TEXT, text=text))
        for tc in self._tool_calls:
            parts.append(ContentPart(kind=ContentKind.TOOL_CALL, tool_call=tc))
        return Response(
            model="",
            content=parts,
            usage=self._usage,
            finish_reason=self._finish_reason or FinishReason.STOP,
        )

    def response(self) -> Response:
        """Return the fully assembled Response (only valid after iteration completes)."""
        parts: list[ContentPart] = []
        text = "".join(self._text_parts)
        if text:
            parts.append(ContentPart(kind=ContentKind.TEXT, text=text))
        for tc in self._tool_calls:
            parts.append(ContentPart(kind=ContentKind.TOOL_CALL, tool_call=tc))
        return Response(
            model="",
            content=parts,
            usage=self._usage,
            finish_reason=self._finish_reason or FinishReason.STOP,
        )


class StreamAccumulator:
    """Accumulates stream events into a final Response.

    Low-level utility for building custom stream processors. Collects
    text deltas, tool calls, reasoning deltas, usage, and finish reason
    from individual StreamEvent objects.

    Usage::

        acc = StreamAccumulator()
        async for event in client.stream(request):
            acc.add(event)
            # process event...
        response = acc.response()
    """

    def __init__(self) -> None:
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._tool_calls: list[ToolCallData] = []
        self._usage = Usage()
        self._finish_reason: FinishReason | None = None

    def process(self, event: StreamEvent) -> None:
        """Ingest a single stream event (alias for :meth:`add`)."""
        self.add(event)

    def add(self, event: StreamEvent) -> None:
        """Ingest a single stream event."""
        if event.kind == StreamEventKind.CONTENT_DELTA:
            delta = event.delta or (event.data or {}).get("text", "")
            if delta:
                self._text_parts.append(delta)
        elif event.kind == StreamEventKind.THINKING_DELTA:
            delta = event.reasoning_delta or ""
            if delta:
                self._reasoning_parts.append(delta)
        elif event.kind == StreamEventKind.TOOL_CALL_END:
            if event.content_part and event.content_part.tool_call:
                self._tool_calls.append(event.content_part.tool_call)
        if event.finish_reason:
            self._finish_reason = event.finish_reason
        if event.usage:
            self._usage = event.usage

    @property
    def text(self) -> str:
        return "".join(self._text_parts)

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning_parts)

    @property
    def tool_calls(self) -> list[ToolCallData]:
        return list(self._tool_calls)

    @property
    def usage(self) -> Usage:
        return self._usage

    @property
    def finish_reason(self) -> FinishReason | None:
        return self._finish_reason

    def response(self) -> Response:
        """Build the accumulated Response."""
        from attractor.llm.types import ThinkingData

        parts: list[ContentPart] = []
        reasoning = "".join(self._reasoning_parts)
        if reasoning:
            parts.append(ContentPart(
                kind=ContentKind.THINKING,
                thinking=ThinkingData(text=reasoning),
            ))
        text = "".join(self._text_parts)
        if text:
            parts.append(ContentPart(kind=ContentKind.TEXT, text=text))
        for tc in self._tool_calls:
            parts.append(ContentPart(kind=ContentKind.TOOL_CALL, tool_call=tc))
        return Response(
            model="",
            content=parts,
            usage=self._usage,
            finish_reason=self._finish_reason or FinishReason.STOP,
        )


async def stream_with_result(
    model: str,
    prompt_or_messages: str | list[Message],
    *,
    tools: list[ToolDefinition | dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    client: Client | None = None,
    abort_signal: AbortSignal | None = None,
) -> StreamResult:
    """Stream LLM response events, returning a StreamResult wrapper.

    The StreamResult can be iterated for events, and after completion
    provides the assembled response.
    """
    events = stream(
        model,
        prompt_or_messages,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        provider=provider,
        provider_options=provider_options,
        client=client,
        abort_signal=abort_signal,
    )
    return StreamResult(events)


# ---------------------------------------------------------------------------
# stream_object()
# ---------------------------------------------------------------------------


async def stream_object(
    model: str,
    prompt_or_messages: str | list[Message],
    *,
    schema: dict[str, Any],
    max_tokens: int | None = None,
    temperature: float | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    client: Client | None = None,
    abort_signal: AbortSignal | None = None,
) -> AsyncIterator[dict[str, Any] | str]:
    """Stream structured output with incremental JSON parsing.

    Yields text deltas as strings. After all events are consumed,
    the final yield is the parsed JSON object (dict). If parsing fails,
    raises :class:`NoObjectGeneratedError`.
    """
    effective_client = client or _get_default_client()
    messages = _coerce_messages(prompt_or_messages)

    request = Request(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        provider=provider,
        provider_options=provider_options,
        response_format={"type": "json_schema", "json_schema": schema, "strict": True},
        abort_signal=abort_signal,
    )

    text_parts: list[str] = []
    async for event in effective_client.stream(request):
        if event.kind == StreamEventKind.CONTENT_DELTA:
            delta = (event.data or {}).get("text", "")
            if delta:
                text_parts.append(delta)
                yield delta

    full_text = "".join(text_parts).strip()
    if not full_text:
        raise NoObjectGeneratedError("Streaming produced no output")

    try:
        yield json.loads(full_text)
    except json.JSONDecodeError as exc:
        raise NoObjectGeneratedError(
            f"Failed to parse streamed output as JSON: {exc}"
        ) from exc
