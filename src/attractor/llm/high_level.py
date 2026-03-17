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
    ContentKind,
    ContentPart,
    FinishReason,
    Message,
    NoObjectGeneratedError,
    ProviderError,
    Request,
    Response,
    RetryPolicy,
    StreamEvent,
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
    tools: list[ToolDefinition | dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    client: Client | None = None,
    max_tool_rounds: int = 0,
    tool_executor: ToolExecutor | None = None,
    stop_when: StopCondition | None = None,
    retry_policy: RetryPolicy | None = None,
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
    coerced_tools = _coerce_tools(tools)
    policy = retry_policy or RetryPolicy(max_retries=0)
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
            reasoning_effort=reasoning_effort,
            provider=provider,
            provider_options=provider_options,
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

        tool_results: list[ToolResultData] = []
        for tc in response.tool_calls:
            try:
                result = tool_executor(tc)
                if isinstance(result, str):
                    result = ToolResultData(tool_call_id=tc.id, content=result)
                tool_results.append(result)
            except Exception as exc:
                tool_results.append(ToolResultData(
                    tool_call_id=tc.id,
                    content=f"Tool error: {exc}",
                    is_error=True,
                ))

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
) -> dict[str, Any]:
    """Generate a structured object matching *schema*.

    Uses the provider's native JSON mode or response_format when available.
    Parses the response text as JSON and returns the resulting dict.

    Raises :class:`NoObjectGeneratedError` if the response cannot be
    parsed as valid JSON.
    """
    effective_client = client or _get_default_client()
    messages = _coerce_messages(prompt_or_messages)
    policy = retry_policy or RetryPolicy(max_retries=0)

    request = Request(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        provider=provider,
        provider_options=provider_options,
        response_format={"type": "json_schema", "json_schema": schema, "strict": True},
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


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


async def stream(
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
    max_tool_rounds: int = 0,
) -> AsyncIterator[StreamEvent]:
    """Stream LLM response events.

    Accepts the same parameters as :func:`generate` but returns an async
    iterator of :class:`StreamEvent` objects instead of a single
    :class:`Response`.
    """
    effective_client = client or _get_default_client()
    messages = _coerce_messages(prompt_or_messages)
    coerced_tools = _coerce_tools(tools)

    request = Request(
        model=model,
        messages=messages,
        tools=coerced_tools,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        provider=provider,
        provider_options=provider_options,
    )

    async for event in effective_client.stream(request):
        yield event
