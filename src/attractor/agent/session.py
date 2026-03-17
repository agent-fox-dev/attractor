"""Core agentic loop Session per Section 2 of the coding-agent-loop-spec.

Implements the full ``process_input`` loop with:
- LLM call -> tool execution -> loop until natural completion
- Steering and follow-up queues
- Loop detection (repeating patterns of length 1, 2, or 3)
- Turn types: UserTurn, AssistantTurn, ToolResultsTurn, SteeringTurn, SystemTurn
- Event emission at every stage
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from attractor.llm.client import Client
from attractor.llm.types import (
    ContentKind,
    ContentPart,
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventKind,
    ToolCallData,
    ToolResultData,
    Usage,
)

from attractor.agent.events import EventEmitter, EventKind, SessionEvent
from attractor.agent.execution.base import ExecutionEnvironment
from attractor.agent.profiles.base import ProviderProfile
from attractor.agent.truncation import truncate_tool_output

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class SessionState(StrEnum):
    IDLE = "idle"
    PROCESSING = "processing"
    AWAITING_INPUT = "awaiting_input"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# Turn types (Section 2.4)
# ---------------------------------------------------------------------------


@dataclass
class UserTurn:
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[ToolCallData] = field(default_factory=list)
    reasoning: str | None = None
    usage: Usage = field(default_factory=Usage)
    response_id: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResultsTurn:
    results: list[ToolResultData] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SteeringTurn:
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemTurn:
    content: str
    timestamp: float = field(default_factory=time.time)


Turn = UserTurn | AssistantTurn | ToolResultsTurn | SteeringTurn | SystemTurn


# ---------------------------------------------------------------------------
# Session configuration (Section 2.2)
# ---------------------------------------------------------------------------


@dataclass
class SessionConfig:
    max_turns: int = 0
    max_tool_rounds_per_input: int = 0
    default_command_timeout_ms: int = 10000
    max_command_timeout_ms: int = 600000
    reasoning_effort: str | None = None
    tool_output_limits: dict[str, int] = field(default_factory=dict)
    tool_line_limits: dict[str, int] = field(default_factory=dict)
    enable_loop_detection: bool = True
    loop_detection_window: int = 10
    max_subagent_depth: int = 1
    max_input_tokens: int = 0
    max_output_tokens: int = 0
    max_total_tokens: int = 0


# ---------------------------------------------------------------------------
# Project document discovery (Section 6.5)
# ---------------------------------------------------------------------------

_PROJECT_DOC_BUDGET = 32 * 1024  # 32 KB

_UNIVERSAL_DOCS = ["AGENTS.md"]
_PROVIDER_DOCS: dict[str, list[str]] = {
    "anthropic": ["CLAUDE.md"],
    "openai": [".codex/instructions.md"],
    "gemini": ["GEMINI.md"],
}


def discover_project_docs(working_dir: str, provider_id: str = "") -> str:
    """Walk from git root to working_dir, loading recognized instruction files."""
    cwd = Path(working_dir).resolve()

    # Find git root
    git_root = cwd
    while git_root != git_root.parent:
        if (git_root / ".git").exists():
            break
        git_root = git_root.parent
    else:
        git_root = cwd

    # Build list of doc filenames to look for
    doc_names = list(_UNIVERSAL_DOCS)
    for name in _PROVIDER_DOCS.get(provider_id, []):
        if name not in doc_names:
            doc_names.append(name)

    # Collect directories from git root to cwd
    dirs: list[Path] = []
    current = git_root
    dirs.append(current)
    try:
        rel = cwd.relative_to(git_root)
        for part in rel.parts:
            current = current / part
            if current != git_root:
                dirs.append(current)
    except ValueError:
        dirs = [cwd]

    # Load files
    collected: list[str] = []
    total_bytes = 0

    for d in dirs:
        for name in doc_names:
            doc_path = d / name
            if doc_path.is_file():
                try:
                    text = doc_path.read_text(encoding="utf-8", errors="replace")
                    if total_bytes + len(text) > _PROJECT_DOC_BUDGET:
                        remaining = _PROJECT_DOC_BUDGET - total_bytes
                        if remaining > 0:
                            text = text[:remaining] + "\n[Project instructions truncated at 32KB]"
                        else:
                            break
                    collected.append(f"# {doc_path}\n\n{text}")
                    total_bytes += len(text)
                except (PermissionError, OSError):
                    continue

    return "\n\n---\n\n".join(collected)


# ---------------------------------------------------------------------------
# History -> Message conversion
# ---------------------------------------------------------------------------


def convert_history_to_messages(history: list[Turn]) -> list[Message]:
    """Convert the turn-based history into LLM ``Message`` objects."""
    messages: list[Message] = []

    for turn in history:
        if isinstance(turn, UserTurn):
            messages.append(Message.user(turn.content))

        elif isinstance(turn, AssistantTurn):
            parts: list[ContentPart] = []
            if turn.reasoning:
                from attractor.llm.types import ThinkingData

                parts.append(
                    ContentPart(
                        kind=ContentKind.THINKING,
                        thinking=ThinkingData(text=turn.reasoning),
                    )
                )
            if turn.content:
                parts.append(ContentPart(kind=ContentKind.TEXT, text=turn.content))
            for tc in turn.tool_calls:
                parts.append(
                    ContentPart(kind=ContentKind.TOOL_CALL, tool_call=tc)
                )
            messages.append(Message(role=Role.ASSISTANT, content=parts))

        elif isinstance(turn, ToolResultsTurn):
            for result in turn.results:
                messages.append(
                    Message.tool_result(
                        tool_call_id=result.tool_call_id,
                        content=result.content,
                        is_error=result.is_error,
                    )
                )

        elif isinstance(turn, SteeringTurn):
            messages.append(Message.user(turn.content))

        elif isinstance(turn, SystemTurn):
            messages.append(Message.system(turn.content))

    return messages


# ---------------------------------------------------------------------------
# Loop detection (Section 2.10)
# ---------------------------------------------------------------------------


def _tool_call_signature(tc: ToolCallData) -> str:
    """Hash a tool call into a comparable signature."""
    args_str = json.dumps(tc.arguments, sort_keys=True) if isinstance(tc.arguments, dict) else str(tc.arguments)
    raw = f"{tc.name}:{args_str}"
    return hashlib.md5(raw.encode()).hexdigest()


def _extract_tool_call_signatures(history: list[Turn], last: int) -> list[str]:
    """Extract the signatures of the most recent *last* tool calls."""
    sigs: list[str] = []
    for turn in reversed(history):
        if isinstance(turn, AssistantTurn):
            for tc in reversed(turn.tool_calls):
                sigs.append(_tool_call_signature(tc))
                if len(sigs) >= last:
                    break
        if len(sigs) >= last:
            break

    sigs.reverse()
    return sigs


def detect_loop(history: list, window_size: int) -> bool:
    """Check for repeating tool-call patterns in the recent history.

    Looks for repeating patterns of length 1, 2, or 3 within the last
    *window_size* tool calls.

    *history* may be a list of ``Turn`` objects (used by the Session) or
    a list of ``(name, args_hash)`` tuples (convenience form for callers).
    """
    # Support the simple tuple form: [(name, args_hash), ...]
    if history and isinstance(history[0], tuple):
        recent = [f"{name}:{args_hash}" for name, args_hash in history[-window_size:]]
    else:
        recent = _extract_tool_call_signatures(history, last=window_size)
    if len(recent) < window_size:
        return False

    for pattern_len in (1, 2, 3):
        if window_size % pattern_len != 0:
            continue
        pattern = recent[:pattern_len]
        all_match = True
        for i in range(pattern_len, window_size, pattern_len):
            if recent[i : i + pattern_len] != pattern:
                all_match = False
                break
        if all_match:
            return True

    return False


# ---------------------------------------------------------------------------
# Count turns
# ---------------------------------------------------------------------------


def _count_turns(history: list[Turn]) -> int:
    """Count the number of user + assistant turns in the history."""
    return sum(
        1
        for t in history
        if isinstance(t, (UserTurn, AssistantTurn))
    )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """The central orchestrator for the agentic loop.

    Holds conversation state, dispatches tool calls, manages the event
    stream, and enforces limits.
    """

    def __init__(
        self,
        profile: ProviderProfile,
        execution_env: ExecutionEnvironment,
        config: SessionConfig | None = None,
        llm_client: Client | None = None,
        depth: int = 0,
    ) -> None:
        self.id: str = str(uuid.uuid4())
        self.profile: ProviderProfile = profile
        self.execution_env: ExecutionEnvironment = execution_env
        self.config: SessionConfig = config or SessionConfig()
        self.llm_client: Client = llm_client or Client.from_env()
        self.history: list[Turn] = []
        self.state: SessionState = SessionState.IDLE
        self.event_emitter: EventEmitter = EventEmitter()
        self.steering_queue: deque[str] = deque()
        self.followup_queue: deque[str] = deque()
        self.subagents: dict[str, Any] = {}
        self.abort_signaled: bool = False
        self.depth: int = depth
        self.total_usage: Usage = Usage()

    # -- public API --------------------------------------------------------

    async def submit(self, user_input: str, *, stream: bool = False) -> None:
        """Submit user input and run the agentic loop to completion.

        If *stream* is True, uses the streaming API and emits
        ``ASSISTANT_TEXT_DELTA`` events as tokens arrive.
        """
        if self.state == SessionState.IDLE and not self.history:
            self._emit(EventKind.SESSION_START, session_id=self.id)
        if stream:
            await self._process_input_streaming(user_input)
        else:
            await self._process_input(user_input)

    def steer(self, message: str) -> None:
        """Queue a steering message for injection after the current tool round."""
        self.steering_queue.append(message)

    def follow_up(self, message: str) -> None:
        """Queue a message to be processed after the current input completes."""
        self.followup_queue.append(message)

    def abort(self) -> None:
        """Signal the loop to abort after the current step."""
        self.abort_signaled = True
        self.state = SessionState.CLOSED

    async def close(self) -> None:
        """Close the session and clean up resources."""
        self.abort_signaled = True
        self.state = SessionState.CLOSED
        self.execution_env.cleanup()
        self._emit(EventKind.SESSION_END, state="closed")

    # -- budget enforcement ------------------------------------------------

    def _check_budget(self) -> bool:
        """Return True if the token budget has been exceeded."""
        cfg = self.config
        if cfg.max_input_tokens > 0 and self.total_usage.input_tokens >= cfg.max_input_tokens:
            return True
        if cfg.max_output_tokens > 0 and self.total_usage.output_tokens >= cfg.max_output_tokens:
            return True
        if cfg.max_total_tokens > 0 and self.total_usage.total_tokens >= cfg.max_total_tokens:
            return True
        return False

    def get_metrics(self) -> dict[str, Any]:
        """Return a snapshot of session metrics."""
        return {
            "session_id": self.id,
            "state": self.state.value,
            "total_turns": _count_turns(self.history),
            "total_input_tokens": self.total_usage.input_tokens,
            "total_output_tokens": self.total_usage.output_tokens,
            "total_tokens": self.total_usage.total_tokens,
            "depth": self.depth,
        }

    # -- event helpers -----------------------------------------------------

    def _emit(self, kind: EventKind, **data: Any) -> None:
        event = SessionEvent(
            kind=kind,
            session_id=self.id,
            data=data,
        )
        self.event_emitter.emit(event)

    # -- steering drain ----------------------------------------------------

    def _drain_steering(self) -> None:
        """Drain pending steering messages into history."""
        while self.steering_queue:
            msg = self.steering_queue.popleft()
            self.history.append(SteeringTurn(content=msg))
            self._emit(EventKind.STEERING_INJECTED, content=msg)

    # -- tool execution ----------------------------------------------------

    async def _execute_tool_calls(
        self, tool_calls: list[ToolCallData]
    ) -> list[ToolResultData]:
        """Execute tool calls, optionally in parallel."""
        if (
            self.profile.supports_parallel_tool_calls
            and len(tool_calls) > 1
        ):
            tasks = [
                asyncio.get_event_loop().run_in_executor(
                    None, self._execute_single_tool, tc
                )
                for tc in tool_calls
            ]
            return list(await asyncio.gather(*tasks))
        else:
            return [self._execute_single_tool(tc) for tc in tool_calls]

    def _execute_single_tool(self, tool_call: ToolCallData) -> ToolResultData:
        """Execute a single tool call through the registry."""
        self._emit(
            EventKind.TOOL_CALL_START,
            tool_name=tool_call.name,
            call_id=tool_call.id,
        )

        registered = self.profile.tool_registry.get(tool_call.name)
        if registered is None:
            error_msg = f"Unknown tool: {tool_call.name}"
            self._emit(EventKind.TOOL_CALL_END, call_id=tool_call.id, error=error_msg)
            return ToolResultData(
                tool_call_id=tool_call.id, content=error_msg, is_error=True
            )

        try:
            # Parse arguments if they're a string
            arguments = tool_call.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"input": arguments}

            raw_output = registered.executor(arguments, self.execution_env)

            # Truncate for LLM
            truncated_output = truncate_tool_output(
                raw_output, tool_call.name, self.config
            )

            # Emit full (untruncated) output via event stream
            self._emit(
                EventKind.TOOL_CALL_END,
                call_id=tool_call.id,
                output=raw_output,
            )

            return ToolResultData(
                tool_call_id=tool_call.id,
                content=truncated_output,
                is_error=False,
            )

        except Exception as exc:
            error_msg = f"Tool error ({tool_call.name}): {exc}"
            self._emit(
                EventKind.TOOL_CALL_END,
                call_id=tool_call.id,
                error=error_msg,
            )
            return ToolResultData(
                tool_call_id=tool_call.id, content=error_msg, is_error=True
            )

    # -- core agentic loop (Section 2.5) -----------------------------------

    async def _process_input(self, user_input: str) -> None:
        """The core agentic loop."""
        self.state = SessionState.PROCESSING
        self.history.append(UserTurn(content=user_input))
        self._emit(EventKind.USER_INPUT, content=user_input)

        # Drain any pending steering messages before the first LLM call
        self._drain_steering()

        round_count = 0

        while True:
            # 1. Check limits
            if (
                self.config.max_tool_rounds_per_input > 0
                and round_count >= self.config.max_tool_rounds_per_input
            ):
                self._emit(EventKind.TURN_LIMIT, round=round_count)
                break

            if (
                self.config.max_turns > 0
                and _count_turns(self.history) >= self.config.max_turns
            ):
                self._emit(
                    EventKind.TURN_LIMIT,
                    total_turns=_count_turns(self.history),
                )
                break

            if self.abort_signaled:
                break

            # 2. Build LLM request
            project_docs = discover_project_docs(
                self.execution_env.working_directory(),
                provider_id=self.profile.id,
            )
            system_prompt = self.profile.build_system_prompt(
                environment=self.execution_env,
                project_docs=project_docs,
            )
            messages = convert_history_to_messages(self.history)
            tool_defs = self.profile.tools()

            request = Request(
                model=self.profile.model,
                messages=[Message.system(system_prompt)] + messages,
                tools=tool_defs if tool_defs else None,
                tool_choice="auto" if tool_defs else None,
                reasoning_effort=self.config.reasoning_effort,
                provider=self.profile.id,
                provider_options=self.profile.provider_options(),
            )

            # 3. Call LLM
            self._emit(EventKind.ASSISTANT_TEXT_START)
            try:
                response: Response = await self.llm_client.complete(request)
            except Exception as exc:
                self._emit(EventKind.ERROR, error=str(exc))
                self.state = SessionState.CLOSED
                return

            # 3b. Track usage and check budget
            self.total_usage = self.total_usage + response.usage
            if self._check_budget():
                self._emit(EventKind.TURN_LIMIT, reason="token_budget_exceeded")
                break

            # 4. Extract reasoning text from response
            reasoning_text: str | None = None
            for part in response.content:
                if part.kind == ContentKind.THINKING and part.thinking:
                    reasoning_text = part.thinking.text

            # 5. Record assistant turn
            assistant_turn = AssistantTurn(
                content=response.text,
                tool_calls=response.tool_calls,
                reasoning=reasoning_text,
                usage=response.usage,
                response_id=response.id,
            )
            self.history.append(assistant_turn)
            self._emit(
                EventKind.ASSISTANT_TEXT_END,
                text=response.text,
                reasoning=reasoning_text,
            )

            # 6. If no tool calls, natural completion or awaiting input
            if not response.tool_calls:
                # If the model ended without tool calls and the response looks
                # like a question (no definitive answer), set AWAITING_INPUT
                if response.finish_reason == FinishReason.STOP and response.text.rstrip().endswith("?"):
                    self.state = SessionState.AWAITING_INPUT
                    self._emit(EventKind.ASSISTANT_TEXT_END, text=response.text, awaiting_input=True)
                break

            # 7. Execute tool calls
            round_count += 1
            results = await self._execute_tool_calls(response.tool_calls)
            self.history.append(ToolResultsTurn(results=results))

            # 8. Drain steering messages injected during tool execution
            self._drain_steering()

            # 9. Loop detection
            if self.config.enable_loop_detection:
                if detect_loop(self.history, self.config.loop_detection_window):
                    warning = (
                        f"Loop detected: the last {self.config.loop_detection_window} "
                        f"tool calls follow a repeating pattern. "
                        f"Try a different approach."
                    )
                    self.history.append(SteeringTurn(content=warning))
                    self._emit(EventKind.LOOP_DETECTION, message=warning)

        # Process follow-up messages if any are queued
        if self.followup_queue:
            next_input = self.followup_queue.popleft()
            await self._process_input(next_input)
            return

        self.state = SessionState.IDLE
        self._emit(EventKind.SESSION_END)

    # -- streaming agentic loop --------------------------------------------

    async def _process_input_streaming(self, user_input: str) -> None:
        """Streaming variant of the agentic loop that emits delta events."""
        self.state = SessionState.PROCESSING
        self.history.append(UserTurn(content=user_input))
        self._emit(EventKind.USER_INPUT, content=user_input)
        self._drain_steering()

        round_count = 0

        while True:
            if (
                self.config.max_tool_rounds_per_input > 0
                and round_count >= self.config.max_tool_rounds_per_input
            ):
                self._emit(EventKind.TURN_LIMIT, round=round_count)
                break

            if self.abort_signaled:
                break

            # Build request
            project_docs = discover_project_docs(
                self.execution_env.working_directory(),
                provider_id=self.profile.id,
            )
            system_prompt = self.profile.build_system_prompt(
                environment=self.execution_env,
                project_docs=project_docs,
            )
            messages = convert_history_to_messages(self.history)
            tool_defs = self.profile.tools()

            request = Request(
                model=self.profile.model,
                messages=[Message.system(system_prompt)] + messages,
                tools=tool_defs if tool_defs else None,
                tool_choice="auto" if tool_defs else None,
                reasoning_effort=self.config.reasoning_effort,
                provider=self.profile.id,
                provider_options=self.profile.provider_options(),
            )

            # Stream from LLM
            self._emit(EventKind.ASSISTANT_TEXT_START)
            text_parts: list[str] = []
            tool_calls: list[ToolCallData] = []
            reasoning_text: str | None = None
            finish_reason: FinishReason | None = None
            usage_obj = Usage()

            try:
                async for event in self.llm_client.stream(request):
                    if event.kind == StreamEventKind.CONTENT_DELTA:
                        delta = (event.data or {}).get("text", "")
                        if delta:
                            text_parts.append(delta)
                            self._emit(EventKind.ASSISTANT_TEXT_DELTA, delta=delta)
                    elif event.kind == StreamEventKind.THINKING_DELTA:
                        delta = (event.data or {}).get("text", "")
                        if delta:
                            reasoning_text = (reasoning_text or "") + delta
                    elif event.kind == StreamEventKind.TOOL_CALL_END:
                        if event.content_part and event.content_part.tool_call:
                            tool_calls.append(event.content_part.tool_call)
                    elif event.finish_reason:
                        finish_reason = event.finish_reason
                    if event.usage:
                        usage_obj = event.usage
            except Exception as exc:
                self._emit(EventKind.ERROR, error=str(exc))
                self.state = SessionState.CLOSED
                return

            full_text = "".join(text_parts)

            # Track usage and check budget
            self.total_usage = self.total_usage + usage_obj
            if self._check_budget():
                self._emit(EventKind.TURN_LIMIT, reason="token_budget_exceeded")
                break

            assistant_turn = AssistantTurn(
                content=full_text,
                tool_calls=tool_calls,
                reasoning=reasoning_text,
                usage=usage_obj,
            )
            self.history.append(assistant_turn)
            self._emit(EventKind.ASSISTANT_TEXT_END, text=full_text, reasoning=reasoning_text)

            if not tool_calls:
                if finish_reason == FinishReason.STOP and full_text.rstrip().endswith("?"):
                    self.state = SessionState.AWAITING_INPUT
                break

            round_count += 1
            results = await self._execute_tool_calls(tool_calls)
            self.history.append(ToolResultsTurn(results=results))
            self._drain_steering()

            if self.config.enable_loop_detection:
                if detect_loop(self.history, self.config.loop_detection_window):
                    warning = (
                        f"Loop detected: the last {self.config.loop_detection_window} "
                        f"tool calls follow a repeating pattern. "
                        f"Try a different approach."
                    )
                    self.history.append(SteeringTurn(content=warning))
                    self._emit(EventKind.LOOP_DETECTION, message=warning)

        if self.followup_queue:
            next_input = self.followup_queue.popleft()
            await self._process_input_streaming(next_input)
            return

        self.state = SessionState.IDLE
        self._emit(EventKind.SESSION_END)

    # -- conversation export / import --------------------------------------

    def export_conversation(self) -> dict:
        """Export the conversation history as a serializable dict."""
        turns = []
        for turn in self.history:
            if isinstance(turn, UserTurn):
                turns.append({"type": "user", "content": turn.content, "timestamp": turn.timestamp})
            elif isinstance(turn, AssistantTurn):
                turns.append({
                    "type": "assistant",
                    "content": turn.content,
                    "reasoning": turn.reasoning,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in turn.tool_calls
                    ],
                    "timestamp": turn.timestamp,
                })
            elif isinstance(turn, ToolResultsTurn):
                turns.append({
                    "type": "tool_results",
                    "results": [
                        {"tool_call_id": r.tool_call_id, "content": r.content, "is_error": r.is_error}
                        for r in turn.results
                    ],
                    "timestamp": turn.timestamp,
                })
            elif isinstance(turn, SteeringTurn):
                turns.append({"type": "steering", "content": turn.content, "timestamp": turn.timestamp})
            elif isinstance(turn, SystemTurn):
                turns.append({"type": "system", "content": turn.content, "timestamp": turn.timestamp})
        return {
            "session_id": self.id,
            "turns": turns,
        }

    def import_conversation(self, data: dict) -> None:
        """Import a previously exported conversation into the session history."""
        for turn_data in data.get("turns", []):
            t = turn_data["type"]
            ts = turn_data.get("timestamp", time.time())
            if t == "user":
                self.history.append(UserTurn(content=turn_data["content"], timestamp=ts))
            elif t == "assistant":
                tc_list = [
                    ToolCallData(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                    for tc in turn_data.get("tool_calls", [])
                ]
                self.history.append(AssistantTurn(
                    content=turn_data["content"],
                    reasoning=turn_data.get("reasoning"),
                    tool_calls=tc_list,
                    timestamp=ts,
                ))
            elif t == "tool_results":
                results = [
                    ToolResultData(
                        tool_call_id=r["tool_call_id"],
                        content=r["content"],
                        is_error=r.get("is_error", False),
                    )
                    for r in turn_data.get("results", [])
                ]
                self.history.append(ToolResultsTurn(results=results, timestamp=ts))
            elif t == "steering":
                self.history.append(SteeringTurn(content=turn_data["content"], timestamp=ts))
            elif t == "system":
                self.history.append(SystemTurn(content=turn_data["content"], timestamp=ts))
