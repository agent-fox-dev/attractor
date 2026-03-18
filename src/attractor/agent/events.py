"""Agent event system per Section 2.9 of the coding-agent-loop-spec."""

from enum import StrEnum
from dataclasses import dataclass, field
from typing import Any
import time


class EventKind(StrEnum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_INPUT = "user_input"
    ASSISTANT_TEXT_START = "assistant_text_start"
    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    ASSISTANT_TEXT_END = "assistant_text_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_OUTPUT_DELTA = "tool_call_output_delta"
    TOOL_CALL_END = "tool_call_end"
    STEERING_INJECTED = "steering_injected"
    AWAITING_INPUT = "awaiting_input"
    TURN_LIMIT = "turn_limit"
    LOOP_DETECTION = "loop_detection"
    TOOL_APPROVAL_REQUIRED = "tool_approval_required"
    ERROR = "error"


@dataclass
class SessionEvent:
    kind: EventKind
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventEmitter:
    def __init__(self) -> None:
        self._callbacks: list = []

    def on(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, event: SessionEvent) -> None:
        for cb in self._callbacks:
            cb(event)
