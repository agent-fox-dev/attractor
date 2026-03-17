"""Pipeline event system per Section 9.6 of the Attractor spec."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable


class PipelineEventKind(StrEnum):
    """All recognised pipeline event types."""
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"
    NODE_RETRY = "node_retry"
    NODE_SKIP = "node_skip"
    EDGE_FOLLOW = "edge_follow"
    CHECKPOINT_SAVE = "checkpoint_save"
    CHECKPOINT_LOAD = "checkpoint_load"
    GOAL_GATE_CHECK = "goal_gate_check"
    GOAL_GATE_FAIL = "goal_gate_fail"
    PARALLEL_FAN_OUT = "parallel_fan_out"
    PARALLEL_FAN_IN = "parallel_fan_in"
    HUMAN_PROMPT = "human_prompt"
    HUMAN_ANSWER = "human_answer"
    TRANSFORM_APPLY = "transform_apply"
    VALIDATION_COMPLETE = "validation_complete"
    ERROR = "error"


@dataclass
class PipelineEvent:
    """A single event emitted during pipeline execution."""
    kind: PipelineEventKind
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)


class EventEmitter:
    """Collects a callback and emits PipelineEvent instances to it."""

    def __init__(self, on_event: Callable[[PipelineEvent], None] | None = None) -> None:
        self._callback = on_event

    def emit(self, kind: PipelineEventKind, **data: Any) -> PipelineEvent:
        """Create and dispatch an event. Returns the event for convenience."""
        event = PipelineEvent(kind=kind, data=data)
        if self._callback is not None:
            self._callback(event)
        return event
