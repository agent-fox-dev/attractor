"""Graph data model for Attractor pipeline definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StageStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    RETRY = "retry"
    FAIL = "fail"
    SKIPPED = "skipped"


class Outcome(BaseModel):
    status: StageStatus
    preferred_label: str = ""
    suggested_next_ids: list[str] = Field(default_factory=list)
    context_updates: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    failure_reason: str = ""


class Node(BaseModel):
    id: str
    label: str = ""
    shape: str = "box"
    type: str = ""
    prompt: str = ""
    max_retries: int = 0
    goal_gate: bool = False
    retry_target: str = ""
    fallback_retry_target: str = ""
    fidelity: str = ""
    thread_id: str = ""
    css_class: str = Field("", alias="class")  # 'class' is reserved
    timeout: str = ""
    llm_model: str = ""
    llm_provider: str = ""
    reasoning_effort: str = "high"
    auto_status: bool = False
    allow_partial: bool = False
    attrs: dict[str, str] = Field(default_factory=dict)  # all raw attributes

    model_config = {"populate_by_name": True}


class Edge(BaseModel):
    from_node: str
    to_node: str
    label: str = ""
    condition: str = ""
    weight: int = 0
    fidelity: str = ""
    thread_id: str = ""
    loop_restart: bool = False
    attrs: dict[str, str] = Field(default_factory=dict)


class Graph(BaseModel):
    name: str = ""
    goal: str = ""
    label: str = ""
    model_stylesheet: str = ""
    default_max_retry: int = 50
    retry_target: str = ""
    fallback_retry_target: str = ""
    default_fidelity: str = ""
    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)
    attrs: dict[str, str] = Field(default_factory=dict)  # all raw graph attrs

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.from_node == node_id]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.to_node == node_id]
