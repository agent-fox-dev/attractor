"""Thread-safe execution context and checkpoint support for pipeline runs."""

from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class Context:
    """Thread-safe key-value store shared across pipeline stages."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._logs: list[str] = []

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._values.get(key, default)

    def get_string(self, key: str, default: str = "") -> str:
        val = self.get(key, default)
        return str(val) if val is not None else default

    def append_log(self, entry: str) -> None:
        with self._lock:
            self._logs.append(entry)

    def snapshot(self) -> dict[str, Any]:
        """Return a deep copy of the current values and logs."""
        with self._lock:
            return {
                "values": copy.deepcopy(self._values),
                "logs": list(self._logs),
            }

    def clone(self) -> Context:
        """Create an independent copy of this context."""
        ctx = Context()
        with self._lock:
            ctx._values = copy.deepcopy(self._values)
            ctx._logs = list(self._logs)
        return ctx

    def apply_updates(self, updates: dict[str, Any]) -> None:
        """Merge *updates* into the context values."""
        with self._lock:
            self._values.update(updates)


@dataclass
class Checkpoint:
    """Serializable snapshot for crash recovery."""

    timestamp: float = 0.0
    current_node: str = ""
    completed_nodes: list[str] = field(default_factory=list)
    node_retries: dict[str, int] = field(default_factory=dict)
    context_values: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Persist checkpoint as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": self.timestamp,
            "current_node": self.current_node,
            "completed_nodes": self.completed_nodes,
            "node_retries": self.node_retries,
            "context_values": self.context_values,
            "logs": self.logs,
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        """Load a checkpoint from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            timestamp=data.get("timestamp", 0.0),
            current_node=data.get("current_node", ""),
            completed_nodes=data.get("completed_nodes", []),
            node_retries=data.get("node_retries", {}),
            context_values=data.get("context_values", {}),
            logs=data.get("logs", []),
        )

    @classmethod
    def from_context(cls, context: Context, current_node: str,
                     completed_nodes: list[str],
                     node_retries: dict[str, int]) -> Checkpoint:
        snap = context.snapshot()
        return cls(
            timestamp=time.time(),
            current_node=current_node,
            completed_nodes=list(completed_nodes),
            node_retries=dict(node_retries),
            context_values=snap["values"],
            logs=snap["logs"],
        )
