"""Handler interface and registry per Section 4 of the Attractor spec."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import Context
    from ..graph import Graph, Node, Outcome


class Handler(ABC):
    """Abstract base for node execution handlers."""

    @abstractmethod
    def execute(
        self,
        node: "Node",
        context: "Context",
        graph: "Graph",
        logs_root: Path | None = None,
    ) -> "Outcome":
        """Execute the handler logic for *node* and return an Outcome."""
        ...


# ---------------------------------------------------------------------------
# Shape-to-type inference
# ---------------------------------------------------------------------------

SHAPE_TO_TYPE: dict[str, str] = {
    # Per Section 2.8 of the Attractor spec
    "Mdiamond": "start",
    "mdiamond": "start",
    "circle": "start",       # fallback
    "point": "start",        # fallback
    "Msquare": "exit",
    "msquare": "exit",
    "doublecircle": "exit",  # fallback
    "box": "codergen",       # default LLM task
    "rect": "codergen",
    "rectangle": "codergen",
    "hexagon": "wait.human",
    "diamond": "conditional",
    "component": "parallel",
    "tripleoctagon": "parallel.fan_in",
    "parallelogram": "tool",
    "house": "stack.manager_loop",
}


def infer_type(node: "Node") -> str:
    """Return the effective type of *node*, inferring from shape if needed."""
    if node.type:
        return node.type.lower()
    return SHAPE_TO_TYPE.get(node.shape, "codergen")


# ---------------------------------------------------------------------------
# HandlerRegistry
# ---------------------------------------------------------------------------

class HandlerRegistry:
    """Maps type strings to Handler instances."""

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, type_str: str, handler: Handler) -> None:
        self._handlers[type_str.lower()] = handler

    def resolve(self, node: "Node") -> Handler:
        """Return the handler for *node*, using type inference if needed.

        Raises KeyError if no handler is registered for the resolved type.
        """
        effective = infer_type(node)
        handler = self._handlers.get(effective)
        if handler is None:
            raise KeyError(
                f"No handler registered for type '{effective}' "
                f"(node '{node.id}', shape='{node.shape}', type='{node.type}')."
            )
        return handler
