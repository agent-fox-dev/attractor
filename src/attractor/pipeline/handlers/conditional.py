"""ConditionalHandler: returns SUCCESS (no-op, routing handled by engine)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..graph import Outcome, StageStatus
from .base import Handler

if TYPE_CHECKING:
    from ..context import Context
    from ..events import EventEmitter
    from ..graph import Graph, Node


class ConditionalHandler(Handler):
    """Handler for conditional/decision nodes.

    The actual routing decision is performed by the engine's ``select_edge``
    function based on edge conditions.  This handler simply returns SUCCESS.
    """

    def execute(
        self,
        node: "Node",
        context: "Context",
        graph: "Graph",
        logs_root: Path | None = None,
        emitter: "EventEmitter | None" = None,
    ) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)
