"""ExitHandler: returns SUCCESS immediately."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..graph import Outcome, StageStatus
from .base import Handler

if TYPE_CHECKING:
    from ..context import Context
    from ..events import EventEmitter
    from ..graph import Graph, Node


class ExitHandler(Handler):
    """Handler for exit nodes -- always succeeds immediately."""

    def execute(
        self,
        node: "Node",
        context: "Context",
        graph: "Graph",
        logs_root: Path | None = None,
        emitter: "EventEmitter | None" = None,
    ) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)
