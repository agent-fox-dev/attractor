"""FanInHandler per Section 4.9: consolidates parallel results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..graph import Outcome, StageStatus
from .base import Handler

if TYPE_CHECKING:
    from ..context import Context
    from ..events import EventEmitter
    from ..graph import Graph, Node


class FanInHandler(Handler):
    """Consolidates results from parallel branches.

    Reads ``parallel_results`` from context (set by ParallelHandler) and
    produces a summary outcome.
    """

    def execute(
        self,
        node: "Node",
        context: "Context",
        graph: "Graph",
        logs_root: Path | None = None,
        emitter: "EventEmitter | None" = None,
    ) -> Outcome:
        parallel_results = context.get("parallel_results", {})

        if not parallel_results:
            return Outcome(
                status=StageStatus.SUCCESS,
                notes="No parallel results to consolidate.",
            )

        # Summarise
        statuses = list(parallel_results.values()) if isinstance(parallel_results, dict) else []
        all_success = all(s == "success" for s in statuses)
        any_fail = any(s == "fail" for s in statuses)

        if all_success:
            status = StageStatus.SUCCESS
        elif any_fail:
            status = StageStatus.PARTIAL_SUCCESS
        else:
            status = StageStatus.SUCCESS

        return Outcome(
            status=status,
            notes=f"Fan-in consolidated {len(statuses)} branch results.",
            context_updates={
                "fan_in_summary": {
                    "total": len(statuses),
                    "success": sum(1 for s in statuses if s == "success"),
                    "fail": sum(1 for s in statuses if s == "fail"),
                },
            },
        )
