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

    Reads ``parallel_results`` from context (set by ParallelHandler),
    selects the best candidate (via heuristic or LLM evaluation if
    node.prompt is provided), and records the winner in context.
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

        # Build candidate list from parallel_results
        candidates = self._build_candidates(parallel_results)

        if not candidates:
            return Outcome(
                status=StageStatus.SUCCESS,
                notes="No candidates to evaluate.",
            )

        # Select best candidate
        # If node.prompt is provided, it would be used for LLM-based evaluation.
        # For now, use heuristic selection (LLM eval requires a backend).
        best = self._heuristic_select(candidates)

        # Summarise
        all_success = all(c["status"] == "success" for c in candidates)
        any_fail = any(c["status"] == "fail" for c in candidates)

        if all_success:
            status = StageStatus.SUCCESS
        elif any_fail:
            status = StageStatus.PARTIAL_SUCCESS
        else:
            status = StageStatus.SUCCESS

        return Outcome(
            status=status,
            notes=f"Fan-in consolidated {len(candidates)} branch results. Best: {best['id']}",
            context_updates={
                "parallel.fan_in.best_id": best["id"],
                "parallel.fan_in.best_outcome": best["status"],
                "fan_in_summary": {
                    "total": len(candidates),
                    "success": sum(1 for c in candidates if c["status"] == "success"),
                    "fail": sum(1 for c in candidates if c["status"] == "fail"),
                    "best_id": best["id"],
                },
            },
        )

    @staticmethod
    def _build_candidates(
        parallel_results: dict | list,
    ) -> list[dict]:
        """Build a list of candidate dicts from parallel_results."""
        candidates: list[dict] = []
        if isinstance(parallel_results, dict):
            for branch_id, result in parallel_results.items():
                if isinstance(result, dict):
                    candidates.append({
                        "id": branch_id,
                        "status": result.get("status", "unknown"),
                        "score": result.get("score", 0),
                        "data": result,
                    })
                else:
                    candidates.append({
                        "id": branch_id,
                        "status": str(result),
                        "score": 1 if result == "success" else 0,
                        "data": result,
                    })
        return candidates

    @staticmethod
    def _heuristic_select(candidates: list[dict]) -> dict:
        """Select best candidate by status priority, then score.

        Priority: success > partial_success > other > fail
        """
        status_priority = {
            "success": 3,
            "partial_success": 2,
            "skipped": 1,
            "fail": 0,
        }
        return max(
            candidates,
            key=lambda c: (
                status_priority.get(c["status"], 1),
                c.get("score", 0),
            ),
        )
