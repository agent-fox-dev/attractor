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
        parallel_results = context.get("parallel.results", {}) or context.get("parallel_results", {})

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
        all_fail = all(c["status"] == "fail" for c in candidates)
        all_success = all(c["status"] == "success" for c in candidates)

        if all_fail:
            status = StageStatus.FAIL
        elif all_success:
            status = StageStatus.SUCCESS
        else:
            status = StageStatus.PARTIAL_SUCCESS

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
        """Select best candidate per spec Section 4.9.

        Sort by (outcome_rank ASC, -score DESC, id ASC).
        outcome_rank: SUCCESS=0, PARTIAL_SUCCESS=1, RETRY=2, FAIL=3.
        """
        outcome_rank = {
            "success": 0,
            "partial_success": 1,
            "retry": 2,
            "fail": 3,
        }
        return min(
            candidates,
            key=lambda c: (
                outcome_rank.get(c["status"], 3),
                -c.get("score", 0),
                c.get("id", ""),
            ),
        )
