"""ManagerLoopHandler per Section 4.11: supervisor loop with observe/steer/wait cycles."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..graph import Outcome, StageStatus
from .base import Handler

if TYPE_CHECKING:
    from ..context import Context
    from ..graph import Graph, Node


class ManagerLoopHandler(Handler):
    """Supervisor loop handler that cycles through observe / steer / wait.

    The manager inspects context for sub-task status and decides whether
    to continue, adjust, or terminate the loop.

    Configuration via node attrs:
      - ``max_cycles``: maximum number of observe/steer cycles (default 10)
      - ``wait_seconds``: seconds to wait between cycles (default 1)
      - ``completion_key``: context key to check for completion signal
    """

    def execute(
        self,
        node: "Node",
        context: "Context",
        graph: "Graph",
        logs_root: Path | None = None,
    ) -> Outcome:
        max_cycles = int(node.attrs.get("max_cycles", "10"))
        wait_seconds = float(node.attrs.get("wait_seconds", "1"))
        completion_key = node.attrs.get("completion_key", "manager_done")

        for cycle in range(max_cycles):
            # --- Observe ---
            # Check if a completion signal has been set in context
            done_signal = context.get(completion_key, False)
            if done_signal:
                context.append_log(
                    f"[manager:{node.id}] Cycle {cycle}: completion signal received."
                )
                return Outcome(
                    status=StageStatus.SUCCESS,
                    notes=f"Manager completed after {cycle + 1} cycle(s).",
                    context_updates={"manager_cycles": cycle + 1},
                )

            # Check parallel results if available
            parallel_results = context.get("parallel_results", {})
            if isinstance(parallel_results, dict) and parallel_results:
                all_done = all(
                    v in ("success", "fail") for v in parallel_results.values()
                )
                if all_done:
                    any_fail = any(v == "fail" for v in parallel_results.values())
                    context.append_log(
                        f"[manager:{node.id}] Cycle {cycle}: all branches complete."
                    )
                    return Outcome(
                        status=StageStatus.PARTIAL_SUCCESS if any_fail else StageStatus.SUCCESS,
                        notes=f"Manager observed all branches complete at cycle {cycle + 1}.",
                        context_updates={"manager_cycles": cycle + 1},
                    )

            # --- Steer ---
            context.append_log(
                f"[manager:{node.id}] Cycle {cycle}: observing, no completion yet."
            )
            context.set("manager_cycle", cycle + 1)

            # --- Wait ---
            if cycle < max_cycles - 1:
                time.sleep(wait_seconds)

        # Exhausted cycles
        return Outcome(
            status=StageStatus.PARTIAL_SUCCESS,
            notes=f"Manager reached max cycles ({max_cycles}) without completion signal.",
            context_updates={"manager_cycles": max_cycles},
        )
