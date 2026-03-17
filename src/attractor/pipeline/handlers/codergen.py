"""CodergenHandler per Section 4.5 of the Attractor spec.

Expands ``$goal`` in the node prompt, optionally delegates to a
CodergenBackend, and writes prompt/response/status files to the stage
log directory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..graph import Outcome, StageStatus
from .base import Handler

if TYPE_CHECKING:
    from ..context import Context
    from ..graph import Graph, Node


@runtime_checkable
class CodergenBackend(Protocol):
    """Protocol for pluggable codergen backends."""

    def run(self, node: "Node", prompt: str, context: "Context") -> str | Outcome:
        """Execute a code-generation request.

        Returns either a response string (treated as SUCCESS) or a full
        Outcome for richer status reporting.
        """
        ...


class CodergenHandler(Handler):
    """Handles codergen / LLM code-generation nodes."""

    def __init__(self, backend: CodergenBackend | None = None) -> None:
        self._backend = backend

    def execute(
        self,
        node: "Node",
        context: "Context",
        graph: "Graph",
        logs_root: Path | None = None,
    ) -> Outcome:
        # Expand $goal in the prompt
        prompt = node.prompt
        if graph.goal and "$goal" in prompt:
            prompt = prompt.replace("$goal", graph.goal)

        # Prepare stage directory for logs
        stage_dir: Path | None = None
        if logs_root is not None:
            stage_dir = logs_root / node.id
            stage_dir.mkdir(parents=True, exist_ok=True)
            (stage_dir / "prompt.md").write_text(prompt, encoding="utf-8")

        # Execute
        start_time = time.time()
        try:
            if self._backend is not None:
                result = self._backend.run(node, prompt, context)
                if isinstance(result, Outcome):
                    outcome = result
                    response_text = result.notes or "(outcome returned)"
                else:
                    response_text = str(result)
                    outcome = Outcome(
                        status=StageStatus.SUCCESS,
                        notes=response_text,
                        context_updates={"last_response": response_text},
                    )
            else:
                # Simulation mode: no backend configured
                response_text = f"[simulated] No backend configured for node '{node.id}'."
                outcome = Outcome(
                    status=StageStatus.SUCCESS,
                    notes=response_text,
                    context_updates={"last_response": response_text},
                )
        except Exception as exc:
            response_text = f"Error: {exc}"
            outcome = Outcome(
                status=StageStatus.FAIL,
                failure_reason=str(exc),
                notes=response_text,
            )

        elapsed = time.time() - start_time

        # Write logs
        if stage_dir is not None:
            (stage_dir / "response.md").write_text(response_text, encoding="utf-8")
            status_data = {
                "node_id": node.id,
                "status": outcome.status.value,
                "elapsed_seconds": round(elapsed, 3),
                "failure_reason": outcome.failure_reason,
            }
            (stage_dir / "status.json").write_text(
                json.dumps(status_data, indent=2), encoding="utf-8"
            )

        # Apply context updates
        if outcome.context_updates:
            context.apply_updates(outcome.context_updates)

        return outcome
