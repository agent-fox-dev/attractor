"""ToolHandler per Section 4.10: executes shell commands from node attrs."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..graph import Outcome, StageStatus
from .base import Handler

if TYPE_CHECKING:
    from ..context import Context
    from ..graph import Graph, Node


class ToolHandler(Handler):
    """Executes a shell command specified in the node's attributes.

    The command is taken from ``node.attrs["command"]`` (or ``node.prompt``
    as fallback).  ``$goal`` is expanded.  stdout/stderr are captured and
    stored in context.
    """

    def execute(
        self,
        node: "Node",
        context: "Context",
        graph: "Graph",
        logs_root: Path | None = None,
    ) -> Outcome:
        command = node.attrs.get("command", "") or node.prompt
        if not command:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Tool node '{node.id}' has no command attribute.",
            )

        # Expand $goal
        if graph.goal and "$goal" in command:
            command = command.replace("$goal", graph.goal)

        # Prepare stage directory
        stage_dir: Path | None = None
        if logs_root is not None:
            stage_dir = logs_root / node.id
            stage_dir.mkdir(parents=True, exist_ok=True)

        timeout_secs: float | None = None
        if node.timeout:
            try:
                timeout_secs = float(node.timeout)
            except ValueError:
                pass

        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_secs,
            )
            elapsed = time.time() - start_time

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if stage_dir is not None:
                (stage_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
                (stage_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
                (stage_dir / "status.json").write_text(json.dumps({
                    "node_id": node.id,
                    "command": command,
                    "return_code": result.returncode,
                    "elapsed_seconds": round(elapsed, 3),
                }, indent=2), encoding="utf-8")

            if result.returncode == 0:
                return Outcome(
                    status=StageStatus.SUCCESS,
                    notes=stdout[:2000],
                    context_updates={
                        "tool_stdout": stdout,
                        "tool_stderr": stderr,
                        "tool_return_code": result.returncode,
                    },
                )
            else:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Command exited with code {result.returncode}: {stderr[:500]}",
                    notes=stdout[:2000],
                    context_updates={
                        "tool_stdout": stdout,
                        "tool_stderr": stderr,
                        "tool_return_code": result.returncode,
                    },
                )

        except subprocess.TimeoutExpired:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Command timed out after {timeout_secs}s.",
            )
        except Exception as exc:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Failed to execute command: {exc}",
            )
