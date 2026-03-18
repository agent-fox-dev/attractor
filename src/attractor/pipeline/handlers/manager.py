"""ManagerLoopHandler per Section 4.11: supervisor loop with observe/steer/wait cycles."""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..graph import Outcome, StageStatus
from .base import Handler

if TYPE_CHECKING:
    from ..context import Context
    from ..events import EventEmitter
    from ..graph import Graph, Node

logger = logging.getLogger(__name__)


_DURATION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)?", re.IGNORECASE)


def _parse_duration(value: str) -> float:
    """Parse a duration string into seconds.

    Supports: ``1s``, ``500ms``, ``2m``, ``1h``, or plain number (seconds).
    """
    m = _DURATION_PATTERN.match(value.strip())
    if not m:
        try:
            return float(value)
        except ValueError:
            return 1.0
    num = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    if unit == "ms":
        return num / 1000.0
    if unit == "m":
        return num * 60.0
    if unit == "h":
        return num * 3600.0
    return num


class ManagerLoopHandler(Handler):
    """Supervisor loop handler that cycles through observe / steer / wait.

    The manager inspects context for sub-task status and decides whether
    to continue, adjust, or terminate the loop.

    Configuration via node attrs:
      - ``max_cycles``: maximum number of observe/steer cycles (default 10)
      - ``wait_seconds`` or ``manager.poll_interval``: wait between cycles
      - ``completion_key``: context key to check for completion signal
      - ``manager.stop_condition``: expression evaluated to check if loop should stop
    """

    def execute(
        self,
        node: "Node",
        context: "Context",
        graph: "Graph",
        logs_root: Path | None = None,
        emitter: "EventEmitter | None" = None,
    ) -> Outcome:
        max_cycles = int(node.attrs.get("manager.max_cycles", "") or node.attrs.get("max_cycles", "10"))
        # Support both manager.poll_interval (duration string) and wait_seconds (plain float)
        poll_interval_str = node.attrs.get("manager.poll_interval", "")
        if poll_interval_str:
            wait_seconds = _parse_duration(poll_interval_str)
        else:
            wait_seconds = float(node.attrs.get("wait_seconds", "1"))
        completion_key = node.attrs.get("completion_key", "manager_done")
        stop_condition = node.attrs.get("manager.stop_condition", "")
        actions = [a.strip() for a in node.attrs.get("manager.actions", "observe,wait").split(",")]

        # Child pipeline support
        child_dotfile = node.attrs.get("stack.child_dotfile", "") or graph.attrs.get("stack.child_dotfile", "")
        child_autostart = node.attrs.get("stack.child_autostart", "true") != "false"
        child_proc: subprocess.Popen | None = None

        if child_dotfile and child_autostart:
            child_workdir = node.attrs.get("stack.child_workdir", "") or "."
            # Inject parent context into child via environment variables
            child_env = dict(__import__("os").environ)
            for key, value in context.snapshot()["values"].items():
                child_env[f"ATTRACTOR_PARENT_{key.upper().replace('.', '_')}"] = str(value)
            child_env["ATTRACTOR_PARENT_GOAL"] = graph.goal or ""
            child_env["ATTRACTOR_PARENT_NODE"] = node.id
            try:
                child_proc = subprocess.Popen(
                    ["attractor", "run", child_dotfile],
                    cwd=child_workdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=child_env,
                )
                context.set("context.stack.child.status", "running")
                context.set("context.stack.child.pid", str(child_proc.pid))
                logger.info("Started child pipeline %s (pid=%d)", child_dotfile, child_proc.pid)
            except (OSError, FileNotFoundError) as exc:
                logger.warning("Failed to start child pipeline: %s", exc)
                context.set("context.stack.child.status", "failed")

        for cycle in range(max_cycles):
            # --- Observe ---
            # Check child process if running
            if child_proc is not None and "observe" in actions:
                retcode = child_proc.poll()
                if retcode is not None:
                    child_status = "completed" if retcode == 0 else "failed"
                    child_outcome = "success" if retcode == 0 else "fail"
                    context.set("context.stack.child.status", child_status)
                    context.set("context.stack.child.outcome", child_outcome)
                    # Merge child stdout/stderr back into parent context
                    child_stdout = child_proc.stdout.read().decode("utf-8", errors="replace") if child_proc.stdout else ""
                    child_stderr = child_proc.stderr.read().decode("utf-8", errors="replace") if child_proc.stderr else ""
                    if child_stdout:
                        context.set("context.stack.child.stdout", child_stdout[:10000])
                    if child_stderr:
                        context.set("context.stack.child.stderr", child_stderr[:5000])
                    if retcode == 0:
                        return Outcome(
                            status=StageStatus.SUCCESS,
                            notes=f"Child pipeline completed at cycle {cycle + 1}.",
                            context_updates={"manager_cycles": cycle + 1, "child_outcome": child_outcome},
                        )
                    else:
                        return Outcome(
                            status=StageStatus.FAIL,
                            failure_reason=f"Child pipeline failed (exit {retcode}).",
                            context_updates={"manager_cycles": cycle + 1, "child_outcome": child_outcome},
                        )

            # Check context-based child status
            child_ctx_status = context.get("context.stack.child.status", "")
            if child_ctx_status in ("completed", "failed"):
                child_outcome = context.get("context.stack.child.outcome", "")
                if child_outcome == "success":
                    return Outcome(
                        status=StageStatus.SUCCESS,
                        notes="Child completed.",
                        context_updates={"manager_cycles": cycle + 1},
                    )
                if child_ctx_status == "failed":
                    return Outcome(
                        status=StageStatus.FAIL,
                        failure_reason="Child failed.",
                        context_updates={"manager_cycles": cycle + 1},
                    )

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

            # Check stop_condition expression
            if stop_condition:
                if self._evaluate_stop_condition(stop_condition, context):
                    context.append_log(
                        f"[manager:{node.id}] Cycle {cycle}: stop condition met."
                    )
                    return Outcome(
                        status=StageStatus.SUCCESS,
                        notes=f"Manager stop condition met at cycle {cycle + 1}.",
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
            if "wait" in actions and cycle < max_cycles - 1:
                time.sleep(wait_seconds)

        # Exhausted cycles — clean up child process if still running
        if child_proc is not None and child_proc.poll() is None:
            child_proc.terminate()
            try:
                child_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child_proc.kill()

        return Outcome(
            status=StageStatus.PARTIAL_SUCCESS,
            notes=f"Manager reached max cycles ({max_cycles}) without completion signal.",
            context_updates={"manager_cycles": max_cycles},
        )

    def _evaluate_stop_condition(self, condition: str, context: "Context") -> bool:
        """Evaluate a simple stop condition against context values.

        Supports: ``key=value``, ``key!=value``, ``key`` (truthy check).
        """
        condition = condition.strip()
        if "!=" in condition:
            key, expected = condition.split("!=", 1)
            actual = str(context.get(key.strip(), ""))
            return actual != expected.strip()
        if "=" in condition:
            key, expected = condition.split("=", 1)
            actual = str(context.get(key.strip(), ""))
            return actual == expected.strip()
        # Truthy check
        return bool(context.get(condition, False))
