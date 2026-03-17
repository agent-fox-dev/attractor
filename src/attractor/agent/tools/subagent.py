"""Subagent tools per Section 7 of the coding-agent-loop-spec.

Provides spawn_agent, send_input, wait, and close_agent tools.
Use ``register_subagent_tools(registry, session_factory)`` to register them.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from attractor.llm.types import ToolDefinition
from attractor.agent.execution.base import ExecutionEnvironment
from attractor.agent.tools.registry import RegisteredTool, ToolRegistry


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SubAgentResult:
    """Result returned when a subagent completes."""

    output: str
    success: bool
    turns_used: int


@dataclass
class SubAgentHandle:
    """Handle to a running subagent session."""

    id: str
    session: Any  # Session -- forward reference to avoid circular import
    status: str = "running"  # "running" | "completed" | "failed"
    task: asyncio.Task | None = None
    result: SubAgentResult | None = None


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

SPAWN_AGENT_DEF = ToolDefinition(
    name="spawn_agent",
    description=(
        "Spawn a subagent to handle a scoped task autonomously. "
        "The subagent runs its own agentic loop with its own conversation "
        "history but shares the parent's execution environment (same filesystem)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Natural language task description for the subagent.",
            },
            "working_dir": {
                "type": "string",
                "description": "Subdirectory to scope the agent to (optional).",
            },
            "model": {
                "type": "string",
                "description": "Model override. Defaults to the parent's model.",
            },
            "max_turns": {
                "type": "integer",
                "description": "Turn limit for the subagent. Default: 0 (unlimited).",
            },
        },
        "required": ["task"],
    },
)

SEND_INPUT_DEF = ToolDefinition(
    name="send_input",
    description="Send a message to a running subagent.",
    parameters={
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "ID of the subagent.",
            },
            "message": {
                "type": "string",
                "description": "Message to send to the subagent.",
            },
        },
        "required": ["agent_id", "message"],
    },
)

WAIT_DEF = ToolDefinition(
    name="wait",
    description="Wait for a subagent to complete and return its result.",
    parameters={
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "ID of the subagent to wait for.",
            },
        },
        "required": ["agent_id"],
    },
)

CLOSE_AGENT_DEF = ToolDefinition(
    name="close_agent",
    description="Terminate a subagent.",
    parameters={
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "ID of the subagent to close.",
            },
        },
        "required": ["agent_id"],
    },
)


# ---------------------------------------------------------------------------
# In-module storage for active subagent handles.
# Each parent session keeps its own map in session.subagents, but we
# also support a factory-scoped registry for the tool executors.
# ---------------------------------------------------------------------------


def _make_executors(
    session_factory: Callable[..., Any],
) -> dict[str, Callable]:
    """Build executor functions that capture *session_factory*.

    ``session_factory`` is a callable with the signature::

        session_factory(
            task: str,
            parent_session: Session,
            working_dir: str | None = None,
            model: str | None = None,
            max_turns: int = 0,
        ) -> Session

    It creates a new child ``Session`` that shares the parent's execution
    environment but has independent history.
    """

    # Shared handle map keyed by agent_id
    handles: dict[str, SubAgentHandle] = {}

    # -- spawn_agent -------------------------------------------------------

    def exec_spawn_agent(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
        task: str = arguments["task"]
        working_dir: str | None = arguments.get("working_dir")
        model: str | None = arguments.get("model")
        max_turns: int = arguments.get("max_turns", 0)

        # Enforce depth limit: check if parent session depth is at max
        parent_depth = getattr(env, "_parent_depth", 0)
        max_depth = getattr(env, "_max_subagent_depth", 1)
        if parent_depth >= max_depth:
            return (
                f"Error: Cannot spawn subagent — depth limit reached "
                f"(current depth: {parent_depth}, max: {max_depth})."
            )

        agent_id = str(uuid.uuid4())

        child_session = session_factory(
            task=task,
            working_dir=working_dir,
            model=model,
            max_turns=max_turns,
        )

        handle = SubAgentHandle(id=agent_id, session=child_session)
        handles[agent_id] = handle

        # Launch the subagent asynchronously
        async def _run() -> None:
            try:
                await child_session.submit(task)
                handle.status = "completed"
                # Gather output from last assistant turn
                output_parts: list[str] = []
                for turn in reversed(child_session.history):
                    if hasattr(turn, "content") and turn.content:
                        output_parts.append(turn.content)
                        break
                turns_used = len(
                    [t for t in child_session.history if hasattr(t, "tool_calls")]
                )
                handle.result = SubAgentResult(
                    output="\n".join(output_parts) if output_parts else "(no output)",
                    success=True,
                    turns_used=turns_used,
                )
            except Exception as exc:
                handle.status = "failed"
                handle.result = SubAgentResult(
                    output=f"Subagent error: {exc}",
                    success=False,
                    turns_used=0,
                )

        try:
            loop = asyncio.get_running_loop()
            handle.task = loop.create_task(_run())
        except RuntimeError:
            # No running event loop -- run synchronously (unlikely in normal use)
            asyncio.run(_run())

        return f"Spawned subagent {agent_id}. Status: {handle.status}"

    # -- send_input --------------------------------------------------------

    def exec_send_input(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
        agent_id: str = arguments["agent_id"]
        message: str = arguments["message"]

        handle = handles.get(agent_id)
        if handle is None:
            return f"Error: Unknown subagent ID '{agent_id}'."
        if handle.status != "running":
            return f"Error: Subagent {agent_id} is not running (status: {handle.status})."

        handle.session.follow_up(message)
        return f"Message sent to subagent {agent_id}."

    # -- wait --------------------------------------------------------------

    def exec_wait(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
        agent_id: str = arguments["agent_id"]

        handle = handles.get(agent_id)
        if handle is None:
            return f"Error: Unknown subagent ID '{agent_id}'."

        if handle.task is not None and not handle.task.done():
            # Block until done -- this works inside an async executor because
            # the session loop will await tool results.
            try:
                loop = asyncio.get_running_loop()
                # We cannot block the event loop, so we return a status
                # indicating the agent is still running.
                if not handle.task.done():
                    return (
                        f"Subagent {agent_id} is still running. "
                        f"Call 'wait' again later or use 'close_agent' to terminate it."
                    )
            except RuntimeError:
                pass

        if handle.result is not None:
            r = handle.result
            return (
                f"Subagent {agent_id} {handle.status}.\n"
                f"Success: {r.success}\n"
                f"Turns used: {r.turns_used}\n"
                f"Output:\n{r.output}"
            )

        return f"Subagent {agent_id} status: {handle.status}. No result yet."

    # -- close_agent -------------------------------------------------------

    def exec_close_agent(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
        agent_id: str = arguments["agent_id"]

        handle = handles.get(agent_id)
        if handle is None:
            return f"Error: Unknown subagent ID '{agent_id}'."

        # Cancel the task if still running
        if handle.task is not None and not handle.task.done():
            handle.task.cancel()

        # Abort the session
        handle.session.abort()
        handle.status = "closed"

        # Clean up
        handles.pop(agent_id, None)
        return f"Subagent {agent_id} closed."

    return {
        "spawn_agent": exec_spawn_agent,
        "send_input": exec_send_input,
        "wait": exec_wait,
        "close_agent": exec_close_agent,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_subagent_tools(
    registry: ToolRegistry,
    session_factory: Callable[..., Any],
) -> None:
    """Register all four subagent tools into *registry*.

    Parameters
    ----------
    registry:
        The tool registry to add tools to.
    session_factory:
        A callable that creates child sessions.  Signature::

            session_factory(
                task: str,
                working_dir: str | None = None,
                model: str | None = None,
                max_turns: int = 0,
            ) -> Session
    """
    executors = _make_executors(session_factory)

    registry.register(
        RegisteredTool(definition=SPAWN_AGENT_DEF, executor=executors["spawn_agent"])
    )
    registry.register(
        RegisteredTool(definition=SEND_INPUT_DEF, executor=executors["send_input"])
    )
    registry.register(
        RegisteredTool(definition=WAIT_DEF, executor=executors["wait"])
    )
    registry.register(
        RegisteredTool(definition=CLOSE_AGENT_DEF, executor=executors["close_agent"])
    )
