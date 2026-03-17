"""ToolRegistry per Section 3.8 of the coding-agent-loop-spec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from attractor.llm.types import ToolDefinition
from attractor.agent.execution.base import ExecutionEnvironment


@dataclass
class RegisteredTool:
    """A tool definition paired with its executor function.

    The executor signature is:
        ``(arguments: dict[str, Any], env: ExecutionEnvironment) -> str``
    """

    definition: ToolDefinition
    executor: Callable[[dict[str, Any], ExecutionEnvironment], str]


class ToolRegistry:
    """Registry of tools available to a provider profile."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        """Register (or replace) a tool.  Latest-wins on name collisions."""
        self._tools[tool.definition.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name.  Silently ignores unknown names."""
        self._tools.pop(name, None)

    def get(self, name: str) -> RegisteredTool | None:
        """Look up a registered tool by name."""
        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        """Return all tool definitions (in registration order)."""
        return [t.definition for t in self._tools.values()]

    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())
