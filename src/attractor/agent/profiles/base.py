"""ProviderProfile interface per Section 3.2 of the coding-agent-loop-spec."""

from __future__ import annotations

from abc import ABC, abstractmethod

from attractor.llm.types import ToolDefinition
from attractor.agent.tools.registry import ToolRegistry
from attractor.agent.execution.base import ExecutionEnvironment


class ProviderProfile(ABC):
    """Abstract base for provider-specific tool and prompt profiles.

    Each profile encapsulates:
    - The model identifier and provider name.
    - A tool registry with the provider's native tools.
    - A system prompt builder aligned with the provider's conventions.
    - Capability flags for the host to introspect.
    """

    id: str
    model: str
    tool_registry: ToolRegistry
    supports_reasoning: bool
    supports_streaming: bool
    supports_parallel_tool_calls: bool
    context_window_size: int

    @abstractmethod
    def build_system_prompt(
        self, environment: ExecutionEnvironment, project_docs: str = ""
    ) -> str:
        """Build the full system prompt for this provider profile.

        Parameters
        ----------
        environment:
            The execution environment (provides working dir, platform, etc.).
        project_docs:
            Discovered project documentation content (AGENTS.md, etc.).
        """
        ...

    @abstractmethod
    def tools(self) -> list[ToolDefinition]:
        """Return the tool definitions for this profile."""
        ...

    def provider_options(self) -> dict | None:
        """Return provider-specific options for the LLM Request, or None."""
        return None
