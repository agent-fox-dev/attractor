"""Abstract base class for provider adapters.

Every LLM provider (Anthropic, OpenAI, Gemini, ...) implements this
interface so the unified client can route requests without caring about
wire-protocol differences.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator

from attractor.llm.types import Request, Response, StreamEvent


class ProviderAdapter(abc.ABC):
    """Protocol that every provider adapter must satisfy."""

    # -- identity -----------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short provider identifier (e.g. ``"anthropic"``)."""

    # -- lifecycle ----------------------------------------------------------

    async def initialize(self) -> None:
        """Optional async startup hook (connection warming, etc.)."""

    async def close(self) -> None:
        """Release any held resources (HTTP clients, etc.)."""

    # -- core operations ----------------------------------------------------

    @abc.abstractmethod
    async def complete(self, request: Request) -> Response:
        """Send *request* and return a complete :class:`Response`."""

    @abc.abstractmethod
    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send *request* and yield :class:`StreamEvent` items."""
        # The method must be an async generator in concrete implementations.
        # This stub exists only to satisfy ``abc.abstractmethod``.
        yield  # type: ignore[misc]  # pragma: no cover

    # -- capability queries -------------------------------------------------

    def supports_tool_choice(self, mode: str) -> bool:
        """Return whether this provider supports the given *mode*.

        Common modes: ``"auto"``, ``"none"``, ``"required"``, ``"any"``,
        or a specific tool name.
        """
        return mode in {"auto", "none", "required"}
