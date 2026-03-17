"""Unified LLM Client.

Routes requests to the appropriate provider adapter, applies middleware
in an onion pattern, and provides a ``from_env()`` factory for
zero-config setup.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from attractor.llm.adapters.base import ProviderAdapter
from attractor.llm.types import (
    AbortError,
    ProviderError,
    Request,
    Response,
    StreamEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Middleware protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Middleware(Protocol):
    """Middleware that wraps request/response processing.

    Middleware is applied in *onion* order: the request phase runs in
    list order, and the response phase runs in reverse order.
    """

    async def on_request(self, request: Request) -> Request:
        """Transform the outgoing request (called in list order)."""
        ...

    async def on_response(self, request: Request, response: Response) -> Response:
        """Transform the incoming response (called in reverse order)."""
        ...

    async def on_stream_event(self, request: Request, event: StreamEvent) -> StreamEvent:
        """Optionally transform each streaming event."""
        ...


class BaseMiddleware:
    """Convenience base with pass-through defaults."""

    async def on_request(self, request: Request) -> Request:
        return request

    async def on_response(self, request: Request, response: Response) -> Response:
        return response

    async def on_stream_event(self, request: Request, event: StreamEvent) -> StreamEvent:
        return event


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Client:
    """Unified client that dispatches to provider adapters.

    Parameters
    ----------
    providers:
        Mapping of provider name to adapter instance.
    default_provider:
        Provider to use when the request does not specify one.
    middleware:
        Ordered list of middleware applied in onion fashion.
    """

    def __init__(
        self,
        providers: dict[str, ProviderAdapter] | None = None,
        default_provider: str | None = None,
        middleware: list[Middleware | BaseMiddleware] | None = None,
    ) -> None:
        self._providers: dict[str, ProviderAdapter] = providers or {}
        self._default_provider = default_provider
        self._middleware: list[Middleware | BaseMiddleware] = middleware or []

    # -- factory ------------------------------------------------------------

    @classmethod
    def from_env(cls) -> Client:
        """Build a :class:`Client` by reading environment variables.

        Recognised variables:

        * ``ANTHROPIC_API_KEY`` -- registers an Anthropic adapter
        * ``OPENAI_API_KEY``    -- registers an OpenAI adapter
        * ``GEMINI_API_KEY``    -- registers a Gemini adapter

        Each adapter also honours ``*_BASE_URL`` overrides.
        """
        providers: dict[str, ProviderAdapter] = {}
        default: str | None = None

        # -- Anthropic
        if api_key := os.environ.get("ANTHROPIC_API_KEY"):
            from attractor.llm.adapters.anthropic import AnthropicAdapter

            providers["anthropic"] = AnthropicAdapter(
                api_key=api_key,
                base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            )
            if default is None:
                default = "anthropic"

        # -- OpenAI
        if api_key := os.environ.get("OPENAI_API_KEY"):
            from attractor.llm.adapters.openai import OpenAIAdapter

            providers["openai"] = OpenAIAdapter(
                api_key=api_key,
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com"),
                organization=os.environ.get("OPENAI_ORG_ID"),
                project=os.environ.get("OPENAI_PROJECT_ID"),
            )
            if default is None:
                default = "openai"

        # -- Gemini (GEMINI_API_KEY or GOOGLE_API_KEY)
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            from attractor.llm.adapters.gemini import GeminiAdapter

            providers["gemini"] = GeminiAdapter(
                api_key=gemini_key,
                base_url=os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
            )
            if default is None:
                default = "gemini"

        return cls(providers=providers, default_provider=default)

    # -- provider management ------------------------------------------------

    def register_provider(self, name: str, adapter: ProviderAdapter) -> None:
        """Register (or replace) a provider adapter at runtime."""
        self._providers[name] = adapter
        if self._default_provider is None:
            self._default_provider = name

    def add_middleware(self, mw: Middleware | BaseMiddleware) -> None:
        """Append a middleware to the processing chain."""
        self._middleware.append(mw)

    # -- core operations ----------------------------------------------------

    async def complete(self, request: Request) -> Response:
        """Send *request* and return a complete :class:`Response`."""
        if request.abort_signal:
            request.abort_signal.check()
        adapter = self._resolve_provider(request)
        request = await self._apply_request_middleware(request)
        response = await adapter.complete(request)
        if request.abort_signal:
            request.abort_signal.check()
        response = await self._apply_response_middleware(request, response)
        return response

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send *request* and yield :class:`StreamEvent` items."""
        if request.abort_signal:
            request.abort_signal.check()
        adapter = self._resolve_provider(request)
        request = await self._apply_request_middleware(request)
        async for event in adapter.stream(request):
            if request.abort_signal:
                request.abort_signal.check()
            event = await self._apply_stream_middleware(request, event)
            yield event

    async def close(self) -> None:
        """Close all registered provider adapters."""
        for adapter in self._providers.values():
            try:
                await adapter.close()
            except Exception:
                logger.warning("Error closing adapter %s", adapter.name, exc_info=True)

    # -- internals ----------------------------------------------------------

    def _resolve_provider(self, request: Request) -> ProviderAdapter:
        name = request.provider or self._default_provider
        if name is None:
            raise ProviderError(
                "No provider specified and no default provider configured.",
                provider=None,
            )
        adapter = self._providers.get(name)
        if adapter is None:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise ProviderError(
                f"Unknown provider '{name}'. Available: {available}",
                provider=name,
            )
        return adapter

    async def _apply_request_middleware(self, request: Request) -> Request:
        for mw in self._middleware:
            request = await mw.on_request(request)
        return request

    async def _apply_response_middleware(
        self, request: Request, response: Response,
    ) -> Response:
        for mw in reversed(self._middleware):
            response = await mw.on_response(request, response)
        return response

    async def _apply_stream_middleware(
        self, request: Request, event: StreamEvent,
    ) -> StreamEvent:
        for mw in reversed(self._middleware):
            event = await mw.on_stream_event(request, event)
        return event
