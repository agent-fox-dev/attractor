"""Model catalog — known models and lookup helpers.

Provides a registry of well-known LLM models with their capabilities,
pricing, and context-window metadata.  Used by the client layer to
validate model identifiers and select appropriate providers.
"""

from __future__ import annotations

from attractor.llm.types import ModelInfo


# ---------------------------------------------------------------------------
# Known models
# ---------------------------------------------------------------------------

MODELS: list[ModelInfo] = [
    # -- Anthropic ----------------------------------------------------------
    ModelInfo(
        id="claude-opus-4-6",
        provider="anthropic",
        display_name="Claude Opus 4.6",
        context_window=200_000,
        max_output=32_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=15.0,
        output_cost_per_million=75.0,
        aliases=["claude-opus"],
    ),
    ModelInfo(
        id="claude-sonnet-4-5",
        provider="anthropic",
        display_name="Claude Sonnet 4.5",
        context_window=200_000,
        max_output=16_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        aliases=["claude-sonnet"],
    ),
    # -- OpenAI -------------------------------------------------------------
    ModelInfo(
        id="gpt-5.2",
        provider="openai",
        display_name="GPT-5.2",
        context_window=256_000,
        max_output=32_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=10.0,
        output_cost_per_million=30.0,
        aliases=[],
    ),
    ModelInfo(
        id="gpt-5.2-mini",
        provider="openai",
        display_name="GPT-5.2 Mini",
        context_window=256_000,
        max_output=16_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=1.5,
        output_cost_per_million=6.0,
        aliases=[],
    ),
    ModelInfo(
        id="gpt-5.2-codex",
        provider="openai",
        display_name="GPT-5.2 Codex",
        context_window=256_000,
        max_output=32_000,
        supports_tools=True,
        supports_vision=False,
        supports_reasoning=True,
        input_cost_per_million=12.0,
        output_cost_per_million=36.0,
        aliases=[],
    ),
    # -- Gemini -------------------------------------------------------------
    ModelInfo(
        id="gemini-3-pro-preview",
        provider="gemini",
        display_name="Gemini 3 Pro (Preview)",
        context_window=1_000_000,
        max_output=32_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=7.0,
        output_cost_per_million=21.0,
        aliases=["gemini-3-pro"],
    ),
    ModelInfo(
        id="gemini-3-flash-preview",
        provider="gemini",
        display_name="Gemini 3 Flash (Preview)",
        context_window=1_000_000,
        max_output=16_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=False,
        input_cost_per_million=0.5,
        output_cost_per_million=1.5,
        aliases=["gemini-3-flash"],
    ),
]

# Build a lookup index (id + aliases -> ModelInfo).
_INDEX: dict[str, ModelInfo] = {}
for _m in MODELS:
    _INDEX[_m.id] = _m
    for _alias in _m.aliases:
        _INDEX[_alias] = _m


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_model_info(model_id: str) -> ModelInfo | None:
    """Return the :class:`ModelInfo` for *model_id*, or ``None``."""
    return _INDEX.get(model_id)


def list_models(provider: str | None = None) -> list[ModelInfo]:
    """Return all known models, optionally filtered by *provider*."""
    if provider is None:
        return list(MODELS)
    return [m for m in MODELS if m.provider == provider]


def get_latest_model(
    provider: str,
    capability: str | None = None,
) -> ModelInfo | None:
    """Return the first model for *provider* matching an optional *capability*.

    *capability* can be ``"tools"``, ``"vision"``, or ``"reasoning"``.
    The "latest" model is simply the first entry in the catalog for that
    provider (the catalog is ordered newest-first by convention).
    """
    for m in MODELS:
        if m.provider != provider:
            continue
        if capability is None:
            return m
        if capability == "tools" and m.supports_tools:
            return m
        if capability == "vision" and m.supports_vision:
            return m
        if capability == "reasoning" and m.supports_reasoning:
            return m
    return None
