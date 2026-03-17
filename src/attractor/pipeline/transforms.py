"""AST transforms per Section 9 of the Attractor spec.

Transforms modify a parsed Graph before execution.  The public entry point
``prepare_pipeline`` parses DOT, applies transforms, and validates.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .graph import Graph
from .parser import parse_dot
from .stylesheet import apply_stylesheet
from .validation import Diagnostic, validate


@runtime_checkable
class Transform(Protocol):
    """Protocol for graph transforms."""

    def apply(self, graph: Graph) -> Graph:
        """Apply the transform to *graph* (may mutate in place) and return it."""
        ...


class VariableExpansionTransform:
    """Replaces ``$goal`` references in node prompts with the graph's goal."""

    def apply(self, graph: Graph) -> Graph:
        goal = graph.goal
        if not goal:
            return graph
        for node in graph.nodes.values():
            if "$goal" in node.prompt:
                node.prompt = node.prompt.replace("$goal", goal)
        return graph


class StylesheetApplicationTransform:
    """Applies the graph's ``model_stylesheet`` to nodes."""

    def apply(self, graph: Graph) -> Graph:
        return apply_stylesheet(graph)


class PreambleTransform:
    """Synthesizes context carryover text for stages that do not use ``full`` fidelity.

    For nodes with a fidelity mode other than ``full``, this transform
    prepends a preamble instruction to the prompt indicating how context
    from previous stages should be summarized or truncated.
    """

    _PREAMBLE_MAP: dict[str, str] = {
        "truncate": (
            "[Context fidelity: truncate] "
            "Previous stage context has been truncated to fit within the context window. "
            "Focus on the most recent information."
        ),
        "compact": (
            "[Context fidelity: compact] "
            "Previous stage context has been compacted. "
            "Key decisions and outputs are preserved; verbose logs are omitted."
        ),
        "summary:low": (
            "[Context fidelity: summary:low] "
            "You are working with a brief summary of previous stages. "
            "Only the final outcomes are included."
        ),
        "summary:medium": (
            "[Context fidelity: summary:medium] "
            "You are working with a medium-detail summary of previous stages. "
            "Key decisions, outputs, and notable issues are included."
        ),
        "summary:high": (
            "[Context fidelity: summary:high] "
            "You are working with a detailed summary of previous stages. "
            "Most decisions, outputs, reasoning, and issues are included."
        ),
    }

    def apply(self, graph: Graph) -> Graph:
        for node in graph.nodes.values():
            fidelity = node.fidelity or node.attrs.get("fidelity", "")
            if not fidelity or fidelity == "full":
                continue
            preamble = self._PREAMBLE_MAP.get(fidelity)
            if preamble and node.prompt:
                node.prompt = f"{preamble}\n\n{node.prompt}"
        return graph


# Default transform pipeline
DEFAULT_TRANSFORMS: list[Transform] = [
    VariableExpansionTransform(),
    StylesheetApplicationTransform(),
    PreambleTransform(),
]


def prepare_pipeline(
    dot_source: str,
    transforms: list[Transform] | None = None,
) -> tuple[Graph, list[Diagnostic]]:
    """Parse DOT source, apply transforms, validate, and return the result.

    Parameters
    ----------
    dot_source:
        The DOT digraph source string.
    transforms:
        Optional list of transforms to apply.  If *None*, the default
        transforms (variable expansion + stylesheet) are used.

    Returns
    -------
    tuple[Graph, list[Diagnostic]]
        The transformed graph and a list of validation diagnostics.
    """
    graph = parse_dot(dot_source)

    pipeline = transforms if transforms is not None else list(DEFAULT_TRANSFORMS)
    for transform in pipeline:
        graph = transform.apply(graph)

    diagnostics = validate(graph)
    return graph, diagnostics
