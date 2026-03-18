"""Pipeline validation per Section 7 of the Attractor spec.

Provides a set of built-in lint rules and a validation harness that produces
Diagnostic objects for any issues found in a Graph.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph import Edge, Graph


# ---------------------------------------------------------------------------
# Severity & Diagnostic
# ---------------------------------------------------------------------------

class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Diagnostic:
    rule: str
    severity: Severity
    message: str
    node_id: str = ""
    edge: str = ""
    fix: str = ""


# ---------------------------------------------------------------------------
# LintRule interface
# ---------------------------------------------------------------------------

class LintRule(ABC):
    """A single validation rule that inspects a Graph."""

    @abstractmethod
    def apply(self, graph: "Graph") -> list[Diagnostic]:
        ...


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------

_START_SHAPES = frozenset({"Mdiamond", "mdiamond", "circle", "point"})
_EXIT_SHAPES = frozenset({"Msquare", "msquare", "doublecircle"})
_LLM_TYPES = frozenset({"llm", "codergen", "coder"})
_KNOWN_TYPES = frozenset({
    "", "start", "exit", "llm", "codergen", "coder",
    "human", "wait_for_human", "wait.human",
    "conditional", "decision",
    "parallel", "fork",
    "fan_in", "join", "parallel.fan_in",
    "tool",
    "manager", "manager_loop", "stack.manager_loop",
})


class StartNodeRule(LintRule):
    """There must be exactly one start node."""

    def apply(self, graph: "Graph") -> list[Diagnostic]:
        starts = [n for n in graph.nodes.values()
                  if n.type == "start" or (not n.type and n.shape in _START_SHAPES)]
        if len(starts) == 0:
            return [Diagnostic(
                rule="start_node", severity=Severity.ERROR,
                message="No start node found. Add a node with type=\"start\" or shape=circle.",
                fix="Add a start node.",
            )]
        if len(starts) > 1:
            ids = ", ".join(s.id for s in starts)
            return [Diagnostic(
                rule="start_node", severity=Severity.ERROR,
                message=f"Multiple start nodes found: {ids}. Only one is allowed.",
            )]
        return []


class TerminalNodeRule(LintRule):
    """There must be at least one terminal (exit) node."""

    def apply(self, graph: "Graph") -> list[Diagnostic]:
        terminals = [n for n in graph.nodes.values()
                     if n.type == "exit"
                     or (not n.type and n.shape in _EXIT_SHAPES)
                     or len(graph.outgoing_edges(n.id)) == 0]
        if not terminals:
            return [Diagnostic(
                rule="terminal_node", severity=Severity.ERROR,
                message="No terminal node found. The pipeline may loop indefinitely.",
                fix="Add an exit node or ensure at least one node has no outgoing edges.",
            )]
        return []


class ReachabilityRule(LintRule):
    """All nodes should be reachable from the start node."""

    def apply(self, graph: "Graph") -> list[Diagnostic]:
        starts = [n for n in graph.nodes.values()
                  if n.type == "start" or (not n.type and n.shape in _START_SHAPES)]
        if not starts:
            return []  # StartNodeRule will report this

        visited: set[str] = set()
        stack = [starts[0].id]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            for edge in graph.outgoing_edges(nid):
                if edge.to_node not in visited:
                    stack.append(edge.to_node)

        unreachable = set(graph.nodes.keys()) - visited
        diags: list[Diagnostic] = []
        for nid in sorted(unreachable):
            diags.append(Diagnostic(
                rule="reachability", severity=Severity.ERROR,
                message=f"Node '{nid}' is not reachable from the start node.",
                node_id=nid,
                fix=f"Add an edge leading to '{nid}' or remove it.",
            ))
        return diags


class EdgeTargetExistsRule(LintRule):
    """Every edge must reference existing nodes."""

    def apply(self, graph: "Graph") -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for edge in graph.edges:
            if edge.from_node not in graph.nodes:
                diags.append(Diagnostic(
                    rule="edge_target_exists", severity=Severity.ERROR,
                    message=f"Edge references unknown source node '{edge.from_node}'.",
                    edge=f"{edge.from_node} -> {edge.to_node}",
                ))
            if edge.to_node not in graph.nodes:
                diags.append(Diagnostic(
                    rule="edge_target_exists", severity=Severity.ERROR,
                    message=f"Edge references unknown target node '{edge.to_node}'.",
                    edge=f"{edge.from_node} -> {edge.to_node}",
                ))
        return diags


class StartNoIncomingRule(LintRule):
    """Start nodes should not have incoming edges."""

    def apply(self, graph: "Graph") -> list[Diagnostic]:
        starts = [n for n in graph.nodes.values()
                  if n.type == "start" or (not n.type and n.shape in _START_SHAPES)]
        diags: list[Diagnostic] = []
        for s in starts:
            incoming = graph.incoming_edges(s.id)
            if incoming:
                diags.append(Diagnostic(
                    rule="start_no_incoming", severity=Severity.ERROR,
                    message=f"Start node '{s.id}' has incoming edges, which is unusual.",
                    node_id=s.id,
                ))
        return diags


class ExitNoOutgoingRule(LintRule):
    """Exit nodes should not have outgoing edges."""

    def apply(self, graph: "Graph") -> list[Diagnostic]:
        exits = [n for n in graph.nodes.values()
                 if n.type == "exit"
                 or (not n.type and n.shape in _EXIT_SHAPES and n.type != "start")]
        diags: list[Diagnostic] = []
        for e in exits:
            outgoing = graph.outgoing_edges(e.id)
            if outgoing:
                diags.append(Diagnostic(
                    rule="exit_no_outgoing", severity=Severity.ERROR,
                    message=f"Exit node '{e.id}' has outgoing edges, which is unusual.",
                    node_id=e.id,
                ))
        return diags


class ConditionSyntaxRule(LintRule):
    """Edge conditions must have valid syntax."""

    _CLAUSE_PATTERN = re.compile(
        r'^[\w.]+\s*(?:!=|=)\s*[\w."\'-]+$'
    )

    def apply(self, graph: "Graph") -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for edge in graph.edges:
            cond = edge.condition.strip()
            if not cond:
                continue
            clauses = cond.split("&&")
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    diags.append(Diagnostic(
                        rule="condition_syntax", severity=Severity.ERROR,
                        message=f"Empty clause in condition '{cond}'.",
                        edge=f"{edge.from_node} -> {edge.to_node}",
                    ))
                elif not self._CLAUSE_PATTERN.match(clause):
                    diags.append(Diagnostic(
                        rule="condition_syntax", severity=Severity.ERROR,
                        message=f"Clause '{clause}' may have invalid syntax.",
                        edge=f"{edge.from_node} -> {edge.to_node}",
                    ))
        return diags


class TypeKnownRule(LintRule):
    """Node types should be recognised."""

    def apply(self, graph: "Graph") -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for node in graph.nodes.values():
            if node.type and node.type.lower() not in _KNOWN_TYPES:
                diags.append(Diagnostic(
                    rule="type_known", severity=Severity.WARNING,
                    message=f"Node '{node.id}' has unknown type '{node.type}'.",
                    node_id=node.id,
                ))
        return diags


class PromptOnLLMNodesRule(LintRule):
    """LLM-type nodes should have a prompt attribute."""

    def apply(self, graph: "Graph") -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for node in graph.nodes.values():
            if node.type.lower() in _LLM_TYPES and not node.prompt:
                diags.append(Diagnostic(
                    rule="prompt_on_llm_nodes", severity=Severity.WARNING,
                    message=f"LLM node '{node.id}' has no prompt attribute.",
                    node_id=node.id,
                    fix="Add a prompt attribute to the node.",
                ))
        return diags


# ---------------------------------------------------------------------------
# Built-in rule registry
# ---------------------------------------------------------------------------

class RetryTargetExistsRule(LintRule):
    """retry_target and fallback_retry_target should reference existing nodes."""

    def apply(self, graph: "Graph") -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for node in graph.nodes.values():
            if node.retry_target and node.retry_target not in graph.nodes:
                diags.append(Diagnostic(
                    rule="retry_target_exists", severity=Severity.WARNING,
                    message=f"Node '{node.id}' has retry_target '{node.retry_target}' which does not exist.",
                    node_id=node.id,
                    fix=f"Create a node named '{node.retry_target}' or remove the retry_target attribute.",
                ))
            if node.fallback_retry_target and node.fallback_retry_target not in graph.nodes:
                diags.append(Diagnostic(
                    rule="retry_target_exists", severity=Severity.WARNING,
                    message=f"Node '{node.id}' has fallback_retry_target '{node.fallback_retry_target}' which does not exist.",
                    node_id=node.id,
                    fix=f"Create a node named '{node.fallback_retry_target}' or remove the attribute.",
                ))
        # Also check graph-level targets
        if graph.retry_target and graph.retry_target not in graph.nodes:
            diags.append(Diagnostic(
                rule="retry_target_exists", severity=Severity.WARNING,
                message=f"Graph retry_target '{graph.retry_target}' does not exist.",
            ))
        if graph.fallback_retry_target and graph.fallback_retry_target not in graph.nodes:
            diags.append(Diagnostic(
                rule="retry_target_exists", severity=Severity.WARNING,
                message=f"Graph fallback_retry_target '{graph.fallback_retry_target}' does not exist.",
            ))
        return diags


class GoalGateHasRetryRule(LintRule):
    """Goal-gate nodes should have a retry target so failures can be recovered."""

    def apply(self, graph: "Graph") -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for node in graph.nodes.values():
            if not node.goal_gate:
                continue
            has_retry = (
                node.retry_target
                or node.fallback_retry_target
                or graph.retry_target
                or graph.fallback_retry_target
            )
            if not has_retry:
                diags.append(Diagnostic(
                    rule="goal_gate_has_retry", severity=Severity.WARNING,
                    message=f"Goal-gate node '{node.id}' has no retry target. "
                            f"If this node fails, the pipeline cannot recover.",
                    node_id=node.id,
                    fix="Add a retry_target attribute to the node or to the graph.",
                ))
        return diags


_VALID_FIDELITY = frozenset({
    "", "full", "truncate", "compact",
    "summary:low", "summary:medium", "summary:high",
})


class FidelityValidRule(LintRule):
    """Node fidelity mode should be one of the recognized values."""

    def apply(self, graph: "Graph") -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for node in graph.nodes.values():
            fidelity = node.attrs.get("fidelity", "")
            if fidelity and fidelity not in _VALID_FIDELITY:
                diags.append(Diagnostic(
                    rule="fidelity_valid", severity=Severity.WARNING,
                    message=f"Node '{node.id}' has unknown fidelity '{fidelity}'. "
                            f"Valid values: {', '.join(sorted(_VALID_FIDELITY - {''}))}.",
                    node_id=node.id,
                ))
        return diags


class StylesheetSyntaxRule(LintRule):
    """Stylesheet attribute should parse without errors."""

    def apply(self, graph: "Graph") -> list[Diagnostic]:
        stylesheet = graph.attrs.get("stylesheet", "")
        if not stylesheet:
            return []
        try:
            from .stylesheet import parse_stylesheet
            parse_stylesheet(stylesheet)
        except Exception as exc:
            return [Diagnostic(
                rule="stylesheet_syntax", severity=Severity.ERROR,
                message=f"Stylesheet parse error: {exc}",
            )]
        return []


BUILTIN_RULES: list[LintRule] = [
    StartNodeRule(),
    TerminalNodeRule(),
    ReachabilityRule(),
    EdgeTargetExistsRule(),
    StartNoIncomingRule(),
    ExitNoOutgoingRule(),
    ConditionSyntaxRule(),
    TypeKnownRule(),
    PromptOnLLMNodesRule(),
    RetryTargetExistsRule(),
    GoalGateHasRetryRule(),
    FidelityValidRule(),
    StylesheetSyntaxRule(),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(graph: "Graph", extra_rules: list[LintRule] | None = None) -> list[Diagnostic]:
    """Run all lint rules against *graph* and return diagnostics."""
    rules = list(BUILTIN_RULES)
    if extra_rules:
        rules.extend(extra_rules)
    diagnostics: list[Diagnostic] = []
    for rule in rules:
        diagnostics.extend(rule.apply(graph))
    return diagnostics


class ValidationError(Exception):
    """Raised when validate_or_raise finds errors."""

    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics = diagnostics
        messages = [d.message for d in diagnostics if d.severity == Severity.ERROR]
        super().__init__(f"{len(messages)} validation error(s): " + "; ".join(messages))


def validate_or_raise(graph: "Graph", extra_rules: list[LintRule] | None = None) -> list[Diagnostic]:
    """Run validation; raise ValidationError if any ERROR-level diagnostics exist.

    Returns the full list of diagnostics (including warnings/info) on success.
    """
    diags = validate(graph, extra_rules)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    if errors:
        raise ValidationError(errors)
    return diags
