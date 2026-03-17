"""Condition expression evaluator per Section 10 of the Attractor spec.

Supports expressions like:
  outcome=success
  outcome!=fail
  context.foo=bar
  outcome=success&&context.mode=fast
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context
    from .graph import Outcome


def evaluate_condition(condition: str, outcome: "Outcome", context: "Context") -> bool:
    """Evaluate a full condition string (may contain && conjunctions).

    Returns True if the condition string is empty or all clauses pass.
    """
    condition = condition.strip()
    if not condition:
        return True

    clauses = condition.split("&&")
    return all(evaluate_clause(clause.strip(), outcome, context) for clause in clauses)


def evaluate_clause(clause: str, outcome: "Outcome", context: "Context") -> bool:
    """Evaluate a single clause like 'outcome=success' or 'context.key!=val'."""
    clause = clause.strip()
    if not clause:
        return True

    # Check for != first (before =)
    if "!=" in clause:
        key, value = clause.split("!=", 1)
        resolved = resolve_key(key.strip(), outcome, context)
        return resolved != value.strip()
    elif "=" in clause:
        key, value = clause.split("=", 1)
        resolved = resolve_key(key.strip(), outcome, context)
        return resolved == value.strip()

    # Bare key: truthy check
    resolved = resolve_key(clause, outcome, context)
    return bool(resolved)


def resolve_key(key: str, outcome: "Outcome", context: "Context") -> str:
    """Resolve a key reference to its string value.

    Supports:
      - ``outcome`` -> outcome.status value
      - ``outcome.status`` -> outcome.status value
      - ``outcome.preferred_label`` -> outcome.preferred_label
      - ``context.<key>`` -> context.get_string(key)
    """
    key = key.strip()

    if key == "outcome" or key == "outcome.status":
        return str(outcome.status.value)
    if key == "outcome.preferred_label":
        return outcome.preferred_label
    if key == "outcome.notes":
        return outcome.notes
    if key == "outcome.failure_reason":
        return outcome.failure_reason
    if key == "preferred_label":
        return outcome.preferred_label
    if key.startswith("context."):
        ctx_key = key[len("context."):]
        return context.get_string(ctx_key, "")

    # Fallback: treat as context key
    return context.get_string(key, "")
