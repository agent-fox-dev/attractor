"""Tests for pipeline validation."""

from attractor.pipeline.parser import parse_dot
from attractor.pipeline.validation import validate, validate_or_raise, Severity, ValidationError

import pytest


def test_valid_pipeline():
    dot = """
    digraph V {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, prompt="Work"]
        start -> a -> exit
    }
    """
    g = parse_dot(dot)
    diags = validate(g)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 0


def test_missing_start_node():
    dot = """
    digraph V {
        exit [shape=Msquare]
        a [shape=box]
        a -> exit
    }
    """
    g = parse_dot(dot)
    diags = validate(g)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert any("start" in d.message.lower() for d in errors)


def test_validate_or_raise_on_error():
    dot = """
    digraph V {
        a [shape=box]
        b [shape=box]
        a -> b
    }
    """
    g = parse_dot(dot)
    with pytest.raises(ValidationError):
        validate_or_raise(g)


def test_unreachable_node_warning():
    dot = """
    digraph V {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box]
        orphan [shape=box]
        start -> a -> exit
    }
    """
    g = parse_dot(dot)
    diags = validate(g)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert any("orphan" in d.message for d in errors)


def test_condition_syntax_valid():
    dot = """
    digraph V {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box]
        start -> a [condition="outcome=success"]
        a -> exit
    }
    """
    g = parse_dot(dot)
    diags = validate(g)
    cond_errors = [d for d in diags if d.rule == "condition_syntax" and d.severity == Severity.ERROR]
    assert len(cond_errors) == 0


def test_retry_target_exists_warning():
    dot = """
    digraph V {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, retry_target="nonexistent"]
        start -> a -> exit
    }
    """
    g = parse_dot(dot)
    diags = validate(g)
    warnings = [d for d in diags if d.rule == "retry_target_exists"]
    assert len(warnings) >= 1
    assert "nonexistent" in warnings[0].message


def test_goal_gate_has_retry_warning():
    dot = """
    digraph V {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, goal_gate="true"]
        start -> a -> exit
    }
    """
    g = parse_dot(dot)
    diags = validate(g)
    warnings = [d for d in diags if d.rule == "goal_gate_has_retry"]
    assert len(warnings) >= 1
    assert "goal-gate" in warnings[0].message.lower() or "goal_gate" in warnings[0].rule


def test_fidelity_valid_warning():
    dot = """
    digraph V {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, fidelity="invalid_mode"]
        start -> a -> exit
    }
    """
    g = parse_dot(dot)
    diags = validate(g)
    fidelity_warnings = [d for d in diags if d.rule == "fidelity_valid"]
    assert len(fidelity_warnings) >= 1


def test_fidelity_valid_no_warning():
    dot = """
    digraph V {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, fidelity="full"]
        start -> a -> exit
    }
    """
    g = parse_dot(dot)
    diags = validate(g)
    fidelity_warnings = [d for d in diags if d.rule == "fidelity_valid"]
    assert len(fidelity_warnings) == 0
