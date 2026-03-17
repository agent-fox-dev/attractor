"""Tests for the condition expression evaluator."""

from attractor.pipeline.conditions import evaluate_condition
from attractor.pipeline.context import Context
from attractor.pipeline.graph import Outcome, StageStatus


def _ctx(**kw):
    c = Context()
    for k, v in kw.items():
        c.set(k, v)
    return c


def test_outcome_equals():
    o = Outcome(status=StageStatus.SUCCESS)
    assert evaluate_condition("outcome=success", o, _ctx()) is True
    assert evaluate_condition("outcome=fail", o, _ctx()) is False


def test_outcome_not_equals():
    o = Outcome(status=StageStatus.FAIL)
    assert evaluate_condition("outcome!=success", o, _ctx()) is True
    assert evaluate_condition("outcome!=fail", o, _ctx()) is False


def test_context_key():
    o = Outcome(status=StageStatus.SUCCESS)
    ctx = _ctx(tests_passed="true")
    assert evaluate_condition("context.tests_passed=true", o, ctx) is True
    assert evaluate_condition("context.tests_passed=false", o, ctx) is False


def test_and_conjunction():
    o = Outcome(status=StageStatus.SUCCESS)
    ctx = _ctx(tests_passed="true")
    assert evaluate_condition("outcome=success && context.tests_passed=true", o, ctx) is True
    assert evaluate_condition("outcome=fail && context.tests_passed=true", o, ctx) is False


def test_empty_condition():
    o = Outcome(status=StageStatus.SUCCESS)
    assert evaluate_condition("", o, _ctx()) is True


def test_missing_context_key():
    o = Outcome(status=StageStatus.SUCCESS)
    assert evaluate_condition("context.missing=value", o, _ctx()) is False


def test_preferred_label():
    o = Outcome(status=StageStatus.SUCCESS, preferred_label="approve")
    assert evaluate_condition("preferred_label=approve", o, _ctx()) is True
    assert evaluate_condition("preferred_label=reject", o, _ctx()) is False
