"""WaitForHumanHandler per Section 4.6 of the Attractor spec.

Derives choices from outgoing edges, parses accelerator keys from labels,
presents a question via the Interviewer, and returns an outcome with
suggested_next_ids.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from ..graph import Outcome, StageStatus
from ..interviewer import (
    Answer,
    AnswerValue,
    AutoApproveInterviewer,
    Interviewer,
    Option,
    Question,
    QuestionType,
)
from .base import Handler

if TYPE_CHECKING:
    from ..context import Context
    from ..events import EventEmitter
    from ..graph import Graph, Node


_ACCEL_PATTERN = re.compile(r"&(\w)")
_BRACKET_PATTERN = re.compile(r"\[(\w)\]\s*(.*)")
_PAREN_PATTERN = re.compile(r"(\w)\)\s*(.*)")
_DASH_PATTERN = re.compile(r"(\w)\s*-\s+(.*)")


def _parse_accelerator(label: str) -> tuple[str, str]:
    """Extract an accelerator key from a label.

    Supports four formats:
      - ``[Y] Yes`` -> key='y', label='Yes'
      - ``&Yes``    -> key='y', label='Yes'
      - ``Y) Yes``  -> key='y', label='Yes'
      - ``Y - Yes`` -> key='y', label='Yes'

    Returns (key, clean_label).
    """
    stripped = label.strip()

    # Try bracket format: [Y] Yes
    bm = _BRACKET_PATTERN.match(stripped)
    if bm:
        key = bm.group(1).lower()
        clean = bm.group(2).strip() or label
        return key, clean

    # Try ampersand format: &Yes
    m = _ACCEL_PATTERN.search(stripped)
    if m:
        key = m.group(1).lower()
        clean = label.replace(f"&{m.group(1)}", m.group(1), 1)
        return key, clean

    # Try paren format: Y) Yes
    pm = _PAREN_PATTERN.match(stripped)
    if pm:
        key = pm.group(1).lower()
        clean = pm.group(2).strip() or label
        return key, clean

    # Try dash format: Y - Yes
    dm = _DASH_PATTERN.match(stripped)
    if dm:
        key = dm.group(1).lower()
        clean = dm.group(2).strip() or label
        return key, clean

    return label.lower().replace(" ", "_"), label


class WaitForHumanHandler(Handler):
    """Presents a question derived from outgoing edges and waits for input."""

    def __init__(self, interviewer: Interviewer | None = None) -> None:
        self._interviewer = interviewer or AutoApproveInterviewer()

    def execute(
        self,
        node: "Node",
        context: "Context",
        graph: "Graph",
        logs_root: Path | None = None,
        emitter: "EventEmitter | None" = None,
    ) -> Outcome:
        outgoing = graph.outgoing_edges(node.id)

        # Build options from outgoing edge labels
        options: list[Option] = []
        key_to_target: dict[str, str] = {}
        for edge in outgoing:
            label = edge.label or edge.to_node
            key, clean_label = _parse_accelerator(label)
            options.append(Option(key=key, label=clean_label))
            key_to_target[key] = edge.to_node

        # Determine question type
        if len(options) == 2 and {o.key for o in options} <= {"y", "n", "yes", "no"}:
            q_type = QuestionType.YES_NO
        elif options:
            q_type = QuestionType.MULTIPLE_CHOICE
        else:
            q_type = QuestionType.CONFIRMATION

        prompt_text = node.prompt or node.label or f"Awaiting input at '{node.id}'"

        question = Question(
            text=prompt_text,
            type=q_type,
            options=options,
            default=options[0].key if options else "",
            stage=node.id,
        )

        if emitter is not None:
            from ..events import PipelineEventKind
            emitter.emit(PipelineEventKind.INTERVIEW_STARTED, node_id=node.id, question_type=q_type.value)

        answer: Answer = self._interviewer.ask(question)

        # Map answer to outcome
        if answer.value == AnswerValue.TIMEOUT:
            if emitter is not None:
                from ..events import PipelineEventKind
                emitter.emit(PipelineEventKind.INTERVIEW_TIMEOUT, node_id=node.id)
            # Check for default_choice on timeout
            default_choice = node.attrs.get("human.default_choice", "")
            if default_choice and default_choice in key_to_target:
                return Outcome(
                    status=StageStatus.SUCCESS,
                    preferred_label=default_choice,
                    suggested_next_ids=[key_to_target[default_choice]],
                    context_updates={"human_input": f"timeout:default:{default_choice}"},
                )
            return Outcome(
                status=StageStatus.RETRY,
                failure_reason="Human gate timeout, no default.",
            )

        if answer.value == AnswerValue.NO:
            if emitter is not None:
                from ..events import PipelineEventKind
                emitter.emit(PipelineEventKind.INTERVIEW_COMPLETED, node_id=node.id, answer="no")
            # Try to find a "no" edge
            no_target = key_to_target.get("n") or key_to_target.get("no")
            if no_target:
                return Outcome(
                    status=StageStatus.SUCCESS,
                    preferred_label="no",
                    suggested_next_ids=[no_target],
                )
            return Outcome(status=StageStatus.FAIL, failure_reason="Declined by human.")

        # YES or SKIPPED with a selected option
        selected = answer.selected_option or answer.text
        suggested: list[str] = []
        preferred_label = ""
        if selected and selected in key_to_target:
            suggested = [key_to_target[selected]]
            # Find the matching option label for preferred_label
            for opt in options:
                if opt.key == selected:
                    preferred_label = opt.label
                    break
        elif options:
            # Default to first option
            suggested = [key_to_target[options[0].key]]
            preferred_label = options[0].label

        if emitter is not None:
            from ..events import PipelineEventKind
            emitter.emit(PipelineEventKind.INTERVIEW_COMPLETED, node_id=node.id, answer=selected or "yes")

        return Outcome(
            status=StageStatus.SUCCESS,
            preferred_label=preferred_label,
            suggested_next_ids=suggested,
            context_updates={"human_input": answer.text or selected},
        )
