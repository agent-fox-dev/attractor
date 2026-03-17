"""Human-in-the-loop interfaces per Section 6 of the Attractor spec."""

from __future__ import annotations

import queue
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class QuestionType(StrEnum):
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    FREEFORM = "freeform"
    CONFIRMATION = "confirmation"


class AnswerValue(StrEnum):
    YES = "yes"
    NO = "no"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class Option:
    key: str
    label: str


@dataclass
class Question:
    text: str
    type: QuestionType = QuestionType.YES_NO
    options: list[Option] = field(default_factory=list)
    default: str = ""
    timeout_seconds: float = 0.0
    stage: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    value: AnswerValue = AnswerValue.YES
    selected_option: str = ""
    text: str = ""


# ---------------------------------------------------------------------------
# Interviewer ABC
# ---------------------------------------------------------------------------

class Interviewer(ABC):
    """Abstract base for presenting questions to a human operator."""

    @abstractmethod
    def ask(self, question: Question) -> Answer:
        ...

    def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        """Ask multiple questions. Default implementation calls ask() in sequence."""
        return [self.ask(q) for q in questions]

    def inform(self, message: str, stage: str = "") -> None:
        """Send an informational message to the operator (no response needed)."""
        pass


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class AutoApproveInterviewer(Interviewer):
    """Always answers YES / selects the default option."""

    def ask(self, question: Question) -> Answer:
        if question.type == QuestionType.FREEFORM:
            return Answer(value=AnswerValue.YES, text=question.default or "")
        if question.type == QuestionType.MULTIPLE_CHOICE and question.options:
            default_key = question.default or question.options[0].key
            return Answer(value=AnswerValue.YES, selected_option=default_key)
        return Answer(value=AnswerValue.YES, text=question.default)


class ConsoleInterviewer(Interviewer):
    """Presents questions on stdout / reads from stdin."""

    def inform(self, message: str, stage: str = "") -> None:
        prefix = f"[{stage}] " if stage else ""
        print(f"\n--- Info ---\n{prefix}{message}")

    def ask(self, question: Question) -> Answer:
        print(f"\n--- Human Input Required ({question.stage}) ---")
        print(question.text)

        if question.type == QuestionType.YES_NO:
            default_hint = " [Y/n]" if question.default != "no" else " [y/N]"
            raw = input(f"(yes/no){default_hint}: ").strip().lower()
            if not raw:
                raw = question.default or "yes"
            if raw in ("y", "yes"):
                return Answer(value=AnswerValue.YES, text="yes")
            return Answer(value=AnswerValue.NO, text="no")

        if question.type == QuestionType.CONFIRMATION:
            default_hint = " [Y/n]" if question.default != "no" else " [y/N]"
            raw = input(f"Confirm?{default_hint}: ").strip().lower()
            if not raw:
                raw = question.default or "yes"
            if raw in ("y", "yes"):
                return Answer(value=AnswerValue.YES, text="yes")
            return Answer(value=AnswerValue.NO, text="no")

        if question.type == QuestionType.MULTIPLE_CHOICE:
            for opt in question.options:
                print(f"  [{opt.key}] {opt.label}")
            default_hint = f" (default: {question.default})" if question.default else ""
            raw = input(f"Choose{default_hint}: ").strip()
            if not raw:
                raw = question.default
            return Answer(value=AnswerValue.YES, selected_option=raw, text=raw)

        # FREEFORM
        default_hint = f" (default: {question.default})" if question.default else ""
        raw = input(f"Enter value{default_hint}: ").strip()
        if not raw:
            raw = question.default
        return Answer(value=AnswerValue.YES, text=raw)


class CallbackInterviewer(Interviewer):
    """Delegates to a user-supplied callback function."""

    def __init__(self, callback: Callable[[Question], Answer]) -> None:
        self._callback = callback

    def ask(self, question: Question) -> Answer:
        return self._callback(question)


class QueueInterviewer(Interviewer):
    """Thread-safe interviewer that reads answers from a queue.

    Questions are placed on ``question_queue``; answers are read from
    ``answer_queue``.  Useful for GUI or remote integrations.
    """

    def __init__(
        self,
        question_queue: queue.Queue[Question] | None = None,
        answer_queue: queue.Queue[Answer] | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.question_queue: queue.Queue[Question] = question_queue or queue.Queue()
        self.answer_queue: queue.Queue[Answer] = answer_queue or queue.Queue()
        self.timeout = timeout

    def ask(self, question: Question) -> Answer:
        self.question_queue.put(question)
        try:
            return self.answer_queue.get(timeout=self.timeout)
        except queue.Empty:
            return Answer(value=AnswerValue.TIMEOUT, text="")


class RecordingInterviewer(Interviewer):
    """Wraps another interviewer and records all Q&A pairs."""

    def __init__(self, delegate: Interviewer) -> None:
        self._delegate = delegate
        self.history: list[tuple[Question, Answer]] = []

    def ask(self, question: Question) -> Answer:
        answer = self._delegate.ask(question)
        self.history.append((question, answer))
        return answer
