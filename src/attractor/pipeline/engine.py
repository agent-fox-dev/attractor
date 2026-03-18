"""Core pipeline execution engine per Section 3 of the Attractor spec.

Provides the main ``run`` function, edge selection logic, retry policies,
checkpoint management, and the ``PipelineRunner`` convenience class.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .conditions import evaluate_condition
from .context import Checkpoint, Context
from .events import EventEmitter, PipelineEvent, PipelineEventKind
from .graph import Edge, Graph, Node, Outcome, StageStatus
from .handlers import (
    Handler,
    HandlerRegistry,
    _build_default_registry,
    _get_default_registry,
    infer_type,
)
from .handlers.codergen import CodergenBackend, CodergenHandler
from .handlers.human import WaitForHumanHandler
from .interviewer import AutoApproveInterviewer, Interviewer
from .transforms import DEFAULT_TRANSFORMS, Transform, prepare_pipeline
from .validation import Diagnostic, Severity, validate_or_raise


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BackoffConfig:
    """Backoff configuration for retries."""
    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: bool = False


@dataclass
class RetryPolicy:
    """Retry policy for a node."""
    max_retries: int = 0
    backoff: BackoffConfig = field(default_factory=BackoffConfig)
    retry_target: str = ""
    fallback_retry_target: str = ""


@dataclass
class PipelineConfig:
    """Top-level configuration for a pipeline run."""
    logs_root: Path | None = None
    checkpoint_dir: Path | None = None
    resume_from: Checkpoint | None = None
    max_steps: int = 10000
    context_values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.logs_root is not None and not isinstance(self.logs_root, Path):
            self.logs_root = Path(self.logs_root)
        if self.checkpoint_dir is not None and not isinstance(self.checkpoint_dir, Path):
            self.checkpoint_dir = Path(self.checkpoint_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_START_SHAPES = frozenset({"Mdiamond", "mdiamond", "circle", "point"})


def find_start_node(graph: Graph) -> Node:
    """Find the unique start node in *graph*. Raises ValueError if missing."""
    # Priority 1: explicit type="start"
    for node in graph.nodes.values():
        if node.type == "start":
            return node
    # Priority 2: shape-based (Mdiamond per spec)
    for node in graph.nodes.values():
        if node.shape in _START_SHAPES:
            return node
    # Priority 3: id-based fallback
    for name in ("start", "Start"):
        if name in graph.nodes:
            return graph.nodes[name]
    raise ValueError("No start node found in the graph.")


def is_terminal(node: Node) -> bool:
    """Return True if *node* is a terminal (exit) node."""
    t = infer_type(node)
    return t == "exit"


_ACCEL_BRACKET = re.compile(r"^\[\w\]\s*")       # [Y] Label
_ACCEL_PAREN = re.compile(r"^\w\)\s*")           # Y) Label
_ACCEL_DASH = re.compile(r"^\w\s*-\s+")          # Y - Label


def normalize_label(label: str) -> str:
    """Normalize an edge label for comparison.

    Strips accelerator prefixes (``[Y] ``, ``Y) ``, ``Y - ``, ``&``),
    lowercases, removes non-alphanumeric characters except underscores.
    """
    label = label.strip()
    # Strip accelerator prefixes
    label = _ACCEL_BRACKET.sub("", label)
    label = _ACCEL_PAREN.sub("", label)
    label = _ACCEL_DASH.sub("", label)
    # Strip & markers (keep the letter after &)
    label = label.replace("&", "")
    label = label.strip().lower()
    label = re.sub(r"[^a-z0-9_]", "", label)
    return label


def select_edge(
    node: Node,
    outcome: Outcome,
    context: Context,
    graph: Graph,
) -> Edge | None:
    """Select the next edge to follow using the 5-step priority system.

    Priority order per spec Section 3.3:
      1. Edges whose condition evaluates to True (highest priority)
      2. Edges whose label matches outcome.preferred_label
      3. Suggested next IDs from outcome
      4. Highest weight / status label match
      5. Lexical tiebreak (default / unlabelled / first available)
    """
    outgoing = graph.outgoing_edges(node.id)
    if not outgoing:
        return None

    # Step 1: condition-based edges (highest priority)
    # When multiple conditions match, sort by weight (desc) then lexical order.
    conditional_edges = [e for e in outgoing if e.condition]
    matched = [e for e in conditional_edges if evaluate_condition(e.condition, outcome, context)]
    if matched:
        matched.sort(key=lambda e: (-e.weight, e.to_node))
        return matched[0]

    # Step 2: label matches preferred_label
    if outcome.preferred_label:
        norm_pref = normalize_label(outcome.preferred_label)
        for edge in outgoing:
            if edge.label and normalize_label(edge.label) == norm_pref:
                return edge

    # Step 3: suggested_next_ids from outcome
    if outcome.suggested_next_ids:
        for sid in outcome.suggested_next_ids:
            for edge in outgoing:
                if edge.to_node == sid:
                    return edge

    # Step 4: label matches outcome status, then highest weight
    status_label = normalize_label(outcome.status.value)
    for edge in outgoing:
        if edge.label and normalize_label(edge.label) == status_label:
            return edge

    # Step 5: default edge (unlabelled, or first available), highest weight wins
    unlabelled = [e for e in outgoing if not e.label and not e.condition]
    if unlabelled:
        unlabelled.sort(key=lambda e: e.weight, reverse=True)
        return unlabelled[0]

    # Fallback: first edge (highest weight)
    return sorted(outgoing, key=lambda e: e.weight, reverse=True)[0]


def check_goal_gates(
    graph: Graph,
    node_outcomes: dict[str, Outcome],
) -> tuple[bool, Node | None]:
    """Check all goal_gate nodes have succeeded.

    Returns (all_passed, first_failing_node).
    """
    for node in graph.nodes.values():
        if not node.goal_gate:
            continue
        outcome = node_outcomes.get(node.id)
        if outcome is None or outcome.status not in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS):
            return False, node
    return True, None


def build_retry_policy(node: Node, graph: Graph) -> RetryPolicy:
    """Build a RetryPolicy for *node*, inheriting graph defaults.

    ``max_retries=0`` on a node means no retries (the default).
    The graph-level ``default_max_retry`` is used as the ceiling for
    nodes that *do* set retries, not as a blanket default.
    """
    max_retries = node.max_retries  # 0 means no retries
    retry_target = node.retry_target or graph.retry_target
    fallback = node.fallback_retry_target or graph.fallback_retry_target

    return RetryPolicy(
        max_retries=max_retries,
        backoff=BackoffConfig(),
        retry_target=retry_target,
        fallback_retry_target=fallback,
    )


def execute_with_retry(
    node: Node,
    context: Context,
    graph: Graph,
    handler: Handler,
    retry_policy: RetryPolicy,
    emitter: EventEmitter | None = None,
    logs_root: Path | None = None,
) -> Outcome:
    """Execute a handler with retry logic.

    Retries on RETRY or FAIL status up to retry_policy.max_retries.
    """
    delay = retry_policy.backoff.initial_delay
    last_outcome: Outcome | None = None

    for attempt in range(retry_policy.max_retries + 1):
        try:
            outcome = handler.execute(node, context, graph, logs_root, emitter=emitter)
        except Exception as exc:
            outcome = Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Handler exception: {exc}",
            )

        last_outcome = outcome

        last_outcome = outcome

        if outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS, StageStatus.SKIPPED):
            return outcome

        # Should we retry?
        if outcome.status == StageStatus.RETRY or (
            outcome.status == StageStatus.FAIL and attempt < retry_policy.max_retries
        ):
            if emitter:
                emitter.emit(
                    PipelineEventKind.NODE_RETRY,
                    node_id=node.id,
                    attempt=attempt + 1,
                    max_retries=retry_policy.max_retries,
                    reason=outcome.failure_reason,
                )
            context.append_log(
                f"[retry] Node '{node.id}' attempt {attempt + 1}/{retry_policy.max_retries}: "
                f"{outcome.failure_reason}"
            )
            time.sleep(delay)
            delay = min(delay * retry_policy.backoff.multiplier, retry_policy.backoff.max_delay)
            continue

        # No more retries
        break

    final = last_outcome or Outcome(status=StageStatus.FAIL, failure_reason="Unknown error.")

    # allow_partial: accept PARTIAL_SUCCESS when retries exhausted instead of FAIL
    if final.status == StageStatus.FAIL and node.allow_partial:
        final = Outcome(
            status=StageStatus.PARTIAL_SUCCESS,
            failure_reason=final.failure_reason,
            notes="allow_partial: converted FAIL to PARTIAL_SUCCESS after retries exhausted",
            context_updates=final.context_updates,
        )

    return final


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    graph: Graph,
    config: PipelineConfig | None = None,
    registry: HandlerRegistry | None = None,
    emitter: EventEmitter | None = None,
) -> Outcome:
    """Execute a pipeline graph and return the final outcome.

    This is the low-level execution function.  Most callers should use
    ``PipelineRunner`` instead.
    """
    config = config or PipelineConfig()
    registry = registry or _get_default_registry()
    emitter = emitter or EventEmitter()

    context = Context()
    if config.context_values:
        context.apply_updates(config.context_values)

    node_outcomes: dict[str, Outcome] = {}
    node_retries: dict[str, int] = {}
    completed_nodes: list[str] = []

    # Resume from checkpoint if provided
    if config.resume_from:
        cp = config.resume_from
        context.apply_updates(cp.context_values)
        for log in cp.logs:
            context.append_log(log)
        completed_nodes = list(cp.completed_nodes)
        node_retries = dict(cp.node_retries)

    # Find start
    start_node = find_start_node(graph)
    current_node: Node | None = start_node

    # Track fidelity degradation on resume
    _degrade_fidelity_on_resume = False
    if config.resume_from and config.resume_from.current_node:
        resume_id = config.resume_from.current_node
        if resume_id in graph.nodes:
            current_node = graph.nodes[resume_id]
            # Degrade fidelity for first resumed node if previous used "full"
            _degrade_fidelity_on_resume = True

    emitter.emit(PipelineEventKind.PIPELINE_START, graph_name=graph.name)

    # Write manifest file
    if config.logs_root:
        import json as _json
        import datetime as _dt

        logs_path = Path(config.logs_root) if isinstance(config.logs_root, str) else config.logs_root
        logs_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "name": graph.name,
            "goal": graph.goal or "",
            "start_time": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
        }
        (logs_path / "manifest.json").write_text(
            _json.dumps(manifest, indent=2), encoding="utf-8"
        )

    step = 0
    final_outcome = Outcome(status=StageStatus.SUCCESS)

    while current_node is not None and step < config.max_steps:
        step += 1
        node = current_node

        emitter.emit(PipelineEventKind.NODE_ENTER, node_id=node.id, step=step)
        context.append_log(f"[step {step}] Entering node '{node.id}'")

        # Apply fidelity degradation on first resumed node
        if _degrade_fidelity_on_resume:
            orig_fidelity = node.attrs.get("fidelity", "")
            if orig_fidelity == "full":
                node.attrs["fidelity"] = "summary:high"
                context.append_log(
                    f"[resume] Degraded fidelity full -> summary:high for '{node.id}'"
                )
            _degrade_fidelity_on_resume = False

        # Resolve handler
        try:
            handler = registry.resolve(node)
        except KeyError as exc:
            context.append_log(f"[error] {exc}")
            emitter.emit(PipelineEventKind.ERROR, node_id=node.id, error=str(exc))
            final_outcome = Outcome(
                status=StageStatus.FAIL,
                failure_reason=str(exc),
            )
            break

        # Build retry policy and execute
        policy = build_retry_policy(node, graph)
        outcome = execute_with_retry(
            node, context, graph, handler, policy,
            emitter=emitter, logs_root=config.logs_root,
        )

        # auto_status: if handler wrote no status.json and auto_status is set,
        # synthesize a SUCCESS outcome (spec Section 2.6, Appendix C)
        if node.auto_status and outcome.status != StageStatus.SUCCESS:
            handler_wrote_status = False
            if config.logs_root:
                handler_wrote_status = (Path(config.logs_root) / node.id / "status.json").exists()
            if not handler_wrote_status:
                outcome = Outcome(
                    status=StageStatus.SUCCESS,
                    notes="auto-status: handler completed without writing status",
                )

        # Write status.json for this node if logs_root is set and handler didn't
        if config.logs_root:
            import json as _json
            stage_dir = Path(config.logs_root) / node.id
            status_file = stage_dir / "status.json"
            if not status_file.exists():
                stage_dir.mkdir(parents=True, exist_ok=True)
                status_data = {
                    "node_id": node.id,
                    "status": outcome.status.value,
                    "notes": outcome.notes or (
                        "auto-status: handler completed without writing status"
                        if getattr(node, "auto_status", False) else ""
                    ),
                    "failure_reason": outcome.failure_reason,
                }
                status_file.write_text(_json.dumps(status_data, indent=2))

        # Apply context updates from outcome
        if outcome.context_updates:
            context.apply_updates(outcome.context_updates)

        node_outcomes[node.id] = outcome
        completed_nodes.append(node.id)
        node_retries[node.id] = node_retries.get(node.id, 0)

        emitter.emit(
            PipelineEventKind.NODE_EXIT,
            node_id=node.id,
            status=outcome.status.value,
            step=step,
        )

        # Save checkpoint
        if config.checkpoint_dir:
            cp = Checkpoint.from_context(
                context, node.id, completed_nodes, node_retries
            )
            cp_path = config.checkpoint_dir / "checkpoint.json"
            cp.save(cp_path)
            emitter.emit(PipelineEventKind.CHECKPOINT_SAVE, path=str(cp_path))

        # Terminal node — check goal gates before exiting (spec Section 3.2)
        if is_terminal(node):
            gates_ok, failing_gate = check_goal_gates(graph, node_outcomes)
            if not gates_ok and failing_gate:
                emitter.emit(
                    PipelineEventKind.GOAL_GATE_FAIL,
                    node_id=failing_gate.id,
                )
                # Jump to retry_target if one exists instead of exiting
                gate_policy = build_retry_policy(failing_gate, graph)
                if gate_policy.retry_target and gate_policy.retry_target in graph.nodes:
                    current_node = graph.nodes[gate_policy.retry_target]
                    emitter.emit(
                        PipelineEventKind.EDGE_FOLLOW,
                        from_node=node.id,
                        to_node=gate_policy.retry_target,
                        reason="goal_gate_retry",
                    )
                    continue
                elif gate_policy.fallback_retry_target and gate_policy.fallback_retry_target in graph.nodes:
                    current_node = graph.nodes[gate_policy.fallback_retry_target]
                    emitter.emit(
                        PipelineEventKind.EDGE_FOLLOW,
                        from_node=node.id,
                        to_node=gate_policy.fallback_retry_target,
                        reason="goal_gate_fallback",
                    )
                    continue
            final_outcome = outcome
            break

        # Check goal gates (non-terminal nodes)
        gates_ok, failing_gate = check_goal_gates(graph, node_outcomes)
        if not gates_ok and failing_gate:
            emitter.emit(
                PipelineEventKind.GOAL_GATE_FAIL,
                node_id=failing_gate.id,
            )

        # Handle failure with retry/fallback targets
        if outcome.status == StageStatus.FAIL:
            if policy.retry_target and policy.retry_target in graph.nodes:
                current_node = graph.nodes[policy.retry_target]
                emitter.emit(
                    PipelineEventKind.EDGE_FOLLOW,
                    from_node=node.id,
                    to_node=policy.retry_target,
                    reason="retry_target_on_fail",
                )
                continue
            elif policy.fallback_retry_target and policy.fallback_retry_target in graph.nodes:
                current_node = graph.nodes[policy.fallback_retry_target]
                emitter.emit(
                    PipelineEventKind.EDGE_FOLLOW,
                    from_node=node.id,
                    to_node=policy.fallback_retry_target,
                    reason="fallback_retry_target_on_fail",
                )
                continue

        # Select next edge
        edge = select_edge(node, outcome, context, graph)
        if edge is None:
            # No outgoing edge -- treat as terminal
            final_outcome = outcome
            break

        emitter.emit(
            PipelineEventKind.EDGE_FOLLOW,
            from_node=node.id,
            to_node=edge.to_node,
            label=edge.label,
        )
        context.append_log(
            f"[step {step}] Following edge '{node.id}' -> '{edge.to_node}'"
            + (f" [{edge.label}]" if edge.label else "")
        )

        next_node = graph.nodes.get(edge.to_node)
        if next_node is None:
            final_outcome = Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Edge target '{edge.to_node}' not found in graph.",
            )
            break

        # Handle loop_restart edges: restart execution from the target node
        if edge.loop_restart:
            emitter.emit(
                PipelineEventKind.EDGE_FOLLOW,
                from_node=node.id,
                to_node=edge.to_node,
                reason="loop_restart",
            )
            context.append_log(
                f"[step {step}] Loop restart -> '{edge.to_node}'"
            )
            # Reset tracking for the restart
            node_outcomes.clear()
            node_retries.clear()
            completed_nodes.clear()

        current_node = next_node

    if step >= config.max_steps:
        final_outcome = Outcome(
            status=StageStatus.FAIL,
            failure_reason=f"Pipeline exceeded maximum steps ({config.max_steps}).",
        )

    emitter.emit(
        PipelineEventKind.PIPELINE_END,
        status=final_outcome.status.value,
        steps=step,
    )

    return final_outcome


# ---------------------------------------------------------------------------
# PipelineRunner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """High-level runner that ties parsing, transforms, validation, and execution together."""

    def __init__(
        self,
        backend: CodergenBackend | None = None,
        interviewer: Interviewer | None = None,
        event_callback: Callable[[PipelineEvent], None] | None = None,
        extra_transforms: list[Transform] | None = None,
        extra_handlers: dict[str, Handler] | None = None,
    ) -> None:
        self._backend = backend
        self._interviewer = interviewer or AutoApproveInterviewer()
        self._emitter = EventEmitter(on_event=event_callback)
        self._extra_transforms = extra_transforms or []
        self._extra_handlers = extra_handlers or {}
        self._registry = self._build_registry()

    def _build_registry(self) -> HandlerRegistry:
        """Build a handler registry with all built-in + extra handlers."""
        registry = _build_default_registry()

        # Replace codergen/llm handlers if a backend is provided
        if self._backend is not None:
            codergen = CodergenHandler(backend=self._backend)
            registry.register("llm", codergen)
            registry.register("codergen", codergen)
            registry.register("coder", codergen)

        # Replace human handler with the configured interviewer
        human = WaitForHumanHandler(interviewer=self._interviewer)
        registry.register("human", human)
        registry.register("wait_for_human", human)

        # Register extra handlers
        for type_str, handler in self._extra_handlers.items():
            registry.register(type_str, handler)

        return registry

    def run(
        self,
        dot_source: str,
        config: PipelineConfig | None = None,
    ) -> Outcome:
        """Parse DOT source, transform, validate, and execute.

        Raises ValidationError if the graph has ERROR-level diagnostics.
        """
        transforms = list(DEFAULT_TRANSFORMS) + self._extra_transforms
        graph, diagnostics = prepare_pipeline(dot_source, transforms=transforms)

        # Raise on errors
        errors = [d for d in diagnostics if d.severity == Severity.ERROR]
        if errors:
            from .validation import ValidationError
            raise ValidationError(errors)

        self._emitter.emit(
            PipelineEventKind.VALIDATION_COMPLETE,
            diagnostics_count=len(diagnostics),
            error_count=len(errors),
        )

        return self.run_graph(graph, config)

    def run_graph(
        self,
        graph: Graph,
        config: PipelineConfig | None = None,
    ) -> Outcome:
        """Execute an already-parsed and transformed graph."""
        return run(
            graph,
            config=config,
            registry=self._registry,
            emitter=self._emitter,
        )
