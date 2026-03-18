"""Tests for the pipeline execution engine."""

import json
import tempfile
from pathlib import Path

import pytest

from attractor.pipeline.engine import PipelineRunner, PipelineConfig
from attractor.pipeline.graph import Outcome, StageStatus
from attractor.pipeline.interviewer import AutoApproveInterviewer, QueueInterviewer, Answer
from attractor.pipeline.transforms import PreambleTransform
from attractor.pipeline.parser import parse_dot


def _run(dot: str, **kwargs) -> Outcome:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(logs_root=tmpdir)
        runner = PipelineRunner(
            interviewer=kwargs.get("interviewer", AutoApproveInterviewer()),
            event_callback=kwargs.get("event_callback"),
            backend=kwargs.get("backend"),
        )
        return runner.run(dot, config=config)


def test_linear_pipeline():
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, prompt="Do A"]
        b [shape=box, prompt="Do B"]
        start -> a -> b -> exit
    }
    """
    outcome = _run(dot)
    assert outcome.status == StageStatus.SUCCESS


def test_conditional_branching():
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        work  [shape=box, prompt="Do work"]
        gate  [shape=diamond]
        start -> work -> gate
        gate -> exit [condition="outcome=success", weight=10]
        gate -> work [condition="outcome!=success"]
    }
    """
    outcome = _run(dot)
    assert outcome.status == StageStatus.SUCCESS


def test_human_gate_auto_approve():
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        gate  [shape=hexagon, label="Approve?"]
        after [shape=box, prompt="After approval"]
        start -> gate
        gate -> after [label="[Y] Yes"]
        gate -> exit  [label="[N] No"]
        after -> exit
    }
    """
    outcome = _run(dot, interviewer=AutoApproveInterviewer())
    assert outcome.status == StageStatus.SUCCESS


def test_goal_gate_satisfied():
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        impl  [shape=box, prompt="Implement", goal_gate=true]
        start -> impl -> exit
    }
    """
    outcome = _run(dot)
    assert outcome.status == StageStatus.SUCCESS


def test_events_emitted():
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, prompt="Work"]
        start -> a -> exit
    }
    """
    events = []
    outcome = _run(dot, event_callback=lambda e: events.append(e.kind.value))
    assert outcome.status == StageStatus.SUCCESS
    assert "pipeline_start" in events
    assert "pipeline_end" in events
    assert "node_enter" in events
    assert "node_exit" in events


def test_stage_log_files():
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        plan  [shape=box, prompt="Make a plan"]
        start -> plan -> exit
    }
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(logs_root=tmpdir)
        runner = PipelineRunner()
        runner.run(dot, config=config)
        stage_dir = Path(tmpdir) / "plan"
        assert stage_dir.is_dir()
        assert (stage_dir / "prompt.md").exists()
        assert (stage_dir / "response.md").exists()
        assert (stage_dir / "status.json").exists()


def test_variable_expansion():
    dot = """
    digraph T {
        graph [goal="Build a calculator"]
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        plan  [shape=box, prompt="Plan for: $goal"]
        start -> plan -> exit
    }
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(logs_root=tmpdir)
        runner = PipelineRunner()
        runner.run(dot, config=config)
        prompt = (Path(tmpdir) / "plan" / "prompt.md").read_text()
        assert "Build a calculator" in prompt


def test_edge_weight_tiebreak():
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, prompt="Work"]
        b [shape=box, prompt="Path B"]
        c [shape=box, prompt="Path C"]
        start -> a
        a -> b [weight=5]
        a -> c [weight=10]
        b -> exit
        c -> exit
    }
    """
    events = []
    outcome = _run(dot, event_callback=lambda e: events.append(e))
    assert outcome.status == StageStatus.SUCCESS
    edge_events = [e for e in events if e.kind.value == "edge_follow"]
    # After node 'a', should follow edge to 'c' (weight=10 > weight=5)
    a_to = [e for e in edge_events if e.data.get("from_node") == "a"]
    assert len(a_to) == 1
    assert a_to[0].data["to_node"] == "c"


def test_ten_plus_nodes():
    nodes = [f"n{i}" for i in range(12)]
    node_defs = "\n".join(f'    {n} [shape=box, prompt="Step {n}"]' for n in nodes)
    chain = " -> ".join(["start"] + nodes + ["exit"])
    dot = f"""
    digraph Big {{
        start [shape=Mdiamond]
        exit  [shape=Msquare]
{node_defs}
        {chain}
    }}
    """
    outcome = _run(dot)
    assert outcome.status == StageStatus.SUCCESS


def test_preamble_transform_injects_fidelity():
    graph = parse_dot("""
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, prompt="Do work", fidelity="compact"]
        b [shape=box, prompt="More work"]
        start -> a -> b -> exit
    }
    """)
    transform = PreambleTransform()
    graph = transform.apply(graph)
    assert "[Context fidelity: compact]" in graph.nodes["a"].prompt
    # Node without fidelity should be unchanged
    assert "[Context fidelity" not in graph.nodes["b"].prompt


def test_preamble_transform_skips_full_fidelity():
    graph = parse_dot("""
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, prompt="Do work", fidelity="full"]
        start -> a -> exit
    }
    """)
    transform = PreambleTransform()
    graph = transform.apply(graph)
    assert "[Context fidelity" not in graph.nodes["a"].prompt


def test_loop_restart_edge():
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, prompt="Work A"]
        b [shape=box, prompt="Work B"]
        start -> a -> b
        b -> exit [weight=10]
        b -> a [loop_restart=true]
    }
    """
    events = []
    outcome = _run(dot, event_callback=lambda e: events.append(e))
    assert outcome.status == StageStatus.SUCCESS
    edge_events = [e for e in events if e.kind.value == "edge_follow"]
    # The highest-weight edge from b goes to exit, so we should reach exit
    b_to = [e for e in edge_events if e.data.get("from_node") == "b"]
    assert any(e.data["to_node"] == "exit" for e in b_to)


def test_human_gate_timeout_default_choice():
    """human.default_choice selects the edge target on timeout."""
    from attractor.pipeline.handlers.human import WaitForHumanHandler
    from attractor.pipeline.context import Context

    graph = parse_dot("""
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        gate [shape=hexagon, label="Approve?"]
        yes_path [shape=box, prompt="Approved"]
        start -> gate
        gate -> yes_path [label="&Yes"]
        gate -> exit [label="&No"]
        yes_path -> exit
    }
    """)
    # Set human.default_choice on the gate node
    graph.nodes["gate"].attrs["human.default_choice"] = "y"

    # Use a QueueInterviewer that times out immediately
    interviewer = QueueInterviewer(timeout=0.01)
    handler = WaitForHumanHandler(interviewer=interviewer)
    ctx = Context()
    outcome = handler.execute(graph.nodes["gate"], ctx, graph)
    # Should succeed with default choice instead of failing on timeout
    assert outcome.status == StageStatus.SUCCESS
    assert "yes_path" in outcome.suggested_next_ids


def test_manager_poll_interval_parsing():
    """manager.poll_interval supports duration strings."""
    from attractor.pipeline.handlers.manager import _parse_duration

    assert _parse_duration("500ms") == 0.5
    assert _parse_duration("2s") == 2.0
    assert _parse_duration("1m") == 60.0
    assert _parse_duration("1h") == 3600.0
    assert _parse_duration("3") == 3.0


def test_tool_command_attribute():
    """tool_command attribute is used by ToolHandler."""
    from attractor.pipeline.handlers.tool_handler import ToolHandler
    from attractor.pipeline.context import Context

    graph = parse_dot("""
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        t [shape=box, type="tool"]
        start -> t -> exit
    }
    """)
    graph.nodes["t"].attrs["tool_command"] = "echo hello_tool"
    handler = ToolHandler()
    ctx = Context()
    outcome = handler.execute(graph.nodes["t"], ctx, graph)
    assert outcome.status == StageStatus.SUCCESS
    assert "hello_tool" in outcome.notes


def test_allow_partial_on_fail():
    """allow_partial converts FAIL to PARTIAL_SUCCESS when retries exhausted."""
    from attractor.pipeline.engine import execute_with_retry, RetryPolicy
    from attractor.pipeline.context import Context
    from attractor.pipeline.handlers.base import Handler

    class FailHandler(Handler):
        def execute(self, node, context, graph, logs_root=None, emitter=None):
            return Outcome(status=StageStatus.FAIL, failure_reason="always fails")

    graph = parse_dot("""
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, prompt="Work", allow_partial=true]
        start -> a -> exit
    }
    """)
    node = graph.nodes["a"]
    policy = RetryPolicy(max_retries=1)
    outcome = execute_with_retry(node, Context(), graph, FailHandler(), policy)
    assert outcome.status == StageStatus.PARTIAL_SUCCESS
    assert "allow_partial" in outcome.notes


def test_manager_child_status_via_context():
    """Manager loop terminates when child status is set to completed in context."""
    from attractor.pipeline.handlers.manager import ManagerLoopHandler
    from attractor.pipeline.context import Context

    graph = parse_dot("""
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        mgr [shape=box, type="stack.manager_loop"]
        start -> mgr -> exit
    }
    """)
    graph.nodes["mgr"].attrs["max_cycles"] = "5"
    graph.nodes["mgr"].attrs["wait_seconds"] = "0"

    ctx = Context()
    # Pre-set child status as if child completed
    ctx.set("context.stack.child.status", "completed")
    ctx.set("context.stack.child.outcome", "success")

    handler = ManagerLoopHandler()
    outcome = handler.execute(graph.nodes["mgr"], ctx, graph)
    assert outcome.status == StageStatus.SUCCESS
    assert "child" in outcome.notes.lower() or "completed" in outcome.notes.lower()


def test_manager_actions_attribute():
    """Manager loop respects manager.actions attribute."""
    from attractor.pipeline.handlers.manager import ManagerLoopHandler
    from attractor.pipeline.context import Context

    graph = parse_dot("""
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        mgr [shape=box, type="stack.manager_loop"]
        start -> mgr -> exit
    }
    """)
    graph.nodes["mgr"].attrs["max_cycles"] = "2"
    graph.nodes["mgr"].attrs["wait_seconds"] = "0"
    graph.nodes["mgr"].attrs["manager.actions"] = "observe"  # no "wait"

    ctx = Context()
    ctx.set("manager_done", True)

    handler = ManagerLoopHandler()
    outcome = handler.execute(graph.nodes["mgr"], ctx, graph)
    assert outcome.status == StageStatus.SUCCESS


def test_edge_selection_conditions_first():
    """Condition-based edges take highest priority per spec Section 3.3."""
    from attractor.pipeline.engine import select_edge
    from attractor.pipeline.context import Context

    graph = parse_dot("""
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box]
        ok [shape=box]
        fallback [shape=box]
        start -> a -> ok [condition="status=success"]
        a -> fallback
    }
    """)
    ctx = Context()
    ctx.set("status", "success")
    outcome = Outcome(status=StageStatus.SUCCESS, suggested_next_ids=["fallback"])
    # Condition should win over suggested_next_ids
    edge = select_edge(graph.nodes["a"], outcome, ctx, graph)
    assert edge is not None
    assert edge.to_node == "ok"


def test_label_normalization_strips_accelerators():
    """Label normalization strips accelerator prefixes."""
    from attractor.pipeline.engine import normalize_label

    assert normalize_label("[Y] Yes") == "yes"
    assert normalize_label("N) No") == "no"
    assert normalize_label("Q - Quit") == "quit"
    assert normalize_label("&Retry") == "retry"
    assert normalize_label("plain_label") == "plain_label"


def test_manifest_file_created():
    """Pipeline run creates a manifest.json in logs_root."""
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        start -> exit
    }
    """
    import json
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(logs_root=tmpdir)
        runner = PipelineRunner()
        runner.run(dot, config=config)
        manifest_path = Path(tmpdir) / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "start_time" in manifest
        assert manifest["node_count"] >= 2


def test_accelerator_paren_format():
    """Human handler parses Y) Yes accelerator format."""
    from attractor.pipeline.handlers.human import _parse_accelerator

    key, label = _parse_accelerator("Y) Yes")
    assert key == "y"
    assert label == "Yes"


def test_accelerator_dash_format():
    """Human handler parses Y - Yes accelerator format."""
    from attractor.pipeline.handlers.human import _parse_accelerator

    key, label = _parse_accelerator("Q - Quit")
    assert key == "q"
    assert label == "Quit"


def test_event_types_completeness():
    """Event types include spec-required stage and parallel events."""
    from attractor.pipeline.events import PipelineEventKind

    assert hasattr(PipelineEventKind, "STAGE_STARTED")
    assert hasattr(PipelineEventKind, "STAGE_COMPLETED")
    assert hasattr(PipelineEventKind, "STAGE_FAILED")
    assert hasattr(PipelineEventKind, "PARALLEL_STARTED")
    assert hasattr(PipelineEventKind, "PARALLEL_BRANCH_STARTED")
    assert hasattr(PipelineEventKind, "INTERVIEW_STARTED")
    assert hasattr(PipelineEventKind, "INTERVIEW_COMPLETED")


def test_checkpoint_save_and_resume():
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, prompt="Step A"]
        b [shape=box, prompt="Step B"]
        start -> a -> b -> exit
    }
    """
    from attractor.pipeline.context import Checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        cp_dir = Path(tmpdir) / "checkpoints"
        cp_dir.mkdir()
        config = PipelineConfig(
            logs_root=tmpdir,
            checkpoint_dir=cp_dir,
        )
        runner = PipelineRunner()
        outcome = runner.run(dot, config=config)
        assert outcome.status == StageStatus.SUCCESS

        # Checkpoint file should exist
        cp_path = cp_dir / "checkpoint.json"
        assert cp_path.exists()

        # Load and verify checkpoint
        cp = Checkpoint.load(cp_path)
        assert "a" in cp.completed_nodes
        assert "b" in cp.completed_nodes


def test_edge_selection_multiple_conditions_weight_sorted():
    """When multiple conditional edges match, highest weight wins."""
    from attractor.pipeline.engine import select_edge
    from attractor.pipeline.context import Context

    graph = parse_dot("""
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box]
        low [shape=box]
        high [shape=box]
        start -> a
        a -> low [condition="status=success", weight=1]
        a -> high [condition="status=success", weight=10]
    }
    """)
    ctx = Context()
    ctx.set("status", "success")
    outcome = Outcome(status=StageStatus.SUCCESS)
    edge = select_edge(graph.nodes["a"], outcome, ctx, graph)
    assert edge is not None
    assert edge.to_node == "high"


def test_parallel_handler_emits_events():
    """ParallelHandler emits PARALLEL_STARTED/BRANCH_STARTED/BRANCH_COMPLETED/COMPLETED."""
    from attractor.pipeline.handlers.parallel import ParallelHandler
    from attractor.pipeline.context import Context
    from attractor.pipeline.events import EventEmitter, PipelineEventKind

    graph = parse_dot("""
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        par [shape=component]
        a [shape=box, prompt="A"]
        b [shape=box, prompt="B"]
        start -> par
        par -> a
        par -> b
        a -> exit
        b -> exit
    }
    """)
    events = []
    emitter = EventEmitter(on_event=lambda e: events.append(e))
    handler = ParallelHandler()
    ctx = Context()
    outcome = handler.execute(graph.nodes["par"], ctx, graph, emitter=emitter)
    assert outcome.status == StageStatus.SUCCESS

    kinds = [e.kind for e in events]
    assert PipelineEventKind.PARALLEL_STARTED in kinds
    assert PipelineEventKind.PARALLEL_COMPLETED in kinds
    branch_started = [e for e in events if e.kind == PipelineEventKind.PARALLEL_BRANCH_STARTED]
    branch_completed = [e for e in events if e.kind == PipelineEventKind.PARALLEL_BRANCH_COMPLETED]
    assert len(branch_started) == 2
    assert len(branch_completed) == 2


def test_human_handler_emits_interview_events():
    """WaitForHumanHandler emits INTERVIEW_STARTED and INTERVIEW_COMPLETED."""
    from attractor.pipeline.handlers.human import WaitForHumanHandler
    from attractor.pipeline.context import Context
    from attractor.pipeline.events import EventEmitter, PipelineEventKind

    graph = parse_dot("""
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        gate [shape=hexagon, label="Approve?"]
        yes_path [shape=box, prompt="Approved"]
        start -> gate
        gate -> yes_path [label="&Yes"]
        gate -> exit [label="&No"]
        yes_path -> exit
    }
    """)
    events = []
    emitter = EventEmitter(on_event=lambda e: events.append(e))
    handler = WaitForHumanHandler(interviewer=AutoApproveInterviewer())
    ctx = Context()
    outcome = handler.execute(graph.nodes["gate"], ctx, graph, emitter=emitter)
    assert outcome.status == StageStatus.SUCCESS

    kinds = [e.kind for e in events]
    assert PipelineEventKind.INTERVIEW_STARTED in kinds
    assert PipelineEventKind.INTERVIEW_COMPLETED in kinds


def test_goal_gate_accepts_partial_success():
    """Goal gates should pass for PARTIAL_SUCCESS status."""
    from attractor.pipeline.engine import check_goal_gates

    graph = parse_dot("""
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        impl  [shape=box, prompt="Implement", goal_gate=true]
        start -> impl -> exit
    }
    """)
    node_outcomes = {
        "impl": Outcome(status=StageStatus.PARTIAL_SUCCESS),
    }
    ok, failing = check_goal_gates(graph, node_outcomes)
    assert ok is True
    assert failing is None


def test_goal_gate_terminal_check():
    """Goal gate check happens at terminal node per spec Section 3.2."""
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        impl  [shape=box, prompt="Implement", goal_gate=true]
        start -> impl -> exit
    }
    """
    # This should succeed because the simulated backend produces SUCCESS
    outcome = _run(dot)
    assert outcome.status == StageStatus.SUCCESS


def test_status_json_written_for_nodes():
    """Engine writes status.json for each node when logs_root is set."""
    import json

    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, prompt="Step A"]
        start -> a -> exit
    }
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(logs_root=tmpdir)
        runner = PipelineRunner()
        runner.run(dot, config=config)
        status_file = Path(tmpdir) / "a" / "status.json"
        assert status_file.exists()
        status = json.loads(status_file.read_text())
        assert status["status"] == "success"
        assert status["node_id"] == "a"


def test_codergen_label_fallback():
    """Codergen handler uses node.label when node.prompt is empty."""
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, label="Do the work"]
        start -> a -> exit
    }
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(logs_root=tmpdir)
        runner = PipelineRunner()
        outcome = runner.run(dot, config=config)
        assert outcome.status == StageStatus.SUCCESS
        # The prompt file should contain the label text
        prompt_file = Path(tmpdir) / "a" / "prompt.md"
        assert prompt_file.exists()
        assert "Do the work" in prompt_file.read_text()


def test_auto_status_synthesizes_success():
    """auto_status=true synthesizes SUCCESS when handler writes no status.json."""
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, auto_status="true", prompt="Do something"]
        start -> a -> exit
    }
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(logs_root=tmpdir)
        runner = PipelineRunner()
        outcome = runner.run(dot, config=config)
        assert outcome.status == StageStatus.SUCCESS
        # status.json should exist with synthesized status
        status_file = Path(tmpdir) / "a" / "status.json"
        assert status_file.exists()
        status = json.loads(status_file.read_text())
        assert status["status"] == "success"
