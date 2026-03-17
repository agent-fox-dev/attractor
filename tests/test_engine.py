"""Tests for the pipeline execution engine."""

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
