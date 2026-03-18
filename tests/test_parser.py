"""Tests for the DOT parser."""

import pytest
from attractor.pipeline.parser import parse_dot, ParseError


def test_simple_linear_pipeline():
    dot = """
    digraph Simple {
        graph [goal="Run tests"]
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        run   [shape=box, prompt="Run tests"]
        start -> run -> exit
    }
    """
    g = parse_dot(dot)
    assert g.name == "Simple"
    assert g.goal == "Run tests"
    assert set(g.nodes.keys()) == {"start", "exit", "run"}
    assert len(g.edges) == 2
    assert g.nodes["start"].shape == "Mdiamond"
    assert g.nodes["exit"].shape == "Msquare"
    assert g.nodes["run"].prompt == "Run tests"


def test_chained_edges():
    dot = """
    digraph Chain {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box]
        b [shape=box]
        start -> a -> b -> exit [label="next"]
    }
    """
    g = parse_dot(dot)
    assert len(g.edges) == 3
    for e in g.edges:
        assert e.label == "next"


def test_edge_attributes():
    dot = """
    digraph E {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box]
        start -> a [condition="outcome=success", weight=10]
        a -> exit
    }
    """
    g = parse_dot(dot)
    edge = [e for e in g.edges if e.from_node == "start"][0]
    assert edge.condition == "outcome=success"
    assert edge.weight == 10


def test_node_defaults():
    dot = """
    digraph D {
        node [shape=box, timeout="900s"]
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [label="A"]
        b [label="B", timeout="1800s"]
        start -> a -> b -> exit
    }
    """
    g = parse_dot(dot)
    assert g.nodes["a"].shape == "box"
    assert g.nodes["a"].timeout == "900s"
    assert g.nodes["b"].timeout == "1800s"


def test_subgraph():
    dot = """
    digraph S {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        subgraph cluster_loop {
            node [shape=box]
            plan [label="Plan"]
            impl [label="Implement"]
        }
        start -> plan -> impl -> exit
    }
    """
    g = parse_dot(dot)
    assert "plan" in g.nodes
    assert "impl" in g.nodes
    assert g.nodes["plan"].shape == "box"


def test_comments_stripped():
    dot = """
    // This is a comment
    digraph C {
        /* block comment */
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        // another comment
        start -> exit
    }
    """
    g = parse_dot(dot)
    assert len(g.nodes) == 2


def test_graph_level_attributes():
    dot = """
    digraph G {
        graph [goal="Build it", label="My Pipeline", default_max_retry=5]
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        start -> exit
    }
    """
    g = parse_dot(dot)
    assert g.goal == "Build it"
    assert g.label == "My Pipeline"
    assert g.default_max_retry == 5


def test_reject_undirected():
    with pytest.raises(ParseError):
        parse_dot("graph G { a -- b }")


def test_boolean_attributes():
    dot = """
    digraph B {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box, goal_gate=true, allow_partial=false]
        start -> a -> exit
    }
    """
    g = parse_dot(dot)
    assert g.nodes["a"].goal_gate is True
    assert g.nodes["a"].allow_partial is False


def test_multiline_attributes():
    dot = """
    digraph M {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        plan [
            shape=box,
            label="Plan Step",
            prompt="Create a detailed plan"
        ]
        start -> plan -> exit
    }
    """
    g = parse_dot(dot)
    assert g.nodes["plan"].label == "Plan Step"
    assert g.nodes["plan"].prompt == "Create a detailed plan"


def test_subgraph_class_derivation():
    dot = """
    digraph S {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        subgraph cluster_planning {
            node [shape=box]
            plan [prompt="Plan it"]
            design [prompt="Design it"]
        }
        subgraph cluster_impl {
            code [shape=box, prompt="Code it"]
        }
        explicit [shape=box, prompt="Has class", class="custom"]
        start -> plan -> design -> code -> explicit -> exit
    }
    """
    g = parse_dot(dot)
    assert g.nodes["plan"].css_class == "planning"
    assert g.nodes["design"].css_class == "planning"
    assert g.nodes["code"].css_class == "impl"
    # Node with explicit class should keep it
    assert g.nodes["explicit"].css_class == "custom"


def test_subgraph_class_from_label():
    """Spec Section 2.10: class derivation prefers label over name."""
    dot = """
    digraph T {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        subgraph cluster_x {
            label="Loop A"
            node [shape=box]
            step1 [prompt="S1"]
            step2 [prompt="S2"]
        }
        start -> step1 -> step2 -> exit
    }
    """
    g = parse_dot(dot)
    # label="Loop A" -> class "loop-a" (not "x" from the name)
    assert g.nodes["step1"].css_class == "loop-a"
    assert g.nodes["step2"].css_class == "loop-a"


def test_subgraph_class_with_stylesheet():
    from attractor.pipeline.stylesheet import apply_stylesheet, parse_stylesheet
    from attractor.pipeline.graph import Graph, Node, Edge

    # Build graph manually to avoid DOT parser issues with braces in attr values.
    graph = Graph(
        name="S",
        model_stylesheet='.planning { reasoning_effort: "medium"; }',
        nodes={
            "start": Node(id="start", shape="Mdiamond"),
            "exit": Node(id="exit", shape="Msquare"),
            "plan": Node(id="plan", shape="box", prompt="Plan", css_class="planning"),
            "other": Node(id="other", shape="box", prompt="Other"),
        },
        edges=[
            Edge(from_node="start", to_node="plan"),
            Edge(from_node="plan", to_node="other"),
            Edge(from_node="other", to_node="exit"),
        ],
    )
    graph = apply_stylesheet(graph)
    assert graph.nodes["plan"].reasoning_effort == "medium"
    # 'other' has no class, so it only gets universal rules (none here)
    assert graph.nodes["other"].reasoning_effort == "high"  # default value
