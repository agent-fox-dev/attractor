"""Tests for the coding agent layer."""

import tempfile
from pathlib import Path

from attractor.agent.execution.local import LocalExecutionEnvironment
from attractor.agent.tools.registry import ToolRegistry
from attractor.agent.tools.core import register_core_tools
from attractor.agent.truncation import truncate_output, truncate_tool_output
from attractor.agent.profiles.anthropic import AnthropicProfile
from attractor.agent.profiles.openai import OpenAIProfile
from attractor.agent.profiles.gemini import GeminiProfile
from attractor.agent.session import SessionConfig, detect_loop


def test_local_env_read_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        env.write_file(str(Path(tmpdir) / "test.txt"), "hello world")
        content = env.read_file(str(Path(tmpdir) / "test.txt"))
        assert "hello world" in content


def test_local_env_exec():
    env = LocalExecutionEnvironment()
    result = env.exec_command("echo hello", timeout_ms=5000)
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_local_env_exec_timeout():
    env = LocalExecutionEnvironment()
    result = env.exec_command("sleep 10", timeout_ms=1000)
    assert result.timed_out is True


def test_core_tools_registration():
    env = LocalExecutionEnvironment()
    registry = ToolRegistry()
    register_core_tools(registry, env)
    assert "read_file" in registry.names()
    assert "write_file" in registry.names()
    assert "edit_file" in registry.names()
    assert "shell" in registry.names()
    assert "grep" in registry.names()
    assert "glob" in registry.names()


def test_truncation_head_tail():
    big = "A" * 100 + "B" * 100
    result = truncate_output(big, 100, "head_tail")
    assert len(result) > 0
    assert "truncated" in result.lower() or "WARNING" in result
    assert result.startswith("A")
    assert result.endswith("B" * 50)


def test_truncation_tail():
    big = "X" * 200
    result = truncate_output(big, 100, "tail")
    assert len(result) <= 300
    assert "truncated" in result.lower() or "WARNING" in result


def test_truncation_no_truncate_small():
    small = "hello"
    result = truncate_output(small, 1000, "head_tail")
    assert result == small


def test_truncate_tool_output():
    big = "x" * 100000
    result = truncate_tool_output(big, "read_file", {})
    assert len(result) < len(big)


def test_anthropic_profile():
    env = LocalExecutionEnvironment()
    profile = AnthropicProfile()
    assert profile.id == "anthropic"
    tools = profile.tools()
    assert len(tools) > 0
    prompt = profile.build_system_prompt(env)
    assert len(prompt) > 0


def test_openai_profile():
    env = LocalExecutionEnvironment()
    profile = OpenAIProfile()
    assert profile.id == "openai"
    tool_names = [t.name for t in profile.tools()]
    assert "apply_patch" in tool_names


def test_gemini_profile():
    env = LocalExecutionEnvironment()
    profile = GeminiProfile()
    assert profile.id == "gemini"


def test_loop_detection_no_loop():
    calls = [("read", "a"), ("write", "b"), ("shell", "c")]
    assert detect_loop(calls, 10) is False


def test_loop_detection_pattern_1():
    calls = [("read", "a")] * 10
    assert detect_loop(calls, 10) is True


def test_loop_detection_pattern_2():
    pattern = [("read", "a"), ("write", "b")]
    calls = pattern * 5
    assert detect_loop(calls, 10) is True


def test_session_config_defaults():
    config = SessionConfig()
    assert config.max_turns == 0
    assert config.max_tool_rounds_per_input == 0
    assert config.default_command_timeout_ms == 10000
    assert config.enable_loop_detection is True
    assert config.max_input_tokens == 0
    assert config.max_output_tokens == 0
    assert config.max_total_tokens == 0


def test_session_metrics():
    from attractor.agent.session import Session
    from attractor.llm.types import Usage

    env = LocalExecutionEnvironment()
    profile = AnthropicProfile()
    session = Session(profile=profile, execution_env=env)

    # Simulate usage accumulation
    session.total_usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
    metrics = session.get_metrics()
    assert metrics["total_input_tokens"] == 100
    assert metrics["total_output_tokens"] == 50
    assert metrics["total_tokens"] == 150
    assert metrics["session_id"] == session.id
    assert metrics["state"] == "idle"


def test_budget_check():
    from attractor.agent.session import Session
    from attractor.llm.types import Usage

    env = LocalExecutionEnvironment()
    profile = AnthropicProfile()
    config = SessionConfig(max_total_tokens=1000)
    session = Session(profile=profile, execution_env=env, config=config)

    session.total_usage = Usage(input_tokens=500, output_tokens=400, total_tokens=900)
    assert session._check_budget() is False

    session.total_usage = Usage(input_tokens=600, output_tokens=500, total_tokens=1100)
    assert session._check_budget() is True


def test_conversation_export_import():
    from attractor.agent.session import (
        Session,
        UserTurn,
        AssistantTurn,
        ToolResultsTurn,
        SteeringTurn,
    )
    from attractor.llm.types import ToolCallData, ToolResultData

    env = LocalExecutionEnvironment()
    profile = AnthropicProfile()
    session = Session(profile=profile, execution_env=env)

    # Manually populate history
    session.history.append(UserTurn(content="Hello"))
    session.history.append(AssistantTurn(
        content="Hi there",
        tool_calls=[ToolCallData(id="tc1", name="shell", arguments={"command": "echo hi"})],
    ))
    session.history.append(ToolResultsTurn(
        results=[ToolResultData(tool_call_id="tc1", content="hi", is_error=False)],
    ))
    session.history.append(SteeringTurn(content="Focus on tests"))

    exported = session.export_conversation()
    assert exported["session_id"] == session.id
    assert len(exported["turns"]) == 4
    assert exported["turns"][0]["type"] == "user"
    assert exported["turns"][1]["type"] == "assistant"
    assert exported["turns"][2]["type"] == "tool_results"
    assert exported["turns"][3]["type"] == "steering"

    # Import into a new session
    session2 = Session(profile=profile, execution_env=env)
    session2.import_conversation(exported)
    assert len(session2.history) == 4
    assert isinstance(session2.history[0], UserTurn)
    assert session2.history[0].content == "Hello"
    assert isinstance(session2.history[1], AssistantTurn)
    assert len(session2.history[1].tool_calls) == 1
    assert session2.history[1].tool_calls[0].name == "shell"
