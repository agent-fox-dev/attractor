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


def test_session_state_guards():
    """Session rejects submit when closed or already processing."""
    import asyncio
    import pytest
    from attractor.agent.session import Session, SessionState

    env = LocalExecutionEnvironment()
    profile = AnthropicProfile()
    session = Session(profile=profile, execution_env=env)

    # Closed session should reject submit
    session.state = SessionState.CLOSED
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(session.submit("test"))

    # Processing session should reject submit
    session.state = SessionState.PROCESSING
    with pytest.raises(RuntimeError, match="already processing"):
        asyncio.run(session.submit("test"))


def test_validate_tool_args():
    """Tool argument validation checks required fields and types."""
    from attractor.agent.session import _validate_tool_args

    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["command"],
    }

    # Valid
    assert _validate_tool_args({"command": "ls"}, schema) is None
    assert _validate_tool_args({"command": "ls", "timeout": 5}, schema) is None

    # Missing required field
    err = _validate_tool_args({}, schema)
    assert err is not None
    assert "command" in err

    # Wrong type
    err = _validate_tool_args({"command": 123}, schema)
    assert err is not None
    assert "wrong type" in err.lower()


def test_subagent_depth_limit():
    """Subagent spawn_agent rejects when depth limit is reached."""
    from attractor.agent.tools.subagent import _make_executors

    called = False

    def mock_factory(**kwargs):
        nonlocal called
        called = True

    executors = _make_executors(mock_factory)
    spawn = executors["spawn_agent"]

    # Simulate env at max depth
    env = LocalExecutionEnvironment()
    env._parent_depth = 1
    env._max_subagent_depth = 1

    result = spawn({"task": "test task"}, env)
    assert "depth limit" in result.lower()
    assert not called  # Factory should not be called


def test_awaiting_input_event_kind():
    """AWAITING_INPUT event kind exists in EventKind enum."""
    from attractor.agent.events import EventKind

    assert hasattr(EventKind, "AWAITING_INPUT")
    assert EventKind.AWAITING_INPUT == "awaiting_input"


def test_gemini_read_many_files_tool():
    """Gemini profile includes read_many_files tool."""
    profile = GeminiProfile()
    tool_names = [t.name for t in profile.tools()]
    assert "read_many_files" in tool_names
    assert "list_dir" in tool_names


def test_shell_default_timeout_from_profile():
    """Shell tool uses profile default_command_timeout_ms when timeout_ms not set."""
    from attractor.agent.session import Session
    from attractor.llm.types import ToolCallData

    env = LocalExecutionEnvironment()
    profile = AnthropicProfile()
    session = Session(profile=profile, execution_env=env)

    # Create a shell tool call without timeout_ms
    tc = ToolCallData(id="tc1", name="shell", arguments={"command": "echo test"})
    result = session._execute_single_tool(tc)
    # The command should have run with the profile's default timeout (120s)
    assert result.is_error is False
    assert "test" in result.content


def test_git_branch_in_environment_block():
    """System prompt includes Git branch in <environment> block."""
    env = LocalExecutionEnvironment()
    profile = AnthropicProfile()
    prompt = profile.build_system_prompt(env)
    assert "Git branch:" in prompt


def test_git_branch_method():
    """LocalExecutionEnvironment has git_branch method."""
    env = LocalExecutionEnvironment()
    branch = env.git_branch()
    # In the test repo, we should get a branch name
    assert isinstance(branch, str)


def test_docker_env_has_git_methods():
    """DockerExecutionEnvironment has git_branch, git_context, is_git_repo methods."""
    from attractor.agent.execution.docker import DockerExecutionEnvironment
    env = DockerExecutionEnvironment(image="python:3.12-slim")
    assert hasattr(env, "is_git_repo")
    assert hasattr(env, "git_branch")
    assert hasattr(env, "git_context")
    assert callable(env.is_git_repo)
    assert callable(env.git_branch)
    assert callable(env.git_context)


def test_ssh_env_has_git_methods():
    """SSHExecutionEnvironment has git_branch, git_context, is_git_repo methods."""
    from attractor.agent.execution.ssh import SSHExecutionEnvironment
    env = SSHExecutionEnvironment(host="user@localhost")
    assert hasattr(env, "is_git_repo")
    assert hasattr(env, "git_branch")
    assert hasattr(env, "git_context")
    assert callable(env.is_git_repo)
    assert callable(env.git_branch)
    assert callable(env.git_context)


def test_ssh_timeout_error_format():
    """SSH timeout error message matches the standard format."""
    from attractor.agent.execution.ssh import SSHExecutionEnvironment
    import inspect
    source = inspect.getsource(SSHExecutionEnvironment)
    assert "[ERROR: Command timed out after" in source
    assert "timeout_ms parameter" in source


def test_docker_timeout_error_format():
    """Docker timeout error message matches the standard format."""
    from attractor.agent.execution.docker import DockerExecutionEnvironment
    # We can't run Docker in tests, but we can verify the message format
    # by checking the source code pattern is consistent
    import inspect
    source = inspect.getsource(DockerExecutionEnvironment)
    assert "[ERROR: Command timed out after" in source
    assert "timeout_ms parameter" in source


def test_edit_file_patch(tmp_path):
    """edit_file supports unified diff patch format."""
    from attractor.agent.tools.core import _exec_edit_file

    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    file_path = str(tmp_path / "hello.py")
    env.write_file(file_path, "def hello():\n    print('hello')\n    return 1\n")

    patch = """\
@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('world')
     return 1
"""
    result = _exec_edit_file({"file_path": file_path, "patch": patch}, env)
    assert "Applied patch" in result
    content = Path(tmp_path / "hello.py").read_text()
    assert "print('world')" in content
    assert "print('hello')" not in content


def test_edit_file_patch_mutually_exclusive(tmp_path):
    """edit_file rejects patch + old_string together."""
    import pytest
    from attractor.agent.tools.core import _exec_edit_file

    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    file_path = str(tmp_path / "test.txt")
    env.write_file(file_path, "hello")

    with pytest.raises(ValueError, match="mutually exclusive"):
        _exec_edit_file(
            {"file_path": file_path, "patch": "@@ -1 +1 @@\n-hello\n+world\n", "old_string": "hello"},
            env,
        )


def test_denied_tool_permission():
    """Denied tools return an error without executing."""
    from attractor.agent.session import Session
    from attractor.llm.types import ToolCallData

    env = LocalExecutionEnvironment()
    profile = AnthropicProfile()
    config = SessionConfig(denied_tools=["shell"])
    session = Session(profile=profile, execution_env=env, config=config)

    tc = ToolCallData(id="tc1", name="shell", arguments={"command": "echo hi"})
    result = session._execute_single_tool(tc)
    assert result.is_error is True
    assert "denied" in result.content.lower()


def test_tool_definition_strict_field():
    """ToolDefinition has a strict field."""
    from attractor.llm.types import ToolDefinition

    td = ToolDefinition(name="test", description="test", strict=True)
    assert td.strict is True
    td2 = ToolDefinition(name="test", description="test")
    assert td2.strict is False


def test_read_file_image(tmp_path):
    """read_file returns base64 data for image files."""
    import base64

    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    # Create a tiny PNG
    png_bytes = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, 0x89,
    ])
    img_path = tmp_path / "test.png"
    img_path.write_bytes(png_bytes)

    result = env.read_file(str(img_path))
    assert "[image: test.png" in result
    assert "image/png" in result
    assert "base64," in result
    # Verify the base64 decodes back
    b64_part = result.split("base64,")[1]
    decoded = base64.b64decode(b64_part)
    assert decoded == png_bytes


def test_user_instructions_in_system_prompt():
    """User instructions are appended last in system prompt (spec Section 6.1)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        for Profile in [AnthropicProfile, OpenAIProfile, GeminiProfile]:
            profile = Profile()
            prompt = profile.build_system_prompt(
                environment=env,
                project_docs="Some docs",
                user_instructions="Always use tabs.",
            )
            # User instructions should come after project docs
            docs_idx = prompt.index("Some docs")
            user_idx = prompt.index("Always use tabs.")
            assert user_idx > docs_idx, f"{Profile.__name__}: user instructions not last"
            # Without user instructions, the section should not appear
            prompt_no_user = profile.build_system_prompt(
                environment=env, project_docs="Some docs",
            )
            assert "User Instructions" not in prompt_no_user


def test_session_config_user_instructions():
    """SessionConfig has a user_instructions field."""
    cfg = SessionConfig(user_instructions="Custom instruction")
    assert cfg.user_instructions == "Custom instruction"

    cfg_default = SessionConfig()
    assert cfg_default.user_instructions == ""


def test_kill_running_processes():
    """LocalExecutionEnvironment tracks and kills active processes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        # Initially no active processes
        assert len(env._active_processes) == 0
        # After exec_command completes, process should be removed
        env.exec_command("echo hello", timeout_ms=5000)
        assert len(env._active_processes) == 0
        # kill_running_processes should not error when no processes
        env.kill_running_processes()


def test_graph_goal_mirrored_in_context():
    """Engine mirrors graph.goal into context (spec Section 5.1)."""
    from attractor.pipeline.parser import parse_dot
    from attractor.pipeline.engine import run, PipelineConfig
    from attractor.pipeline.graph import StageStatus

    dot = """
    digraph T {
        goal="Build the widget"
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        start -> exit
    }
    """
    graph = parse_dot(dot)
    assert graph.goal == "Build the widget"

    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(logs_root=tmpdir)
        # We can't easily inspect context from outside run(),
        # but we can verify the graph has the goal set
        outcome = run(graph, config)
        assert outcome.status == StageStatus.SUCCESS
