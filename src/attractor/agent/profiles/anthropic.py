"""AnthropicProfile (Claude Code-aligned) per Section 3.5 of the coding-agent-loop-spec.

- Registers core tools with edit_file as the native editing format.
- System prompt mirrors Claude Code style.
- Default command timeout: 120s.
"""

from __future__ import annotations

import datetime

from attractor.llm.types import ToolDefinition
from attractor.agent.execution.base import ExecutionEnvironment
from attractor.agent.profiles.base import ProviderProfile
from attractor.agent.tools.core import register_core_tools
from attractor.agent.tools.registry import ToolRegistry


class AnthropicProfile(ProviderProfile):
    """Claude Code-aligned provider profile for Anthropic models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        *,
        context_window_size: int = 200_000,
    ) -> None:
        self.id = "anthropic"
        self.model = model
        self.tool_registry = ToolRegistry()
        self.supports_reasoning = True
        self.supports_streaming = True
        self.supports_parallel_tool_calls = True
        self.context_window_size = context_window_size
        self.default_command_timeout_ms = 120_000

        # Register core tools -- edit_file is the native editing format
        # We pass a placeholder env; actual execution uses the session env.
        # Registration only needs definitions, not the env.
        register_core_tools(self.tool_registry, _PlaceholderEnv())

    def build_system_prompt(
        self, environment: ExecutionEnvironment, project_docs: str = ""
    ) -> str:
        today = datetime.date.today().isoformat()
        platform = environment.platform()
        working_dir = environment.working_directory()
        os_version = environment.os_version()

        is_git = environment.is_git_repo()
        git_branch = environment.git_branch() if is_git else ""
        git_context = environment.git_context() if is_git else ""

        prompt = f"""You are an expert software engineer and an AI assistant. You help users with coding tasks by reading files, editing code, running commands, and iterating until the task is done.

You have access to tools for interacting with the user's codebase and development environment. Use them to explore, understand, and modify code.

# Tool Usage Guidelines

- **Read before edit.** Always read a file before editing it. Never assume file contents.
- **Edit over write.** Prefer ``edit_file`` to ``write_file`` for modifying existing files. Only use ``write_file`` for creating entirely new files.
- **Unique match.** When using ``edit_file``, the ``old_string`` must uniquely identify the text to replace. Include enough surrounding context (3-5 lines) to make it unique. If the string appears multiple times, include more context.
- **Exact match.** The ``old_string`` must match the file content exactly, including whitespace and indentation. Read the file first to get the exact text.
- **Minimal edits.** Make the smallest edit that accomplishes the goal. Do not reformat or restructure code unnecessarily.
- **Verify changes.** After editing, consider reading the file to verify your changes, or run tests/linters to validate.
- **Shell for verification.** Use the shell tool to run tests, linters, type checkers, and build commands to verify your work.

# Coding Best Practices

- Follow the existing code style and conventions in the project.
- Write clear, maintainable code with appropriate comments.
- Handle errors gracefully.
- Prefer editing existing files over creating new ones.
- Do not add unnecessary dependencies.
- When creating files, always create them with complete, working content.

<environment>
Working directory: {working_dir}
Platform: {platform}
OS version: {os_version}
Today's date: {today}
Knowledge cutoff: May 2025
Model: {self.model}
Is git repository: {is_git}
Git branch: {git_branch}
</environment>"""

        if git_context:
            prompt += f"\n\n<git-context>\n{git_context}\n</git-context>"

        if project_docs:
            prompt += f"\n\n# Project Instructions\n\n{project_docs}"

        return prompt

    def tools(self) -> list[ToolDefinition]:
        return self.tool_registry.definitions()

    def provider_options(self) -> dict | None:
        return {
            "anthropic": {
                "beta_headers": ["interleaved-thinking-2025-05-14"],
            }
        }


class _PlaceholderEnv(ExecutionEnvironment):
    """Minimal placeholder used only during tool registration."""

    def read_file(self, path, offset=None, limit=None):
        raise NotImplementedError

    def write_file(self, path, content):
        raise NotImplementedError

    def file_exists(self, path):
        raise NotImplementedError

    def list_directory(self, path, depth=1):
        raise NotImplementedError

    def exec_command(self, command, timeout_ms=10000, working_dir=None, env_vars=None):
        raise NotImplementedError

    def grep(self, pattern, path, **options):
        raise NotImplementedError

    def glob(self, pattern, path):
        raise NotImplementedError

    def working_directory(self):
        return "."

    def platform(self):
        return "unknown"
