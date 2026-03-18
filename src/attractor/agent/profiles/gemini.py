"""GeminiProfile (gemini-cli-aligned) per Section 3.6 of the coding-agent-loop-spec.

- Registers core tools + list_dir.
- System prompt mirrors gemini-cli style.
"""

from __future__ import annotations

import datetime
from typing import Any

from attractor.llm.types import ToolDefinition
from attractor.agent.execution.base import ExecutionEnvironment
from attractor.agent.profiles.base import ProviderProfile
from attractor.agent.tools.core import register_core_tools
from attractor.agent.tools.registry import RegisteredTool, ToolRegistry


# ---------------------------------------------------------------------------
# Gemini-specific tools
# ---------------------------------------------------------------------------

READ_MANY_FILES_DEF = ToolDefinition(
    name="read_many_files",
    description=(
        "Read multiple files in a single call. More efficient than calling "
        "read_file repeatedly. Returns the contents of each file separated by headers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths to read.",
            },
        },
        "required": ["paths"],
    },
)


def _exec_read_many_files(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Execute the read_many_files tool."""
    paths: list[str] = arguments["paths"]
    results: list[str] = []
    for path in paths:
        try:
            content = env.read_file(path)
            results.append(f"=== {path} ===\n{content}")
        except Exception as exc:
            results.append(f"=== {path} ===\nError: {exc}")
    return "\n\n".join(results)


LIST_DIR_DEF = ToolDefinition(
    name="list_dir",
    description=(
        "List the contents of a directory with optional depth for recursive listing."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the directory to list.",
            },
            "depth": {
                "type": "integer",
                "description": "How many levels deep to recurse. Default: 1.",
            },
        },
        "required": ["path"],
    },
)


def _exec_list_dir(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Execute the list_dir tool."""
    path: str = arguments["path"]
    depth: int = arguments.get("depth", 1)

    entries = env.list_directory(path, depth=depth)
    if not entries:
        return "(empty directory)"

    lines: list[str] = []
    for entry in entries:
        suffix = "/" if entry.is_dir else ""
        size_str = f"  ({entry.size} bytes)" if entry.size is not None else ""
        lines.append(f"{entry.name}{suffix}{size_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


class GeminiProfile(ProviderProfile):
    """gemini-cli-aligned provider profile for Gemini models."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        context_window_size: int = 1_000_000,
    ) -> None:
        self.id = "gemini"
        self.model = model
        self.tool_registry = ToolRegistry()
        self.supports_reasoning = True
        self.supports_streaming = True
        self.supports_parallel_tool_calls = True
        self.context_window_size = context_window_size
        self.default_command_timeout_ms = 10_000

        # Register core tools
        register_core_tools(self.tool_registry, _PlaceholderEnv())

        # Register Gemini-specific tools
        self.tool_registry.register(
            RegisteredTool(definition=READ_MANY_FILES_DEF, executor=_exec_read_many_files)
        )
        self.tool_registry.register(
            RegisteredTool(definition=LIST_DIR_DEF, executor=_exec_list_dir)
        )

    def build_system_prompt(
        self, environment: ExecutionEnvironment, project_docs: str = ""
    ) -> str:
        today = datetime.date.today().isoformat()
        platform = environment.platform()
        working_dir = environment.working_directory()
        os_version = environment.os_version()

        prompt = f"""You are a helpful coding assistant powered by Gemini. You help users understand, write, and modify code by using the available tools to interact with their project.

# How to Work

1. **Understand first.** Use `read_file`, `grep`, `glob`, and `list_dir` to explore the codebase before making changes.
2. **Make precise edits.** Use `edit_file` for modifications to existing files. Use `write_file` to create new files.
3. **Verify your work.** Use `shell` to run tests, linters, builds, or any other verification commands.
4. **Search effectively.** Use `grep` for content search and `glob` for file discovery. Use `list_dir` to understand project structure.

# Tool Guidelines

- Always read a file before editing it.
- When editing, the `old_string` must exactly match the text in the file, including whitespace.
- Prefer small, targeted edits over rewriting entire files.
- Use `list_dir` with depth > 1 to understand directory structure.
- Follow the project's existing coding style and conventions.
- Handle errors gracefully and provide clear explanations.

# Coding Best Practices

- Write clean, readable, and maintainable code.
- Follow the language's idioms and best practices.
- Add appropriate error handling.
- Write descriptive commit messages when asked to commit.
- Respect existing project configuration and dependencies.

<environment>
Working directory: {working_dir}
Platform: {platform}
OS version: {os_version}
Today's date: {today}
Knowledge cutoff: May 2025
Model: {self.model}
Is git repository: {environment.is_git_repo()}
Git branch: {environment.git_branch()}
</environment>"""

        git_context = environment.git_context()
        if git_context:
            prompt += f"\n\n<git-context>\n{git_context}\n</git-context>"

        if project_docs:
            prompt += f"\n\n# Project Instructions\n\n{project_docs}"

        return prompt

    def tools(self) -> list[ToolDefinition]:
        return self.tool_registry.definitions()

    def provider_options(self) -> dict | None:
        return {
            "gemini": {
                "safety_settings": [],
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
