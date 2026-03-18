"""Shared core tools per Section 3.3 of the coding-agent-loop-spec.

Provides six tools: read_file, write_file, edit_file, shell, grep, glob.
Each has a ``ToolDefinition`` and an executor function.

Use ``register_core_tools(registry, env)`` to register all six tools
into a :class:`ToolRegistry`.
"""

from __future__ import annotations

import difflib
from typing import Any

from attractor.llm.types import ToolDefinition
from attractor.agent.execution.base import ExecutionEnvironment
from attractor.agent.tools.registry import RegisteredTool, ToolRegistry


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

READ_FILE_DEF = ToolDefinition(
    name="read_file",
    description=(
        "Read a file from the filesystem. Returns line-numbered content. "
        "For large files, use offset and limit to read a specific section."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file.",
            },
            "offset": {
                "type": "integer",
                "description": "1-based line number to start reading from.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read. Default: 2000.",
            },
        },
        "required": ["file_path"],
    },
)

WRITE_FILE_DEF = ToolDefinition(
    name="write_file",
    description=(
        "Write content to a file. Creates the file and parent directories "
        "if needed. Overwrites any existing content."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file.",
            },
            "content": {
                "type": "string",
                "description": "The full file content to write.",
            },
        },
        "required": ["file_path", "content"],
    },
)

EDIT_FILE_DEF = ToolDefinition(
    name="edit_file",
    description=(
        "Edit a file by either replacing an exact string or applying a unified diff patch. "
        "Use old_string/new_string for simple replacements, or patch for unified diffs. "
        "These modes are mutually exclusive."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file.",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find in the file.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences. Default: false.",
            },
            "patch": {
                "type": "string",
                "description": "Unified diff to apply (mutually exclusive with old_string/new_string).",
            },
        },
        "required": ["file_path"],
    },
)

SHELL_DEF = ToolDefinition(
    name="shell",
    description=(
        "Execute a shell command. Returns stdout, stderr, and exit code. "
        "Commands run with a default timeout; set timeout_ms to override."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to run.",
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Command timeout in milliseconds. Overrides the session default.",
            },
            "description": {
                "type": "string",
                "description": "Human-readable description of what this command does.",
            },
        },
        "required": ["command"],
    },
)

GREP_DEF = ToolDefinition(
    name="grep",
    description=(
        "Search file contents using regex patterns. "
        "Returns matching lines with file paths and line numbers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search. Defaults to the working directory.",
            },
            "glob_filter": {
                "type": "string",
                "description": 'File pattern filter (e.g., "*.py").',
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Case-insensitive search. Default: false.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines. Default: 100.",
            },
            "output_mode": {
                "type": "string",
                "description": 'Output mode: "content" (matching lines) or "files_with_matches" (file paths only). Default: "content".',
                "enum": ["content", "files_with_matches"],
            },
        },
        "required": ["pattern"],
    },
)

GLOB_DEF = ToolDefinition(
    name="glob",
    description=(
        "Find files matching a glob pattern. "
        "Results are sorted by modification time (newest first)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": 'Glob pattern (e.g., "**/*.ts").',
            },
            "path": {
                "type": "string",
                "description": "Base directory to search. Defaults to the working directory.",
            },
        },
        "required": ["pattern"],
    },
)


# ---------------------------------------------------------------------------
# Executor functions
# ---------------------------------------------------------------------------


def _exec_read_file(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    file_path: str = arguments["file_path"]
    offset: int | None = arguments.get("offset")
    limit: int | None = arguments.get("limit", 2000)
    return env.read_file(file_path, offset=offset, limit=limit)


def _exec_write_file(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    file_path: str = arguments["file_path"]
    content: str = arguments["content"]
    env.write_file(file_path, content)
    byte_count = len(content.encode("utf-8"))
    return f"Successfully wrote {byte_count} bytes to {file_path}"


def _apply_unified_diff(file_path: str, patch: str, env: ExecutionEnvironment) -> str:
    """Apply a unified diff patch to a file."""
    from pathlib import Path

    p = Path(file_path)
    if not p.is_absolute():
        p = Path(env.working_directory()) / p

    if p.exists():
        original = p.read_text(encoding="utf-8", errors="replace")
        original_lines = original.splitlines(keepends=True)
    else:
        original_lines = []

    # Parse unified diff hunks
    patch_lines = patch.splitlines(keepends=True)
    # Ensure all lines have newline endings for processing
    patch_lines = [l if l.endswith("\n") else l + "\n" for l in patch_lines]

    hunks: list[tuple[int, int, list[str]]] = []
    i = 0
    while i < len(patch_lines):
        line = patch_lines[i]
        if line.startswith("@@"):
            # Parse hunk header: @@ -start,count +start,count @@
            import re
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if not m:
                i += 1
                continue
            old_start = int(m.group(1))
            hunk_lines: list[str] = []
            i += 1
            while i < len(patch_lines) and not patch_lines[i].startswith("@@") and not patch_lines[i].startswith("diff ") and not patch_lines[i].startswith("---") and not patch_lines[i].startswith("+++"):
                hunk_lines.append(patch_lines[i])
                i += 1
            hunks.append((old_start, 0, hunk_lines))
        else:
            i += 1

    if not hunks:
        raise ValueError("No valid hunks found in the patch.")

    # Apply hunks in reverse order to preserve line numbers
    result_lines = list(original_lines)
    for old_start, _, hunk_lines in reversed(hunks):
        # Build expected removed lines and new lines
        remove_lines: list[str] = []
        add_lines: list[str] = []
        context_before = 0
        for hl in hunk_lines:
            if hl.startswith("-"):
                remove_lines.append(hl[1:])
            elif hl.startswith("+"):
                add_lines.append(hl[1:])
            elif hl.startswith(" "):
                # Context line — part of both old and new
                remove_lines.append(hl[1:])
                add_lines.append(hl[1:])

        # Find where to apply (0-indexed)
        start_idx = old_start - 1
        # Replace the old lines with new lines
        result_lines[start_idx:start_idx + len(remove_lines)] = add_lines

    content = "".join(result_lines)
    env.write_file(file_path, content)
    return f"Applied patch to {file_path} ({len(hunks)} hunk(s))."


def _exec_edit_file(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    file_path: str = arguments["file_path"]

    # Patch mode
    if "patch" in arguments:
        if "old_string" in arguments or "new_string" in arguments:
            raise ValueError("patch is mutually exclusive with old_string/new_string.")
        return _apply_unified_diff(file_path, arguments["patch"], env)

    old_string: str = arguments["old_string"]
    new_string: str = arguments["new_string"]
    replace_all: bool = arguments.get("replace_all", False)

    # Read current content
    from pathlib import Path

    p = Path(file_path)
    if not p.is_absolute():
        p = Path(env.working_directory()) / p
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = p.read_text(encoding="utf-8", errors="replace")

    # Exact match
    count = content.count(old_string)

    if count == 0:
        # Attempt fuzzy fallback: normalize whitespace
        normalized_content = " ".join(content.split())
        normalized_old = " ".join(old_string.split())
        if normalized_old in normalized_content:
            # Find the actual substring using line-level matching
            old_lines = old_string.splitlines(keepends=True)
            content_lines = content.splitlines(keepends=True)
            matcher = difflib.SequenceMatcher(
                None,
                [l.strip() for l in content_lines],
                [l.strip() for l in old_lines],
            )
            best = max(
                matcher.get_matching_blocks(), key=lambda m: m.size
            )
            if best.size >= max(1, len(old_lines) - 1):
                start_line = best.a
                end_line = best.a + len(old_lines)
                original_chunk = "".join(content_lines[start_line:end_line])
                content = content.replace(original_chunk, new_string, 1)
                env.write_file(file_path, content)
                return (
                    f"Replaced 1 occurrence in {file_path} (fuzzy match: "
                    f"whitespace differences were normalized)."
                )
        raise ValueError(
            f"old_string not found in {file_path}. "
            f"Make sure the text matches exactly, including whitespace and indentation."
        )

    if count > 1 and not replace_all:
        raise ValueError(
            f"old_string found {count} times in {file_path}. "
            f"Provide more surrounding context to make the match unique, "
            f"or set replace_all=true to replace all occurrences."
        )

    if replace_all:
        content = content.replace(old_string, new_string)
    else:
        content = content.replace(old_string, new_string, 1)

    env.write_file(file_path, content)
    replaced = count if replace_all else 1
    return f"Replaced {replaced} occurrence(s) in {file_path}."


def _exec_shell(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    command: str = arguments["command"]
    timeout_ms: int | None = arguments.get("timeout_ms")

    kwargs: dict[str, Any] = {"command": command}
    if timeout_ms is not None:
        kwargs["timeout_ms"] = timeout_ms

    result = env.exec_command(**kwargs)

    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(result.stderr)
    if not parts:
        parts.append("(no output)")

    output = "\n".join(parts)
    output += f"\n\nExit code: {result.exit_code}"
    output += f"\nDuration: {result.duration_ms}ms"
    if result.timed_out:
        output += "\n(command timed out)"
    return output


def _exec_grep(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    pattern: str = arguments["pattern"]
    path: str = arguments.get("path", env.working_directory())
    options: dict[str, Any] = {}
    if "glob_filter" in arguments:
        options["glob_filter"] = arguments["glob_filter"]
    if "case_insensitive" in arguments:
        options["case_insensitive"] = arguments["case_insensitive"]
    if "max_results" in arguments:
        options["max_results"] = arguments["max_results"]
    if "output_mode" in arguments:
        options["output_mode"] = arguments["output_mode"]
    return env.grep(pattern, path, **options)


def _exec_glob(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    pattern: str = arguments["pattern"]
    path: str = arguments.get("path", env.working_directory())
    results = env.glob(pattern, path)
    if not results:
        return "No files matched the pattern."
    return "\n".join(results)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_CORE_TOOLS: list[RegisteredTool] = [
    RegisteredTool(definition=READ_FILE_DEF, executor=_exec_read_file),
    RegisteredTool(definition=WRITE_FILE_DEF, executor=_exec_write_file),
    RegisteredTool(definition=EDIT_FILE_DEF, executor=_exec_edit_file),
    RegisteredTool(definition=SHELL_DEF, executor=_exec_shell),
    RegisteredTool(definition=GREP_DEF, executor=_exec_grep),
    RegisteredTool(definition=GLOB_DEF, executor=_exec_glob),
]


def register_core_tools(registry: ToolRegistry, env: ExecutionEnvironment) -> None:
    """Register all six core tools into *registry*."""
    for tool in _CORE_TOOLS:
        registry.register(tool)
