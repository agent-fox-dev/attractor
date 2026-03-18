"""OpenAIProfile (codex-rs-aligned) per Section 3.4 of the coding-agent-loop-spec.

- Registers core tools + apply_patch.
- System prompt mirrors codex-rs style.
- apply_patch tool with v4a format parser (Appendix A of the spec).
"""

from __future__ import annotations

import datetime
import os
import re
from pathlib import Path
from typing import Any

from attractor.llm.types import ToolDefinition
from attractor.agent.execution.base import ExecutionEnvironment
from attractor.agent.profiles.base import ProviderProfile
from attractor.agent.tools.core import register_core_tools
from attractor.agent.tools.registry import RegisteredTool, ToolRegistry


# ---------------------------------------------------------------------------
# apply_patch v4a format parser (Appendix A)
# ---------------------------------------------------------------------------

APPLY_PATCH_DEF = ToolDefinition(
    name="apply_patch",
    description=(
        "Apply code changes using the v4a patch format. "
        "Supports creating, deleting, and modifying files in a single operation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": "The patch content in v4a format.",
            },
        },
        "required": ["patch"],
    },
)


def _parse_v4a_patch(patch_text: str) -> list[dict[str, Any]]:
    """Parse a v4a format patch into a list of operations.

    Returns a list of dicts, each with:
        - ``op``: "add", "delete", or "update"
        - ``path``: file path
        - ``move_to``: new path (for renames, update only)
        - ``content``: full file content (add only)
        - ``hunks``: list of hunk dicts (update only)
    """
    lines = patch_text.split("\n")
    operations: list[dict[str, Any]] = []
    i = 0

    # Skip to "*** Begin Patch"
    while i < len(lines) and lines[i].strip() != "*** Begin Patch":
        i += 1
    i += 1  # skip the Begin Patch line

    while i < len(lines):
        line = lines[i].strip()

        if line == "*** End Patch":
            break

        if line.startswith("*** Add File: "):
            path = line[len("*** Add File: "):]
            i += 1
            content_lines: list[str] = []
            while i < len(lines):
                l = lines[i]
                if l.startswith("***"):
                    break
                if l.startswith("+"):
                    content_lines.append(l[1:])
                i += 1
            operations.append({
                "op": "add",
                "path": path,
                "content": "\n".join(content_lines),
            })

        elif line.startswith("*** Delete File: "):
            path = line[len("*** Delete File: "):]
            i += 1
            operations.append({"op": "delete", "path": path})

        elif line.startswith("*** Update File: "):
            path = line[len("*** Update File: "):]
            i += 1
            move_to: str | None = None

            if i < len(lines) and lines[i].strip().startswith("*** Move to: "):
                move_to = lines[i].strip()[len("*** Move to: "):]
                i += 1

            # Parse hunks
            hunks: list[dict[str, Any]] = []
            while i < len(lines):
                l = lines[i]
                if l.strip().startswith("***") and not l.strip().startswith("*** End of File"):
                    break

                if l.startswith("@@ "):
                    context_hint = l[3:].strip()
                    i += 1
                    hunk_lines: list[tuple[str, str]] = []  # (type, line)
                    while i < len(lines):
                        hl = lines[i]
                        if hl.startswith("@@ ") or (
                            hl.strip().startswith("***")
                            and not hl.strip().startswith("*** End of File")
                        ):
                            break
                        if hl.strip() == "*** End of File":
                            i += 1
                            break
                        if hl.startswith("+"):
                            hunk_lines.append(("add", hl[1:]))
                        elif hl.startswith("-"):
                            hunk_lines.append(("delete", hl[1:]))
                        elif hl.startswith(" "):
                            hunk_lines.append(("context", hl[1:]))
                        else:
                            # Treat as context if no prefix
                            hunk_lines.append(("context", hl))
                        i += 1
                    hunks.append({
                        "context_hint": context_hint,
                        "lines": hunk_lines,
                    })
                else:
                    i += 1

            op: dict[str, Any] = {"op": "update", "path": path, "hunks": hunks}
            if move_to:
                op["move_to"] = move_to
            operations.append(op)

        else:
            i += 1

    return operations


def _find_hunk_position(
    file_lines: list[str], hunk: dict[str, Any]
) -> int | None:
    """Find the line index in *file_lines* where *hunk* should be applied.

    Uses the context lines from the hunk for matching.  Falls back to
    fuzzy (whitespace-normalized) matching if exact match fails.
    """
    # Build the sequence of context + delete lines that must match
    match_lines: list[str] = []
    for line_type, line_text in hunk["lines"]:
        if line_type in ("context", "delete"):
            match_lines.append(line_text)

    if not match_lines:
        return 0

    # Exact match
    for start in range(len(file_lines) - len(match_lines) + 1):
        if all(
            file_lines[start + j] == match_lines[j]
            for j in range(len(match_lines))
        ):
            return start

    # Fuzzy match (whitespace normalization)
    normalized_match = [l.strip() for l in match_lines]
    for start in range(len(file_lines) - len(match_lines) + 1):
        if all(
            file_lines[start + j].strip() == normalized_match[j]
            for j in range(len(match_lines))
        ):
            return start

    # Try context hint
    context_hint = hunk.get("context_hint", "").strip()
    if context_hint:
        for idx, fl in enumerate(file_lines):
            if context_hint in fl:
                return idx

    return None


def _apply_hunk(file_lines: list[str], hunk: dict[str, Any], pos: int) -> list[str]:
    """Apply a single hunk at position *pos*, returning the new file lines."""
    result = list(file_lines[:pos])
    i = pos

    for line_type, line_text in hunk["lines"]:
        if line_type == "context":
            result.append(file_lines[i] if i < len(file_lines) else line_text)
            i += 1
        elif line_type == "delete":
            i += 1  # skip the deleted line
        elif line_type == "add":
            result.append(line_text)

    # Append remaining lines
    result.extend(file_lines[i:])
    return result


def _exec_apply_patch(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Execute the apply_patch tool."""
    patch_text: str = arguments["patch"]
    operations = _parse_v4a_patch(patch_text)

    if not operations:
        raise ValueError("No operations found in patch. Check the patch format.")

    affected: list[str] = []
    for op in operations:
        path = op["path"]

        if op["op"] == "add":
            env.write_file(path, op["content"])
            affected.append(f"Created: {path}")

        elif op["op"] == "delete":
            # Read to confirm existence, then overwrite with empty (or use os.remove)
            full_path = Path(path)
            if not full_path.is_absolute():
                full_path = Path(env.working_directory()) / full_path
            if full_path.exists():
                full_path.unlink()
                affected.append(f"Deleted: {path}")
            else:
                affected.append(f"Already absent: {path}")

        elif op["op"] == "update":
            # Read current file
            full_path = Path(path)
            if not full_path.is_absolute():
                full_path = Path(env.working_directory()) / full_path
            if not full_path.exists():
                raise FileNotFoundError(f"Cannot update non-existent file: {path}")

            content = full_path.read_text(encoding="utf-8", errors="replace")
            file_lines = content.split("\n")

            # Apply hunks in order
            for hunk in op["hunks"]:
                pos = _find_hunk_position(file_lines, hunk)
                if pos is None:
                    ctx = hunk.get("context_hint", "(no context)")
                    raise ValueError(
                        f"Could not locate hunk in {path} near: {ctx}"
                    )
                file_lines = _apply_hunk(file_lines, hunk, pos)

            new_content = "\n".join(file_lines)

            # Handle rename
            write_path = op.get("move_to", path)
            env.write_file(write_path, new_content)

            if "move_to" in op:
                # Delete old file if rename
                if full_path.exists() and str(full_path) != str(
                    Path(env.working_directory()) / write_path
                    if not Path(write_path).is_absolute()
                    else Path(write_path)
                ):
                    full_path.unlink()
                affected.append(f"Updated + renamed: {path} -> {write_path}")
            else:
                affected.append(f"Updated: {path}")

    return "\n".join(affected)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


class OpenAIProfile(ProviderProfile):
    """codex-rs-aligned provider profile for OpenAI models."""

    def __init__(
        self,
        model: str = "gpt-5.2-codex",
        *,
        context_window_size: int = 200_000,
    ) -> None:
        self.id = "openai"
        self.model = model
        self.tool_registry = ToolRegistry()
        self.supports_reasoning = True
        self.supports_streaming = True
        self.supports_parallel_tool_calls = True
        self.context_window_size = context_window_size
        self.default_command_timeout_ms = 10_000

        # Register core tools
        register_core_tools(self.tool_registry, _PlaceholderEnv())

        # Register apply_patch (OpenAI-specific)
        self.tool_registry.register(
            RegisteredTool(definition=APPLY_PATCH_DEF, executor=_exec_apply_patch)
        )

    def build_system_prompt(
        self, environment: ExecutionEnvironment, project_docs: str = ""
    ) -> str:
        today = datetime.date.today().isoformat()
        platform = environment.platform()
        working_dir = environment.working_directory()
        os_version = environment.os_version()

        prompt = f"""You are a coding assistant. You help users by reading, writing, and editing code in their projects. You have access to tools that let you interact with the filesystem and run commands.

# Guidelines

- Use the `apply_patch` tool for making code changes. It supports creating, deleting, and modifying files using the v4a patch format.
- Use `read_file` to understand existing code before making changes.
- Use `shell` to run commands such as tests, linters, and builds to verify your work.
- Use `grep` and `glob` to search and navigate the codebase.
- When creating new files, you may use either `apply_patch` (Add File) or `write_file`.
- Always verify your changes by running relevant tests or commands.
- Follow the existing code style and conventions of the project.
- Keep changes minimal and focused on the task.

# apply_patch Format

The `apply_patch` tool accepts patches in v4a format:
- `*** Add File: <path>` to create new files (lines prefixed with `+`)
- `*** Delete File: <path>` to delete files
- `*** Update File: <path>` to modify files with context-based hunks
  - `@@` lines provide context hints for locating changes
  - Space-prefixed lines are context (unchanged)
  - `-` prefixed lines are deletions
  - `+` prefixed lines are additions
- Wrap the entire patch in `*** Begin Patch` / `*** End Patch`

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
        return None


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
