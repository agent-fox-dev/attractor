"""Tool output truncation per Section 5 of the coding-agent-loop-spec."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Default limits (Section 5.2)
# ---------------------------------------------------------------------------

DEFAULT_CHAR_LIMITS: dict[str, int] = {
    "read_file": 50_000,
    "shell": 30_000,
    "grep": 20_000,
    "glob": 20_000,
    "edit_file": 10_000,
    "apply_patch": 10_000,
    "write_file": 1_000,
    "spawn_agent": 20_000,
}

DEFAULT_TRUNCATION_MODES: dict[str, str] = {
    "read_file": "head_tail",
    "shell": "head_tail",
    "grep": "tail",
    "glob": "tail",
    "edit_file": "tail",
    "apply_patch": "tail",
    "write_file": "tail",
    "spawn_agent": "head_tail",
}

DEFAULT_LINE_LIMITS: dict[str, int | None] = {
    "shell": 256,
    "grep": 200,
    "glob": 500,
    "read_file": None,
    "edit_file": None,
    "apply_patch": None,
    "write_file": None,
    "spawn_agent": None,
}

# Fallback for unknown tools
_FALLBACK_CHAR_LIMIT = 30_000
_FALLBACK_TRUNCATION_MODE = "head_tail"
_FALLBACK_LINE_LIMIT: int | None = None


# ---------------------------------------------------------------------------
# Character-based truncation (Section 5.1)
# ---------------------------------------------------------------------------


def truncate_output(output: str, max_chars: int, mode: str = "head_tail") -> str:
    """Truncate *output* to *max_chars* using the specified mode.

    Modes:
        ``head_tail`` -- keep the first half and last half, removing the middle.
        ``tail``      -- keep only the last *max_chars* characters.
    """
    if len(output) <= max_chars:
        return output

    if mode == "head_tail":
        half = max_chars // 2
        removed = len(output) - max_chars
        return (
            output[:half]
            + f"\n\n[WARNING: Tool output was truncated. "
            f"{removed} characters were removed from the middle. "
            f"The full output is available in the event stream. "
            f"If you need to see specific parts, re-run the tool "
            f"with more targeted parameters.]\n\n"
            + output[-half:]
        )

    if mode == "tail":
        removed = len(output) - max_chars
        return (
            f"[WARNING: Tool output was truncated. First "
            f"{removed} characters were removed. "
            f"The full output is available in the event stream.]\n\n"
            + output[-max_chars:]
        )

    # Unknown mode, fall back to head_tail
    return truncate_output(output, max_chars, "head_tail")


# ---------------------------------------------------------------------------
# Line-based truncation (Section 5.3)
# ---------------------------------------------------------------------------


def truncate_lines(output: str, max_lines: int) -> str:
    """Truncate *output* to at most *max_lines* using a head/tail split."""
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output

    head_count = max_lines // 2
    tail_count = max_lines - head_count
    omitted = len(lines) - head_count - tail_count

    return (
        "\n".join(lines[:head_count])
        + f"\n[... {omitted} lines omitted ...]\n"
        + "\n".join(lines[-tail_count:])
    )


# ---------------------------------------------------------------------------
# Combined pipeline (Section 5.3)
# ---------------------------------------------------------------------------


def truncate_tool_output(output: str, tool_name: str, config) -> str:
    """Apply the full truncation pipeline for a tool output.

    Parameters
    ----------
    output:
        Raw tool output string.
    tool_name:
        Name of the tool that produced the output.
    config:
        A ``SessionConfig`` (or any object with ``tool_output_limits``
        and optionally ``tool_line_limits`` attributes).
    """
    # Resolve character limit
    tool_output_limits: dict[str, int] = getattr(config, "tool_output_limits", {})
    max_chars = tool_output_limits.get(
        tool_name, DEFAULT_CHAR_LIMITS.get(tool_name, _FALLBACK_CHAR_LIMIT)
    )

    # Step 1: Character-based truncation (always runs first)
    mode = DEFAULT_TRUNCATION_MODES.get(tool_name, _FALLBACK_TRUNCATION_MODE)
    result = truncate_output(output, max_chars, mode)

    # Step 2: Line-based truncation (secondary, for readability)
    tool_line_limits: dict[str, int] = getattr(config, "tool_line_limits", {})
    max_lines_val = tool_line_limits.get(
        tool_name, DEFAULT_LINE_LIMITS.get(tool_name, _FALLBACK_LINE_LIMIT)
    )
    if max_lines_val is not None:
        result = truncate_lines(result, max_lines_val)

    return result
