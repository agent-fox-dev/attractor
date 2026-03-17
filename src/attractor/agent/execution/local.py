"""LocalExecutionEnvironment per Section 4.2 of the coding-agent-loop-spec.

Runs everything on the local machine with:
- File operations via pathlib
- Command execution via subprocess with process-group isolation
- Environment variable filtering (exclude secrets by default)
- Grep via ripgrep (rg) with Python re fallback
- Glob via pathlib sorted by mtime
"""

from __future__ import annotations

import fnmatch
import os
import platform as _platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from attractor.agent.execution.base import DirEntry, ExecResult, ExecutionEnvironment

# ---------------------------------------------------------------------------
# Environment variable filtering
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS: list[str] = [
    "*_API_KEY",
    "*_SECRET",
    "*_TOKEN",
    "*_PASSWORD",
    "*_CREDENTIAL",
]

_ALWAYS_INCLUDE: set[str] = {
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "LANG",
    "TERM",
    "TMPDIR",
    # Language-specific paths
    "GOPATH",
    "GOROOT",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "NVM_DIR",
    "PYENV_ROOT",
    "VIRTUAL_ENV",
    "CONDA_DEFAULT_ENV",
    "JAVA_HOME",
    "ANDROID_HOME",
    "NODE_PATH",
    "RBENV_ROOT",
    "GEM_HOME",
}


def _is_sensitive(name: str) -> bool:
    upper = name.upper()
    return any(fnmatch.fnmatch(upper, pat) for pat in _SENSITIVE_PATTERNS)


def _filter_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a filtered copy of the environment, excluding sensitive variables."""
    source = base_env if base_env is not None else os.environ
    filtered: dict[str, str] = {}
    for key, value in source.items():
        if key in _ALWAYS_INCLUDE or not _is_sensitive(key):
            filtered[key] = value
    return filtered


# ---------------------------------------------------------------------------
# LocalExecutionEnvironment
# ---------------------------------------------------------------------------


class LocalExecutionEnvironment(ExecutionEnvironment):
    """Default execution environment that runs everything locally."""

    def __init__(self, working_dir: str | None = None) -> None:
        self._working_dir = working_dir or os.getcwd()

    # -- helpers -----------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """Resolve *path* relative to the working directory."""
        p = Path(path)
        if not p.is_absolute():
            p = Path(self._working_dir) / p
        return p

    # -- file operations ---------------------------------------------------

    def read_file(
        self, path: str, offset: int | None = None, limit: int | None = None
    ) -> str:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not resolved.is_file():
            raise IsADirectoryError(f"Not a file: {path}")

        text = resolved.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)

        start = (offset - 1) if offset and offset >= 1 else 0
        end = (start + limit) if limit else len(lines)
        selected = lines[start:end]

        # Format with line numbers
        numbered: list[str] = []
        for i, line in enumerate(selected, start=start + 1):
            numbered.append(f"{i:4d} | {line.rstrip()}")
        return "\n".join(numbered)

    def write_file(self, path: str, content: str) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

    def file_exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]:
        resolved = self._resolve(path)
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        entries: list[DirEntry] = []
        self._walk_directory(resolved, entries, depth, current_depth=0)
        return entries

    def _walk_directory(
        self, directory: Path, entries: list[DirEntry], max_depth: int, current_depth: int
    ) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda p: p.name)
        except PermissionError:
            return
        for child in children:
            is_dir = child.is_dir()
            size = child.stat().st_size if child.is_file() else None
            entries.append(DirEntry(name=str(child), is_dir=is_dir, size=size))
            if is_dir and current_depth + 1 < max_depth:
                self._walk_directory(child, entries, max_depth, current_depth + 1)

    # -- command execution -------------------------------------------------

    def exec_command(
        self,
        command: str,
        timeout_ms: int = 10000,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult:
        cwd = working_dir or self._working_dir
        env = _filter_env()
        if env_vars:
            env.update(env_vars)

        timeout_s = timeout_ms / 1000.0
        timed_out = False
        start = time.monotonic()

        # Choose shell
        if sys.platform == "win32":
            shell_cmd = ["cmd.exe", "/c", command]
            kwargs: dict = {}
        else:
            shell_cmd = ["/bin/bash", "-c", command]
            kwargs = {"start_new_session": True}

        try:
            proc = subprocess.Popen(
                shell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
                **kwargs,
            )
        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            return ExecResult(
                stdout="",
                stderr=str(exc),
                exit_code=1,
                timed_out=False,
                duration_ms=elapsed,
            )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            # SIGTERM to process group
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except (ProcessLookupError, OSError):
                pass

            # Wait 2s for graceful shutdown
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                # SIGKILL
                try:
                    if sys.platform != "win32":
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
                except (ProcessLookupError, OSError):
                    pass
                stdout_bytes, stderr_bytes = proc.communicate()

        elapsed = int((time.monotonic() - start) * 1000)
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if timed_out:
            stderr += (
                f"\n[ERROR: Command timed out after {timeout_ms}ms. "
                f"Partial output is shown above. "
                f"You can retry with a longer timeout by setting the "
                f"timeout_ms parameter.]"
            )

        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode if proc.returncode is not None else -1,
            timed_out=timed_out,
            duration_ms=elapsed,
        )

    # -- search operations -------------------------------------------------

    def grep(self, pattern: str, path: str, **options) -> str:
        """Search file contents using regex.

        Tries ripgrep (rg) first; falls back to Python re if rg is not
        available.
        """
        resolved = self._resolve(path)
        case_insensitive: bool = options.get("case_insensitive", False)
        glob_filter: str | None = options.get("glob_filter")
        max_results: int = options.get("max_results", 100)

        # Try ripgrep first
        try:
            return self._grep_rg(
                pattern, str(resolved), case_insensitive, glob_filter, max_results
            )
        except FileNotFoundError:
            pass  # rg not installed, fall back to Python

        return self._grep_python(
            pattern, resolved, case_insensitive, glob_filter, max_results
        )

    def _grep_rg(
        self,
        pattern: str,
        path: str,
        case_insensitive: bool,
        glob_filter: str | None,
        max_results: int,
    ) -> str:
        cmd = ["rg", "--line-number", "--no-heading", f"--max-count={max_results}"]
        if case_insensitive:
            cmd.append("-i")
        if glob_filter:
            cmd.extend(["--glob", glob_filter])
        cmd.extend([pattern, path])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 2:
            raise RuntimeError(f"Grep error: {result.stderr.strip()}")
        return result.stdout

    def _grep_python(
        self,
        pattern: str,
        search_path: Path,
        case_insensitive: bool,
        glob_filter: str | None,
        max_results: int,
    ) -> str:
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {exc}") from exc

        matches: list[str] = []
        files: list[Path] = []

        if search_path.is_file():
            files = [search_path]
        elif search_path.is_dir():
            file_pattern = glob_filter or "**/*"
            files = [
                f for f in search_path.glob(file_pattern) if f.is_file()
            ]
        else:
            raise FileNotFoundError(f"Path not found: {search_path}")

        for fpath in files:
            if len(matches) >= max_results:
                break
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except (PermissionError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    matches.append(f"{fpath}:{lineno}:{line}")
                    if len(matches) >= max_results:
                        break

        return "\n".join(matches)

    def glob(self, pattern: str, path: str) -> list[str]:
        """Find files matching a glob pattern, sorted by mtime (newest first)."""
        resolved = self._resolve(path)
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        results = [str(p) for p in resolved.glob(pattern) if p.is_file()]
        # Sort by mtime descending (newest first)
        results.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return results

    # -- metadata ----------------------------------------------------------

    def working_directory(self) -> str:
        return self._working_dir

    def platform(self) -> str:
        plat = sys.platform
        if plat == "darwin":
            return "darwin"
        if plat.startswith("linux"):
            return "linux"
        if plat == "win32":
            return "windows"
        return plat

    def os_version(self) -> str:
        return _platform.platform()

    def is_git_repo(self) -> bool:
        result = self.exec_command("git rev-parse --git-dir", timeout_ms=3000)
        return result.exit_code == 0

    def git_context(self) -> str:
        """Return branch, short status, and recent commits."""
        if not self.is_git_repo():
            return ""

        parts: list[str] = []
        # Branch
        result = self.exec_command(
            "git branch --show-current", timeout_ms=3000
        )
        if result.exit_code == 0 and result.stdout.strip():
            parts.append(f"Branch: {result.stdout.strip()}")

        # Short status (file counts)
        result = self.exec_command(
            "git status --porcelain", timeout_ms=3000
        )
        if result.exit_code == 0:
            lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
            modified = sum(1 for l in lines if l.startswith(" M") or l.startswith("M"))
            untracked = sum(1 for l in lines if l.startswith("??"))
            if modified or untracked:
                parts.append(f"Modified: {modified}, Untracked: {untracked}")
            else:
                parts.append("Working tree: clean")

        # Recent commits
        result = self.exec_command(
            'git log --oneline -5 --no-decorate', timeout_ms=3000
        )
        if result.exit_code == 0 and result.stdout.strip():
            parts.append(f"Recent commits:\n{result.stdout.strip()}")

        return "\n".join(parts)
