"""SSHExecutionEnvironment per Section 4.4 of the coding-agent-loop-spec.

Runs commands and file operations on a remote host via SSH/SCP.
Requires the ``ssh`` and ``scp`` CLIs to be available on the host.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from attractor.agent.execution.base import DirEntry, ExecResult, ExecutionEnvironment


class SSHExecutionEnvironment(ExecutionEnvironment):
    """Execution environment that runs on a remote host via SSH.

    Parameters
    ----------
    host:
        Remote hostname or IP (e.g. ``user@host``).
    working_dir:
        Working directory on the remote host.
    port:
        SSH port (default 22).
    identity_file:
        Path to SSH private key file. If ``None``, uses default.
    ssh_options:
        Extra options passed to ``ssh -o``.
    """

    def __init__(
        self,
        host: str,
        working_dir: str = "/home",
        port: int = 22,
        identity_file: str | None = None,
        ssh_options: list[str] | None = None,
    ) -> None:
        self._host = host
        self._working_dir = working_dir
        self._port = port
        self._identity_file = identity_file
        self._ssh_options = ssh_options or []

    def _ssh_base(self) -> list[str]:
        cmd = ["ssh", "-p", str(self._port)]
        if self._identity_file:
            cmd.extend(["-i", self._identity_file])
        for opt in self._ssh_options:
            cmd.extend(["-o", opt])
        cmd.append(self._host)
        return cmd

    def _scp_base(self) -> list[str]:
        cmd = ["scp", "-P", str(self._port)]
        if self._identity_file:
            cmd.extend(["-i", self._identity_file])
        for opt in self._ssh_options:
            cmd.extend(["-o", opt])
        return cmd

    def _ssh_exec(self, command: str, timeout_ms: int = 10000) -> ExecResult:
        cmd = self._ssh_base() + [command]
        timeout_s = timeout_ms / 1000.0
        start = time.monotonic()

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s,
            )
            elapsed = int((time.monotonic() - start) * 1000)
            return ExecResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                timed_out=False,
                duration_ms=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = int((time.monotonic() - start) * 1000)
            return ExecResult(
                stdout="",
                stderr=(
                    f"\n[ERROR: Command timed out after {timeout_ms}ms. "
                    f"Partial output is shown above. "
                    f"You can retry with a longer timeout by setting the "
                    f"timeout_ms parameter.]"
                ),
                exit_code=-1,
                timed_out=True,
                duration_ms=elapsed,
            )

    # -- file operations ---------------------------------------------------

    def read_file(
        self, path: str, offset: int | None = None, limit: int | None = None
    ) -> str:
        result = self._ssh_exec(f"cat {shlex.quote(path)}", timeout_ms=5000)
        if result.exit_code != 0:
            raise FileNotFoundError(f"File not found on remote: {path}")

        lines = result.stdout.splitlines(keepends=True)
        start_line = (offset - 1) if offset and offset >= 1 else 0
        end = (start_line + limit) if limit else len(lines)
        selected = lines[start_line:end]

        numbered: list[str] = []
        for i, line in enumerate(selected, start=start_line + 1):
            numbered.append(f"{i:4d} | {line.rstrip()}")
        return "\n".join(numbered)

    def write_file(self, path: str, content: str) -> None:
        # Ensure parent directory exists
        parent = str(Path(path).parent)
        self._ssh_exec(f"mkdir -p {shlex.quote(parent)}")
        # Write via stdin over ssh
        cmd = self._ssh_base() + [f"cat > {shlex.quote(path)}"]
        subprocess.run(cmd, input=content, capture_output=True, text=True, timeout=10)

    def file_exists(self, path: str) -> bool:
        result = self._ssh_exec(f"test -e {shlex.quote(path)} && echo yes", timeout_ms=3000)
        return "yes" in result.stdout

    def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]:
        result = self._ssh_exec(
            f"find {shlex.quote(path)} -maxdepth {depth} -printf '%y %s %p\\n'",
            timeout_ms=5000,
        )
        entries: list[DirEntry] = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            ftype, size_str, name = parts
            is_dir = ftype == "d"
            size = int(size_str) if not is_dir else None
            entries.append(DirEntry(name=name, is_dir=is_dir, size=size))
        return entries

    # -- command execution -------------------------------------------------

    def exec_command(
        self,
        command: str,
        timeout_ms: int = 10000,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult:
        cwd = working_dir or self._working_dir
        prefix = f"cd {shlex.quote(cwd)} && "
        if env_vars:
            env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_vars.items())
            prefix += f"export {env_str} && "
        return self._ssh_exec(prefix + command, timeout_ms=timeout_ms)

    # -- search operations -------------------------------------------------

    def grep(self, pattern: str, path: str, **options) -> str:
        case_flag = "-i" if options.get("case_insensitive") else ""
        max_results = options.get("max_results", 100)
        cmd = f"grep -rn {case_flag} -m {max_results} {shlex.quote(pattern)} {shlex.quote(path)}"
        result = self._ssh_exec(cmd, timeout_ms=10000)
        return result.stdout

    def glob(self, pattern: str, path: str) -> list[str]:
        result = self._ssh_exec(
            f"find {shlex.quote(path)} -name {shlex.quote(pattern)} -type f",
            timeout_ms=10000,
        )
        return [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]

    # -- metadata ----------------------------------------------------------

    def working_directory(self) -> str:
        return self._working_dir

    def platform(self) -> str:
        result = self._ssh_exec("uname -s", timeout_ms=3000)
        plat = result.stdout.strip().lower()
        if "linux" in plat:
            return "linux"
        if "darwin" in plat:
            return "darwin"
        return plat or "linux"

    def os_version(self) -> str:
        result = self._ssh_exec("uname -a", timeout_ms=3000)
        return result.stdout.strip()

    def is_git_repo(self) -> bool:
        result = self.exec_command("git rev-parse --git-dir", timeout_ms=3000)
        return result.exit_code == 0

    def git_branch(self) -> str:
        result = self.exec_command("git branch --show-current", timeout_ms=3000)
        return result.stdout.strip() if result.exit_code == 0 else ""

    def git_context(self) -> str:
        if not self.is_git_repo():
            return ""
        parts: list[str] = []
        result = self.exec_command("git branch --show-current", timeout_ms=3000)
        if result.exit_code == 0 and result.stdout.strip():
            parts.append(f"Branch: {result.stdout.strip()}")
        result = self.exec_command("git status --porcelain", timeout_ms=3000)
        if result.exit_code == 0:
            lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
            modified = sum(1 for l in lines if l.startswith(" M") or l.startswith("M"))
            untracked = sum(1 for l in lines if l.startswith("??"))
            if modified or untracked:
                parts.append(f"Modified: {modified}, Untracked: {untracked}")
            else:
                parts.append("Working tree: clean")
        result = self.exec_command("git log --oneline -5 --no-decorate", timeout_ms=3000)
        if result.exit_code == 0 and result.stdout.strip():
            parts.append(f"Recent commits:\n{result.stdout.strip()}")
        return "\n".join(parts)
