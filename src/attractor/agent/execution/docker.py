"""DockerExecutionEnvironment per Section 4.3 of the coding-agent-loop-spec.

Runs commands and file operations inside a Docker container, providing
sandboxed execution. Requires Docker CLI (``docker``) to be available
on the host.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

from attractor.agent.execution.base import DirEntry, ExecResult, ExecutionEnvironment


class DockerExecutionEnvironment(ExecutionEnvironment):
    """Execution environment that runs inside a Docker container.

    Parameters
    ----------
    image:
        Docker image to use (e.g. ``python:3.12-slim``).
    working_dir:
        Working directory inside the container.
    mount_dir:
        Host directory to mount into the container at *working_dir*.
        If ``None``, no volume mount is used (container-only filesystem).
    container_name:
        Optional container name. Auto-generated if not provided.
    extra_args:
        Additional arguments to pass to ``docker run``.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        working_dir: str = "/workspace",
        mount_dir: str | None = None,
        container_name: str | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self._image = image
        self._working_dir = working_dir
        self._mount_dir = mount_dir
        self._container_name = container_name or f"attractor-{uuid.uuid4().hex[:8]}"
        self._extra_args = extra_args or []
        self._container_id: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        """Start the Docker container."""
        cmd = [
            "docker", "run", "-d",
            "--name", self._container_name,
            "-w", self._working_dir,
        ]
        if self._mount_dir:
            cmd.extend(["-v", f"{self._mount_dir}:{self._working_dir}"])
        cmd.extend(self._extra_args)
        cmd.extend([self._image, "sleep", "infinity"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start Docker container: {result.stderr}")
        self._container_id = result.stdout.strip()

    def cleanup(self) -> None:
        """Stop and remove the container."""
        if self._container_id or self._container_name:
            name = self._container_id or self._container_name
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True, timeout=30,
            )
            self._container_id = None

    # -- helpers -----------------------------------------------------------

    def _docker_exec(
        self, command: str, timeout_ms: int = 10000
    ) -> ExecResult:
        """Run a command inside the container."""
        name = self._container_id or self._container_name
        cmd = ["docker", "exec", "-w", self._working_dir, name, "sh", "-c", command]
        timeout_s = timeout_ms / 1000.0
        timed_out = False
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
                stderr=f"Command timed out after {timeout_ms}ms",
                exit_code=-1,
                timed_out=True,
                duration_ms=elapsed,
            )

    # -- file operations ---------------------------------------------------

    def read_file(
        self, path: str, offset: int | None = None, limit: int | None = None
    ) -> str:
        result = self._docker_exec(f"cat {shlex.quote(path)}", timeout_ms=5000)
        if result.exit_code != 0:
            raise FileNotFoundError(f"File not found in container: {path}")

        lines = result.stdout.splitlines(keepends=True)
        start = (offset - 1) if offset and offset >= 1 else 0
        end = (start + limit) if limit else len(lines)
        selected = lines[start:end]

        numbered: list[str] = []
        for i, line in enumerate(selected, start=start + 1):
            numbered.append(f"{i:4d} | {line.rstrip()}")
        return "\n".join(numbered)

    def write_file(self, path: str, content: str) -> None:
        # Ensure parent directory exists
        parent = str(Path(path).parent)
        self._docker_exec(f"mkdir -p {shlex.quote(parent)}")
        # Write via stdin
        name = self._container_id or self._container_name
        cmd = ["docker", "exec", "-i", "-w", self._working_dir, name, "sh", "-c",
               f"cat > {shlex.quote(path)}"]
        subprocess.run(cmd, input=content, capture_output=True, text=True, timeout=10)

    def file_exists(self, path: str) -> bool:
        result = self._docker_exec(f"test -e {shlex.quote(path)} && echo yes", timeout_ms=3000)
        return "yes" in result.stdout

    def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]:
        result = self._docker_exec(
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
        name = self._container_id or self._container_name
        cmd = ["docker", "exec"]
        if working_dir:
            cmd.extend(["-w", working_dir])
        else:
            cmd.extend(["-w", self._working_dir])
        if env_vars:
            for k, v in env_vars.items():
                cmd.extend(["-e", f"{k}={v}"])
        cmd.extend([name, "sh", "-c", command])

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
                stderr=f"Command timed out after {timeout_ms}ms",
                exit_code=-1,
                timed_out=True,
                duration_ms=elapsed,
            )

    # -- search operations -------------------------------------------------

    def grep(self, pattern: str, path: str, **options) -> str:
        case_flag = "-i" if options.get("case_insensitive") else ""
        max_results = options.get("max_results", 100)
        cmd = f"grep -rn {case_flag} -m {max_results} {shlex.quote(pattern)} {shlex.quote(path)}"
        result = self._docker_exec(cmd, timeout_ms=10000)
        return result.stdout

    def glob(self, pattern: str, path: str) -> list[str]:
        result = self._docker_exec(
            f"find {shlex.quote(path)} -name {shlex.quote(pattern)} -type f",
            timeout_ms=10000,
        )
        return [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]

    # -- metadata ----------------------------------------------------------

    def working_directory(self) -> str:
        return self._working_dir

    def platform(self) -> str:
        result = self._docker_exec("uname -s", timeout_ms=3000)
        plat = result.stdout.strip().lower()
        if "linux" in plat:
            return "linux"
        return plat or "linux"

    def os_version(self) -> str:
        result = self._docker_exec("uname -a", timeout_ms=3000)
        return result.stdout.strip()
