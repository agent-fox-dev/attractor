"""ExecutionEnvironment interface per Section 4 of the coding-agent-loop-spec."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: int


@dataclass
class DirEntry:
    name: str
    is_dir: bool
    size: int | None = None


class ExecutionEnvironment(ABC):
    """Abstract interface for tool execution environments.

    All tool operations pass through this interface, decoupling tool logic
    from where it runs.  The default implementation is
    ``LocalExecutionEnvironment``; alternatives can target Docker,
    Kubernetes, WASM, SSH, etc.
    """

    @abstractmethod
    def read_file(
        self, path: str, offset: int | None = None, limit: int | None = None
    ) -> str:
        ...

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        ...

    @abstractmethod
    def file_exists(self, path: str) -> bool:
        ...

    @abstractmethod
    def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]:
        ...

    @abstractmethod
    def exec_command(
        self,
        command: str,
        timeout_ms: int = 10000,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult:
        ...

    @abstractmethod
    def grep(self, pattern: str, path: str, **options) -> str:
        ...

    @abstractmethod
    def glob(self, pattern: str, path: str) -> list[str]:
        ...

    @abstractmethod
    def working_directory(self) -> str:
        ...

    @abstractmethod
    def platform(self) -> str:
        ...

    def os_version(self) -> str:
        """Return the OS version string.  Default uses ``platform.platform()``."""
        import platform as _platform

        return _platform.platform()

    def is_git_repo(self) -> bool:
        """Return True if the working directory is inside a git repository."""
        return False

    def git_context(self) -> str:
        """Return a short git context string (branch, status, recent commits)."""
        return ""

    def initialize(self) -> None:
        """Called once before the session begins.  Override to set up resources."""
        pass

    def cleanup(self) -> None:
        """Called once when the session ends.  Override to tear down resources."""
        pass
