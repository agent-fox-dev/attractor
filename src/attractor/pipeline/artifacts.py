"""ArtifactStore per Section 5.5 of the Attractor spec.

Stores named binary artifacts in memory, spilling to disk when they exceed
the file-backing threshold (100 KB by default).
"""

from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DEFAULT_THRESHOLD = 100 * 1024  # 100 KB


@dataclass
class ArtifactInfo:
    """Metadata about a stored artifact."""
    id: str
    name: str
    size_bytes: int
    stored_at: float
    is_file_backed: bool


class ArtifactStore:
    """Named artifact storage with automatic file-backing for large items.

    Small artifacts (< *threshold* bytes) are held in memory.
    Larger artifacts are written to a temporary directory on disk.
    """

    def __init__(
        self,
        threshold: int = _DEFAULT_THRESHOLD,
        storage_dir: Path | None = None,
    ) -> None:
        self._threshold = threshold
        self._storage_dir = storage_dir or Path(tempfile.mkdtemp(prefix="attractor_artifacts_"))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, bytes] = {}
        self._info: dict[str, ArtifactInfo] = {}

    # ---- public API ----

    def store(self, name: str, data: bytes) -> ArtifactInfo:
        """Store an artifact under *name*, returning its metadata."""
        artifact_id = uuid.uuid4().hex[:12]
        size = len(data)
        is_file_backed = size >= self._threshold

        if is_file_backed:
            path = self._file_path(artifact_id)
            path.write_bytes(data)
        else:
            self._memory[artifact_id] = data

        info = ArtifactInfo(
            id=artifact_id,
            name=name,
            size_bytes=size,
            stored_at=time.time(),
            is_file_backed=is_file_backed,
        )
        self._info[artifact_id] = info
        return info

    def retrieve(self, artifact_id: str) -> bytes:
        """Retrieve artifact data by id. Raises KeyError if not found."""
        info = self._info.get(artifact_id)
        if info is None:
            raise KeyError(f"Artifact '{artifact_id}' not found.")

        if info.is_file_backed:
            path = self._file_path(artifact_id)
            return path.read_bytes()
        return self._memory[artifact_id]

    def has(self, artifact_id: str) -> bool:
        return artifact_id in self._info

    def list(self) -> list[ArtifactInfo]:
        """Return metadata for all stored artifacts."""
        return list(self._info.values())

    def remove(self, artifact_id: str) -> None:
        """Remove a single artifact."""
        info = self._info.pop(artifact_id, None)
        if info is None:
            return
        if info.is_file_backed:
            path = self._file_path(artifact_id)
            path.unlink(missing_ok=True)
        else:
            self._memory.pop(artifact_id, None)

    def clear(self) -> None:
        """Remove all artifacts."""
        for aid in list(self._info.keys()):
            self.remove(aid)
        self._info.clear()
        self._memory.clear()

    # ---- internal ----

    def _file_path(self, artifact_id: str) -> Path:
        return self._storage_dir / artifact_id
