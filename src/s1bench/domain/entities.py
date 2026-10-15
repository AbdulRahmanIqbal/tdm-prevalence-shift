"""Core entities. Pure Python: no I/O, no third-party imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SourceKind(str, Enum):
    HF_SNAPSHOT = "hf_snapshot"
    URL = "url"


@dataclass(frozen=True)
class SourceFile:
    """One remote file to fetch, and where it goes inside the dataset's raw folder."""

    url: str
    target: str


@dataclass(frozen=True)
class DatasetSource:
    """Where a dataset's raw files come from. Always pinned (commit / fixed URL)."""

    dataset_id: str
    kind: SourceKind
    license: str
    homepage: str = ""
    notes: str = ""
    # hf_snapshot
    repo_id: str = ""
    revision: str = ""
    allow_patterns: tuple[str, ...] = ()
    # url
    files: tuple[SourceFile, ...] = ()

    def origin(self) -> str:
        """Stable description of the source; a change means the raw data must be refetched."""
        if self.kind is SourceKind.HF_SNAPSHOT:
            patterns = ",".join(self.allow_patterns)
            return f"hf://datasets/{self.repo_id}@{self.revision}[{patterns}]"
        return ";".join(f"{f.url}->{f.target}" for f in self.files)


@dataclass(frozen=True)
class RawFile:
    """A downloaded file. `path` is relative to the dataset's raw folder (POSIX style)."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RawDatasetRecord:
    """Manifest entry: exactly what was downloaded, from where, and when."""

    dataset_id: str
    kind: SourceKind
    origin: str
    license: str
    downloaded_at: str
    files: tuple[RawFile, ...] = field(default_factory=tuple)

    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)
