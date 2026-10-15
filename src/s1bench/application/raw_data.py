"""Use cases for raw data: download (idempotent) and integrity verification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from s1bench.domain.entities import DatasetSource, RawDatasetRecord, SourceKind
from s1bench.domain.ports import Clock, FileHasher, ManifestStore, RawFetcher


class DownloadStatus(str, Enum):
    DOWNLOADED = "downloaded"
    UP_TO_DATE = "up-to-date"


@dataclass(frozen=True)
class DownloadResult:
    dataset_id: str
    status: DownloadStatus
    record: RawDatasetRecord


class DownloadRawDatasets:
    """Fetch raw datasets into raw_root/<id>/ and record them in the manifest.

    A dataset is skipped when the manifest already has the same origin and every
    recorded file is still present with the recorded hash (unless force=True).
    """

    def __init__(
        self,
        fetchers: Mapping[SourceKind, RawFetcher],
        manifest: ManifestStore,
        hasher: FileHasher,
        clock: Clock,
        raw_root: Path,
    ) -> None:
        self._fetchers = fetchers
        self._manifest = manifest
        self._hasher = hasher
        self._clock = clock
        self._raw_root = raw_root

    def execute(self, sources: Iterable[DatasetSource], force: bool = False) -> list[DownloadResult]:
        records = self._manifest.load()
        results: list[DownloadResult] = []
        for source in sources:
            existing = records.get(source.dataset_id)
            if not force and existing is not None and self._is_intact(source, existing):
                results.append(DownloadResult(source.dataset_id, DownloadStatus.UP_TO_DATE, existing))
                continue

            dest = self._raw_root / source.dataset_id
            files = self._fetchers[source.kind].fetch(source, dest)
            record = RawDatasetRecord(
                dataset_id=source.dataset_id,
                kind=source.kind,
                origin=source.origin(),
                license=source.license,
                downloaded_at=self._clock.now_iso(),
                files=tuple(sorted(files, key=lambda f: f.path)),
            )
            records[source.dataset_id] = record
            # Save after every dataset so a later failure keeps earlier work.
            self._manifest.save(records)
            results.append(DownloadResult(source.dataset_id, DownloadStatus.DOWNLOADED, record))
        return results

    def _is_intact(self, source: DatasetSource, record: RawDatasetRecord) -> bool:
        if record.origin != source.origin() or not record.files:
            return False
        dest = self._raw_root / source.dataset_id
        return all(
            (dest / f.path).is_file() and self._hasher.sha256(dest / f.path) == f.sha256
            for f in record.files
        )


@dataclass(frozen=True)
class FileProblem:
    dataset_id: str
    path: str
    problem: str  # "missing" | "hash mismatch"


class VerifyRawDatasets:
    """Re-hash every file in the manifest. data/raw must never change after download."""

    def __init__(self, manifest: ManifestStore, hasher: FileHasher, raw_root: Path) -> None:
        self._manifest = manifest
        self._hasher = hasher
        self._raw_root = raw_root

    def execute(self) -> list[FileProblem]:
        problems: list[FileProblem] = []
        for dataset_id, record in sorted(self._manifest.load().items()):
            for f in record.files:
                path = self._raw_root / dataset_id / f.path
                if not path.is_file():
                    problems.append(FileProblem(dataset_id, f.path, "missing"))
                elif self._hasher.sha256(path) != f.sha256:
                    problems.append(FileProblem(dataset_id, f.path, "hash mismatch"))
        return problems
