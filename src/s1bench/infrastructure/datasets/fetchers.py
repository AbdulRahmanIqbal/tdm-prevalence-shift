"""RawFetcher implementations: pinned Hugging Face snapshots and plain URLs."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from s1bench.domain.entities import DatasetSource, RawFile, SourceKind
from s1bench.domain.ports import FileHasher

_USER_AGENT = "s1bench/0.1 (research benchmark; raw data download)"


def _collect(dest_dir: Path, hasher: FileHasher) -> list[RawFile]:
    files: list[RawFile] = []
    for path in sorted(dest_dir.rglob("*")):
        rel = path.relative_to(dest_dir)
        if not path.is_file() or rel.parts[0] == ".cache":
            continue
        files.append(RawFile(path=rel.as_posix(), sha256=hasher.sha256(path), size_bytes=path.stat().st_size))
    return files


def _reset_dir(dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)


class HfSnapshotFetcher:
    def __init__(self, hasher: FileHasher) -> None:
        self._hasher = hasher

    def fetch(self, source: DatasetSource, dest_dir: Path) -> list[RawFile]:
        assert source.kind is SourceKind.HF_SNAPSHOT
        from huggingface_hub import snapshot_download  # imported lazily: optional heavy dependency

        _reset_dir(dest_dir)
        snapshot_download(
            repo_id=source.repo_id,
            repo_type="dataset",
            revision=source.revision,
            allow_patterns=list(source.allow_patterns) or None,
            local_dir=dest_dir,
        )
        # huggingface_hub keeps download metadata in <local_dir>/.cache; it is not raw data.
        shutil.rmtree(dest_dir / ".cache", ignore_errors=True)
        return _collect(dest_dir, self._hasher)


class UrlFetcher:
    def __init__(self, hasher: FileHasher, timeout_s: float = 120.0) -> None:
        self._hasher = hasher
        self._timeout_s = timeout_s

    def fetch(self, source: DatasetSource, dest_dir: Path) -> list[RawFile]:
        assert source.kind is SourceKind.URL
        _reset_dir(dest_dir)
        for f in source.files:
            target = dest_dir / f.target
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            request = urllib.request.Request(f.url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
            tmp.replace(target)
        return _collect(dest_dir, self._hasher)
