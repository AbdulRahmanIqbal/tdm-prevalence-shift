from __future__ import annotations

import hashlib
from pathlib import Path

from s1bench.application.raw_data import DownloadRawDatasets, DownloadStatus, VerifyRawDatasets
from s1bench.domain.entities import DatasetSource, RawDatasetRecord, RawFile, SourceFile, SourceKind


class FakeHasher:
    def sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeClock:
    def now_iso(self) -> str:
        return "2026-10-06T00:00:00+00:00"


class MemoryManifest:
    def __init__(self) -> None:
        self.records: dict[str, RawDatasetRecord] = {}

    def load(self) -> dict[str, RawDatasetRecord]:
        return dict(self.records)

    def save(self, records: dict[str, RawDatasetRecord]) -> None:
        self.records = dict(records)


class FakeFetcher:
    def __init__(self) -> None:
        self.calls = 0

    def fetch(self, source: DatasetSource, dest_dir: Path) -> list[RawFile]:
        self.calls += 1
        dest_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for f in source.files:
            path = dest_dir / f.target
            path.write_text(f"content of {f.url}", encoding="utf-8")
            files.append(RawFile(f.target, FakeHasher().sha256(path), path.stat().st_size))
        return files


def _source(url: str = "https://example.org/a.txt") -> DatasetSource:
    return DatasetSource(
        dataset_id="toy",
        kind=SourceKind.URL,
        license="CC-BY-4.0",
        files=(SourceFile(url=url, target="a.txt"),),
    )


def _use_case(tmp_path: Path, fetcher: FakeFetcher, manifest: MemoryManifest) -> DownloadRawDatasets:
    return DownloadRawDatasets(
        fetchers={SourceKind.URL: fetcher},
        manifest=manifest,
        hasher=FakeHasher(),
        clock=FakeClock(),
        raw_root=tmp_path / "raw",
    )


def test_download_then_skip_when_intact(tmp_path: Path) -> None:
    fetcher, manifest = FakeFetcher(), MemoryManifest()
    use_case = _use_case(tmp_path, fetcher, manifest)

    first = use_case.execute([_source()])
    second = use_case.execute([_source()])

    assert first[0].status is DownloadStatus.DOWNLOADED
    assert second[0].status is DownloadStatus.UP_TO_DATE
    assert fetcher.calls == 1
    assert manifest.records["toy"].files[0].path == "a.txt"


def test_redownload_when_origin_changes_or_file_tampered(tmp_path: Path) -> None:
    fetcher, manifest = FakeFetcher(), MemoryManifest()
    use_case = _use_case(tmp_path, fetcher, manifest)
    use_case.execute([_source()])

    (tmp_path / "raw" / "toy" / "a.txt").write_text("edited by hand", encoding="utf-8")
    assert use_case.execute([_source()])[0].status is DownloadStatus.DOWNLOADED

    assert use_case.execute([_source(url="https://example.org/b.txt")])[0].status is DownloadStatus.DOWNLOADED
    assert fetcher.calls == 3


def test_verify_reports_missing_and_modified_files(tmp_path: Path) -> None:
    fetcher, manifest = FakeFetcher(), MemoryManifest()
    _use_case(tmp_path, fetcher, manifest).execute([_source()])
    verify = VerifyRawDatasets(manifest, FakeHasher(), tmp_path / "raw")

    assert verify.execute() == []

    (tmp_path / "raw" / "toy" / "a.txt").write_text("changed", encoding="utf-8")
    assert [p.problem for p in verify.execute()] == ["hash mismatch"]

    (tmp_path / "raw" / "toy" / "a.txt").unlink()
    assert [p.problem for p in verify.execute()] == ["missing"]
