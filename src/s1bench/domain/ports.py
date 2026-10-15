"""Interfaces the application layer depends on. Infrastructure implements them."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from s1bench.domain.entities import DatasetSource, RawDatasetRecord, RawFile
from s1bench.domain.candidates import CandidateSet
from s1bench.domain.examples import Example, LabelSet, RawExample
from s1bench.domain.predictions import Prediction


class RawFetcher(Protocol):
    def fetch(self, source: DatasetSource, dest_dir: Path) -> list[RawFile]:
        """Download all files of `source` into `dest_dir` and return them with hashes."""
        ...


class FileHasher(Protocol):
    def sha256(self, path: Path) -> str: ...


class ManifestStore(Protocol):
    def load(self) -> dict[str, RawDatasetRecord]: ...

    def save(self, records: dict[str, RawDatasetRecord]) -> None: ...


class Clock(Protocol):
    def now_iso(self) -> str: ...


class RawSplitReader(Protocol):
    def read(self, dataset: str) -> dict[str, list[RawExample]]:
        """Raw rows per source split ("train", "test", and "dev" where the source has one)."""
        ...


class LabelRepository(Protocol):
    def load(self, dataset: str) -> LabelSet: ...


class DecisionModel(Protocol):
    name: str

    def predict(self, example: Example, candidates: CandidateSet, labels: LabelSet) -> Prediction: ...

    def describe(self) -> dict[str, object]:
        """Model id/version/settings for environment.json."""
        ...


class CandidateReader(Protocol):
    def read_split(self, dataset: str, split: str) -> dict[str, CandidateSet]: ...


class PredictionStore(Protocol):
    def write(self, model: str, dataset: str, split: str, predictions: list[Prediction]) -> RawFile: ...

    def write_environment(self, payload: dict[str, object], name: str = "environment") -> RawFile: ...


class TextEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def describe(self) -> dict[str, str]:
        """Model id and revision, for the manifest."""
        ...


class ProcessedReader(Protocol):
    def read_split(self, dataset: str, split: str) -> list[Example]: ...


class CandidateStore(Protocol):
    def write(self, dataset: str, name: str, rows: list[dict[str, object]]) -> RawFile: ...

    def write_json(self, dataset: str, name: str, payload: dict[str, object]) -> RawFile: ...

    def write_manifest(self, entries: dict[str, dict[str, object]]) -> None: ...


class ProcessedStore(Protocol):
    def write_split(self, dataset: str, split: str, examples: list[Example]) -> RawFile: ...

    def write_report(self, dataset: str, report: dict[str, object]) -> RawFile: ...

    def write_manifest(self, entries: dict[str, dict[str, object]]) -> None: ...
