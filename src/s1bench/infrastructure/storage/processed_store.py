from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import yaml

from s1bench.domain.entities import RawFile
from s1bench.domain.examples import Example, Label, LabelSet
from s1bench.domain.ports import FileHasher


class YamlLabelRepository:
    """Reads the frozen label texts in data/labels/<dataset>.yaml."""

    def __init__(self, labels_dir: Path) -> None:
        self._dir = labels_dir

    def load(self, dataset: str) -> LabelSet:
        raw = yaml.safe_load((self._dir / f"{dataset}.yaml").read_text(encoding="utf-8"))
        labels = tuple(
            Label(id=str(item["id"]), name=item["name"], description=item["description"],
                  raw_name=str(item.get("raw_name", item["id"])))
            for item in raw["labels"]
        )
        return LabelSet(dataset=raw["dataset"], task_question=raw["task_question"], labels=labels)


class JsonlProcessedStore:
    """data/processed/<dataset>/<split>.jsonl + report.json + MANIFEST.json."""

    def __init__(self, root: Path, hasher: FileHasher) -> None:
        self._root = root
        self._hasher = hasher

    def _finish(self, path: Path) -> RawFile:
        return RawFile(
            path=path.relative_to(self._root).as_posix(),
            sha256=self._hasher.sha256(path),
            size_bytes=path.stat().st_size,
        )

    def write_split(self, dataset: str, split: str, examples: list[Example]) -> RawFile:
        path = self._root / dataset / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for e in examples:
                fh.write(json.dumps(asdict(e), ensure_ascii=False, sort_keys=True) + "\n")
        return self._finish(path)

    def write_report(self, dataset: str, report: dict[str, object]) -> RawFile:
        path = self._root / dataset / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return self._finish(path)

    def write_manifest(self, entries: dict[str, dict[str, object]]) -> None:
        path = self._root / "MANIFEST.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"datasets": {}}
        existing.setdefault("datasets", {}).update(entries)
        existing["datasets"] = dict(sorted(existing["datasets"].items()))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


class JsonlProcessedReader:
    """Reads data/processed/<dataset>/<split>.jsonl back into Example objects."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def read_split(self, dataset: str, split: str) -> list[Example]:
        path = self._root / dataset / f"{split}.jsonl"
        with path.open(encoding="utf-8") as fh:
            return [Example(**json.loads(line)) for line in fh if line.strip()]


class JsonlCandidateStore:
    """data/candidates/<dataset>/<name>.jsonl|json + MANIFEST.json."""

    def __init__(self, root: Path, hasher: FileHasher) -> None:
        self._root = root
        self._hasher = hasher

    def _finish(self, path: Path) -> RawFile:
        return RawFile(path=path.relative_to(self._root).as_posix(), sha256=self._hasher.sha256(path),
                       size_bytes=path.stat().st_size)

    def write(self, dataset: str, name: str, rows: list[dict[str, object]]) -> RawFile:
        path = self._root / dataset / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return self._finish(path)

    def write_json(self, dataset: str, name: str, payload: dict[str, object]) -> RawFile:
        path = self._root / dataset / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return self._finish(path)

    def write_manifest(self, entries: dict[str, dict[str, object]]) -> None:
        path = self._root / "MANIFEST.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"datasets": {}}
        existing.setdefault("datasets", {}).update(entries)
        existing["datasets"] = dict(sorted(existing["datasets"].items()))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
