"""Load configs/*.yaml into domain objects."""

from __future__ import annotations

from pathlib import Path

import yaml

from s1bench.domain.entities import DatasetSource, SourceFile, SourceKind


def load_dataset_sources(path: Path) -> list[DatasetSource]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources: list[DatasetSource] = []
    seen: set[str] = set()
    for item in raw.get("datasets", []):
        dataset_id = item["id"]
        if dataset_id in seen:
            raise ValueError(f"Duplicate dataset id in {path}: {dataset_id}")
        seen.add(dataset_id)

        kind = SourceKind(item["kind"])
        if kind is SourceKind.HF_SNAPSHOT and not (item.get("repo_id") and item.get("revision")):
            raise ValueError(f"{dataset_id}: hf_snapshot needs repo_id and a pinned revision")
        if kind is SourceKind.URL and not item.get("files"):
            raise ValueError(f"{dataset_id}: url source needs at least one file")

        sources.append(
            DatasetSource(
                dataset_id=dataset_id,
                kind=kind,
                license=str(item.get("license", "")),
                homepage=item.get("homepage", ""),
                notes=" ".join(str(item.get("notes", "")).split()),
                repo_id=item.get("repo_id", ""),
                revision=item.get("revision", ""),
                allow_patterns=tuple(item.get("allow_patterns", ())),
                files=tuple(SourceFile(url=f["url"], target=f["target"]) for f in item.get("files", ())),
            )
        )
    return sources
