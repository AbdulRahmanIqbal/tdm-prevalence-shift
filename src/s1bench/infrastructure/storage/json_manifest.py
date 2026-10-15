from __future__ import annotations

import json
from pathlib import Path

from s1bench.domain.entities import RawDatasetRecord, RawFile, SourceKind


class JsonManifestStore:
    """data/MANIFEST.json: one entry per dataset with every file's sha256."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> dict[str, RawDatasetRecord]:
        if not self._path.exists():
            return {}
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        records: dict[str, RawDatasetRecord] = {}
        for dataset_id, item in payload.get("datasets", {}).items():
            records[dataset_id] = RawDatasetRecord(
                dataset_id=dataset_id,
                kind=SourceKind(item["kind"]),
                origin=item["origin"],
                license=item["license"],
                downloaded_at=item["downloaded_at"],
                files=tuple(RawFile(**f) for f in item["files"]),
            )
        return records

    def save(self, records: dict[str, RawDatasetRecord]) -> None:
        payload = {
            "schema": 1,
            "datasets": {
                dataset_id: {
                    "kind": r.kind.value,
                    "origin": r.origin,
                    "license": r.license,
                    "downloaded_at": r.downloaded_at,
                    "total_bytes": r.total_bytes(),
                    "files": [{"path": f.path, "sha256": f.sha256, "size_bytes": f.size_bytes} for f in r.files],
                }
                for dataset_id, r in sorted(records.items())
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self._path)
