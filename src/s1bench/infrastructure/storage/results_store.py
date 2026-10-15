"""results/<run_id>/: predictions JSONL, environment.json, MANIFEST.json (spec sec 14). Never writes secrets."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

from s1bench.domain.candidates import CandidateSet
from s1bench.domain.entities import RawFile
from s1bench.domain.ports import FileHasher
from s1bench.domain.predictions import Prediction


class JsonlCandidateReader:
    def __init__(self, root: Path) -> None:
        self._root = root

    def read_split(self, dataset: str, split: str) -> dict[str, CandidateSet]:
        out = {}
        with (self._root / dataset / f"{split}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                out[row["id"]] = CandidateSet(row["id"], tuple(row["candidates"]), int(row["gold_position"]))
        return out


def read_predictions(run_dir: Path, model: str, dataset: str, split: str) -> list[Prediction]:
    """results/<run_id>/predictions/<model>/<dataset>_<split>.jsonl -> Prediction list (file order)."""
    out = []
    with (run_dir / "predictions" / model / f"{dataset}_{split}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            d["candidates"], d["probabilities"] = tuple(d["candidates"]), tuple(d["probabilities"])
            out.append(Prediction(**d))
    return out


class JsonlPredictionStore:
    def __init__(self, run_dir: Path, hasher: FileHasher) -> None:
        self._dir = run_dir
        self._hasher = hasher

    def _finish(self, path: Path) -> RawFile:
        f = RawFile(path.relative_to(self._dir).as_posix(), self._hasher.sha256(path), path.stat().st_size)
        manifest = self._dir / "MANIFEST.json"
        data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {"files": {}}
        data["files"][f.path] = {"sha256": f.sha256, "size_bytes": f.size_bytes}
        data["files"] = dict(sorted(data["files"].items()))
        manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return f

    def write(self, model: str, dataset: str, split: str, predictions: list[Prediction]) -> RawFile:
        path = self._dir / "predictions" / model / f"{dataset}_{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for p in predictions:
                fh.write(json.dumps(asdict(p), ensure_ascii=False, sort_keys=True) + "\n")
        return self._finish(path)

    def write_json(self, rel_path: str, payload: object) -> RawFile:
        """Any other run-machine output (e.g. train_summary.json), hashed into MANIFEST.json."""
        path = self._dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + LF.decode(), encoding="utf-8")
        return self._finish(path)

    def write_environment(self, payload: dict[str, object], name: str = "environment") -> RawFile:
        """environment/<name>.json, one file per invocation (several models/datasets share one run dir)."""
        path = self._dir / "environment" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return self._finish(path)


# Only files that can change a PREDICTION (spec sec 14). Analysis-only configs (exp2/primary/efficiency,
# overlap references) are deliberately excluded: editing them later must not invalidate earlier runs.
SNAPSHOT_FILES = ("configs/datasets.yaml", "configs/processing.yaml", "configs/candidates.yaml", "configs/models.yaml",
                  "configs/seeds.yaml",
                  "data/labels/MANIFEST.json", "data/processed/MANIFEST.json",
                  "data/candidates/MANIFEST.json")
# Raw-data manifest: hashed WITHOUT the per-dataset "downloaded_at" timestamps, which `download` rewrites on every
# machine although the raw files (sha256/size) are identical. Older snapshots stored the plain file hash under
# LEGACY_RAW_MANIFEST_KEY (verify-results reports that key as a warning, not an error).
RAW_MANIFEST, RAW_MANIFEST_KEY, LEGACY_RAW_MANIFEST_KEY = "data/MANIFEST.json", "data/MANIFEST.json#files",     "data/MANIFEST.json"
# Prediction-relevant code (no git in this project): differences are reported for review, keyed "code:<path>".
SNAPSHOT_CODE = ("src/s1bench/infrastructure/models/*.py", "src/s1bench/application/run_predictions.py",
                 "src/s1bench/application/train_classifier.py", "src/s1bench/domain/predictions.py",
                 "src/s1bench/domain/budgets.py")


CRLF, LF = bytes([13, 10]), bytes([10])


def _text_sha256(path: Path) -> str:
    """sha256 with CRLF -> LF, so copying between machines/editors does not change the hash."""
    import hashlib

    return hashlib.sha256(path.read_bytes().replace(CRLF, LF)).hexdigest()


def raw_manifest_sha256(path: Path) -> str:
    """sha256 of the raw-data manifest content without the "downloaded_at" timestamps."""
    import hashlib

    manifest = json.loads(path.read_text(encoding="utf-8"))
    for entry in manifest.get("datasets", {}).values():
        entry.pop("downloaded_at", None)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def config_snapshot(project_root: Path, hasher: FileHasher | None = None) -> dict[str, str]:
    """sha256 of prediction-relevant configs, frozen-data manifests and code (spec sec 14 config_snapshot).
    verify-results compares these with the machine the results are brought back to."""
    out = {}
    for pattern in SNAPSHOT_FILES:
        for f in sorted(project_root.glob(pattern)):
            out[f.relative_to(project_root).as_posix()] = _text_sha256(f)
    raw = project_root / RAW_MANIFEST
    if raw.exists():
        out[RAW_MANIFEST_KEY] = raw_manifest_sha256(raw)
    for pattern in SNAPSHOT_CODE:
        for f in sorted(project_root.glob(pattern)):
            out["code:" + f.relative_to(project_root).as_posix()] = _text_sha256(f)
    return out


# Frozen protocol (spec sec 16, v1.1): the spec, analysis configs and pinned requirements are hashed in addition to
# the prediction-relevant config_snapshot. No git/OSF: this file is the record that the protocol was fixed.
FREEZE_EXTRA = ("docs/benchmark_spec_v1.md", "configs/experiments/*.yaml", "configs/overlap_references.yaml",
                "requirements-run.txt", "pyproject.toml", "src/s1bench/**/*.py")


def freeze_record(project_root: Path) -> dict[str, object]:
    files = {}
    for pattern in FREEZE_EXTRA:
        for f in sorted(project_root.glob(pattern)):
            if "__pycache__" not in f.parts:
                files[f.relative_to(project_root).as_posix()] = _text_sha256(f)
    return {"config_snapshot": config_snapshot(project_root), "files": files}


def freeze_differences(frozen: dict[str, object], project_root: Path) -> list[str]:
    """Paths whose hash differs from the freeze file (missing/extra included)."""
    now = freeze_record(project_root)
    out = []
    for part in ("config_snapshot", "files"):
        a, b = frozen.get(part, {}), now[part]
        for path in sorted(set(a) | set(b)):
            if a.get(path) != b.get(path):
                out.append(f"{path}: " + ("missing now" if path not in b else "not in freeze" if path not in a
                                          else "hash differs"))
    return out


def capture_environment(extra: dict[str, object]) -> dict[str, object]:
    """Software/hardware facts for the paper (spec sec 3). GPU info added by local-model runners."""
    pkgs = {}
    for name in ("s1bench", "torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors",
                 "tokenizers", "sentence-transformers", "scikit-learn", "numpy", "pandas", "huggingface-hub",
                 "flash-linear-attention", "causal-conv1d"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(), "packages": pkgs, **extra}
