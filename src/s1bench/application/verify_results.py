"""Use case: check a results/<run_id>/ folder brought back from the run machine (spec sec 14).

Errors (exit 1) mean the run cannot be used for the paper as is; warnings are reported for review.
1. Integrity: every MANIFEST.json entry exists with the same sha256/size; files under predictions/ and
   environment/ missing from the manifest are errors; other unlisted files (analysis outputs written
   later, e.g. primary/, exp2/) are warnings.
2. Packaging: no secrets file, raw data or model weights; the local secret VALUES (if any) and secret-like
   patterns (bearer tokens, HF tokens, Cloudflare account URLs) must not occur in any file (never printed).
3. Predictions vs frozen artifacts: candidate order = data/candidates (id must exist), gold =
   data/processed, unique ids, full split (or exactly the seeded subset size); statuses only ok / failed /
   blocked; ok rows have K finite non-negative probabilities summing to 1 (Jev: within its 2-decimal
   rounding) and predicted = argmax (ties by position).
4. Environment: every predictions file has one (predict: <model>/<dataset>_<split>; train: <base model>/
   train_* listing the dataset); config_snapshot must be non-empty and its configs/data manifests equal
   this machine's (error); prediction-code hashes ("code:") that differ are warnings to review.
   Smoke runs (dev_smoke) outside a dev_ run id are errors.
5. Failure rules (spec sec 13): failed > 1% on a dataset -> error (rerun); blocked rate reported;
   Jev: exactly one model version across the run (and across runs: CLI).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from s1bench.domain.candidates import CandidateSet
from s1bench.domain.examples import Example
from s1bench.domain.predictions import STATUS_BLOCKED, STATUS_FAILED, STATUS_OK, argmax_label
from s1bench.domain.ports import FileHasher

FORBIDDEN_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth", ".ckpt")
FAILED_RERUN_RATE = 0.01
JEV_EXPECTED_VERSION = "jev-1.13.0"
KNOWN_STATUSES = {STATUS_OK, STATUS_FAILED, STATUS_BLOCKED}
PROB_TOL = 1e-4                   # softmax outputs (local models, classifiers)
SECRET_PATTERNS = (re.compile(rb"(?i)bearer\s+[a-z0-9._\-]{20,}"), re.compile(rb"hf_[A-Za-z0-9]{30,}"),
                   re.compile(rb"api\.cloudflare\.com/client/v4/accounts/[0-9a-f]{32}"))
ANALYSIS_FILES = {"MANIFEST.json", "verify_report.json"}
# Snapshot key of the raw-data manifest before 2026-10-08 (whole-file hash, incl. download timestamps); now
# "data/MANIFEST.json#files" (config_snapshot), so an old key on one side only is not a data difference.
LEGACY_RAW_MANIFEST_KEY, RAW_MANIFEST_KEY = "data/MANIFEST.json", "data/MANIFEST.json#files"


def prob_tolerance(model: str, k: int) -> float:
    """Jev returns 2-decimal probabilities (sum can be 0.99-1.01; analysis renormalizes). Others: softmax."""
    return 0.005 * k + 1e-9 if model == "jev" else PROB_TOL


@dataclass
class VerifyReport:
    run_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files: list[dict[str, object]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class VerifyResults:
    def __init__(self, hasher: FileHasher, read_examples: Callable[[str, str], list[Example]],
                 read_candidates: Callable[[str, str], Mapping[str, CandidateSet]],
                 local_snapshot: Mapping[str, str], secret_values: Iterable[str] = (),
                 frozen_snapshot: Mapping[str, str] | None = None) -> None:
        self._hasher = hasher
        self._examples = read_examples
        self._candidates = read_candidates
        self._snapshot = dict(local_snapshot)
        # spec 16 (v1.1): final runs must match the FREEZE file exactly, code included (difference = error)
        self._frozen = dict(frozen_snapshot) if frozen_snapshot is not None else None
        self._secrets = [s.encode("utf-8") for s in secret_values if s and len(s) >= 8]

    def execute(self, run_dir: Path) -> VerifyReport:
        rep = VerifyReport(run_dir.name)
        if run_dir.name.startswith("dev_"):
            rep.warnings.append("dev run: never a paper result")
        if not self._secrets:
            rep.warnings.append("no secret values loaded from .env on this machine: value scan skipped "
                                "(pattern scan still runs)")
        if not self._snapshot:
            rep.errors.append("local config snapshot is empty (run from the project root)")
        all_files = sorted(p for p in run_dir.rglob("*") if p.is_file())
        self._packaging(run_dir, all_files, rep)
        manifest = self._integrity(run_dir, all_files, rep)
        if manifest is None:
            return rep
        envs = self._environments(run_dir, rep)
        for f in sorted(run_dir.glob("predictions/*/*.jsonl")):
            self._predictions(run_dir, f, envs, rep)
        return rep

    # 1 -----------------------------------------------------------------------------------------
    def _integrity(self, run_dir: Path, all_files: list[Path], rep: VerifyReport) -> dict | None:
        mpath = run_dir / "MANIFEST.json"
        if not mpath.exists():
            rep.errors.append("MANIFEST.json missing")
            return None
        listed = json.loads(mpath.read_text(encoding="utf-8"))["files"]
        for rel, meta in listed.items():
            p = run_dir / rel
            if not p.exists():
                rep.errors.append(f"{rel}: listed in MANIFEST.json but missing")
            elif p.stat().st_size != meta["size_bytes"] or self._hasher.sha256(p) != meta["sha256"]:
                rep.errors.append(f"{rel}: sha256/size differs from MANIFEST.json (modified after the run)")
        for p in all_files:
            rel = p.relative_to(run_dir).as_posix()
            if rel in listed or rel in ANALYSIS_FILES:
                continue
            if rel.split("/")[0] in ("predictions", "environment"):
                rep.errors.append(f"{rel}: not in MANIFEST.json")
            else:
                rep.warnings.append(f"{rel}: not in MANIFEST.json (analysis output? not integrity-protected)")
        return listed

    # 2 -----------------------------------------------------------------------------------------
    def _packaging(self, run_dir: Path, all_files: list[Path], rep: VerifyReport) -> None:
        for p in all_files:
            rel = p.relative_to(run_dir).as_posix()
            if p.name.startswith(".env") or p.suffix.lower() in FORBIDDEN_SUFFIXES or "/raw/" in f"/{rel}":
                rep.errors.append(f"{rel}: must never be packaged (secrets / weights / raw data)")
            data = p.read_bytes()
            if self._secrets and any(s in data for s in self._secrets):
                rep.errors.append(f"{rel}: contains a secret value from .env (value not shown)")
            if any(pat.search(data) for pat in SECRET_PATTERNS):
                rep.errors.append(f"{rel}: contains a secret-like pattern (token / account URL; value not shown)")

    # 4 -----------------------------------------------------------------------------------------
    def _environments(self, run_dir: Path, rep: VerifyReport) -> dict[str, dict]:
        envs = {}
        for f in sorted(run_dir.glob("environment/*/*.json")):
            env = json.loads(f.read_text(encoding="utf-8"))
            envs[f.relative_to(run_dir / "environment").with_suffix("").as_posix()] = env
            name = f"environment/{f.parent.name}/{f.name}"
            snap = env.get("config_snapshot")
            if not snap:
                rep.errors.append(f"{name}: no (or empty) config_snapshot")
                continue
            if self._frozen is not None:
                for path in sorted(set(snap) | set(self._frozen)):
                    if snap.get(path) != self._frozen.get(path):
                        rep.errors.append(f"{name}: {path} differs from the FREEZE file (run not on the frozen protocol)")
            raw_keys = {LEGACY_RAW_MANIFEST_KEY, RAW_MANIFEST_KEY}
            old_format = (LEGACY_RAW_MANIFEST_KEY in snap) != (LEGACY_RAW_MANIFEST_KEY in self._snapshot)
            if old_format:
                rep.warnings.append(f"{name}: old-format snapshot of {LEGACY_RAW_MANIFEST_KEY} (hash included download "
                                    "timestamps); raw data not compared here, run `verify` on the run machine")
            for path in sorted(set(snap) | set(self._snapshot)):
                if snap.get(path) != self._snapshot.get(path):
                    why = ("missing there" if path not in snap else "missing here" if path not in self._snapshot
                           else "hash differs")
                    if path in raw_keys and old_format:
                        continue
                    target = rep.warnings if path.startswith("code:") else rep.errors
                    target.append(f"{name}: {path} differs from this machine ({why})")
            model_desc = env.get("model") if isinstance(env.get("model"), dict) else {}
            if model_desc.get("dev_smoke") and not run_dir.name.startswith("dev_"):
                rep.errors.append(f"{name}: dev_smoke run outside a dev_ run id")
        if (run_dir / "environment.json").exists():
            rep.warnings.append("environment.json (old single-file layout): may have been overwritten by later "
                                "invocations; per-file environment check skipped for it")
        return envs

    # 3 + 5 -------------------------------------------------------------------------------------
    def _predictions(self, run_dir: Path, f: Path, envs: dict[str, dict], rep: VerifyReport) -> None:
        model = f.parent.name
        rel = f.relative_to(run_dir).as_posix()
        if "_" not in f.stem or f.stem.rsplit("_", 1)[1] not in ("train", "dev", "test"):
            rep.errors.append(f"{rel}: unexpected file name (expected <dataset>_<split>.jsonl)")
            return
        dataset, split = f.stem.rsplit("_", 1)
        env = envs.get(f"{model}/{dataset}_{split}")
        base_model = model.split("@")[0]       # budget runs: <model>@b<budget>s<seed>
        train_env = [v for k, v in envs.items() if k.startswith(f"{base_model}/train_")
                     and dataset in (v.get("datasets") or [])]
        if env is None and not train_env:
            (rep.warnings if (run_dir / "environment.json").exists() else rep.errors).append(
                f"{rel}: no environment file for this model/dataset/split")
        rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        examples = {e.id: e for e in self._examples(dataset, split)}
        cands = self._candidates(dataset, split)
        ids = [r["example_id"] for r in rows]
        if len(set(ids)) != len(ids):
            rep.errors.append(f"{rel}: duplicate example ids")
        unknown = [i for i in ids if i not in examples]
        if unknown:
            rep.errors.append(f"{rel}: {len(unknown)} ids not in data/processed ({unknown[:3]})")
        limit = (env or {}).get("limit")
        if len(set(ids)) < len(examples):
            msg = f"{rel}: {len(set(ids))}/{len(examples)} examples"
            if limit and len(set(ids)) == min(int(limit), len(examples)):
                rep.warnings.append(msg + f" (seeded subset, limit={limit}: not a full run)")
            elif limit:
                rep.errors.append(msg + f" (seeded subset incomplete: expected {min(int(limit), len(examples))})")
            else:
                rep.errors.append(msg + " (incomplete full run)")
        bad_cand = bad_gold = bad_prob = bad_pred = bad_meta = no_cand = 0
        statuses: dict[str, int] = {}
        versions: set[str] = set()
        for r in rows:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
            if r["model"] != model or r["dataset"] != dataset or r["split"] != split:
                bad_meta += 1
            eid = r["example_id"]
            if eid not in cands:
                no_cand += 1
            elif tuple(r["candidates"]) != cands[eid].candidates:
                bad_cand += 1
            if eid in examples and r["gold"] != examples[eid].label:
                bad_gold += 1
            if r["status"] == STATUS_OK:
                probs = r["probabilities"]
                finite = bool(probs) and all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0
                                             for x in probs)
                if (not finite or len(probs) != len(r["candidates"])
                        or abs(sum(probs) - 1.0) > prob_tolerance(model, len(probs))):
                    bad_prob += 1
                elif r["predicted"] != argmax_label(r["candidates"], probs):
                    bad_pred += 1
                if r.get("model_version"):
                    versions.add(r["model_version"])
        for n, what in ((bad_meta, "model/dataset/split fields differ from the file path"),
                        (bad_cand, "candidate order differs from data/candidates"),
                        (bad_gold, "gold differs from data/processed"),
                        (no_cand, "ids missing from data/candidates (candidate order unchecked)"),
                        (bad_prob, "ok rows with bad probability vectors (length, NaN/inf, sign or sum)"),
                        (bad_pred, "predicted != argmax of probabilities")):
            if n:
                rep.errors.append(f"{rel}: {n} {what}")
        unknown_status = sorted(set(statuses) - KNOWN_STATUSES)
        if unknown_status:
            rep.errors.append(f"{rel}: unknown statuses {unknown_status}")
        n = max(len(rows), 1)
        failed, blocked = statuses.get(STATUS_FAILED, 0), statuses.get(STATUS_BLOCKED, 0)
        if failed / n > FAILED_RERUN_RATE:
            rep.errors.append(f"{rel}: failed {failed}/{n} > 1% -> rerun the whole dataset (spec sec 13)")
        if model == "jev":
            if len(versions) != 1:
                rep.errors.append(f"{rel}: Jev model versions {sorted(versions)} (must be exactly one)")
            elif versions != {JEV_EXPECTED_VERSION}:
                rep.warnings.append(f"{rel}: Jev version {sorted(versions)[0]} (spec pilot saw {JEV_EXPECTED_VERSION})")
        rep.files.append({"file": rel, "n": len(rows), "statuses": statuses, "blocked_rate": blocked / n,
                          "model_versions": sorted(versions)})


def check_jev_versions_across_runs(reports: Iterable[VerifyReport]) -> set[str]:
    """All Jev runs of the paper must share one version (spec sec 12); caller compares the set size."""
    return {v for r in reports for f in r.files if f["file"].startswith("predictions/jev/")
            for v in f["model_versions"]}
