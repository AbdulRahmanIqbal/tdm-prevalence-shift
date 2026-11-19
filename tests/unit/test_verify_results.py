from __future__ import annotations

from pathlib import Path

from s1bench.application.verify_results import VerifyResults
from s1bench.domain.candidates import CandidateSet
from s1bench.domain.examples import Example
from s1bench.domain.predictions import Prediction
from s1bench.infrastructure.hashing import Sha256Hasher
from s1bench.infrastructure.storage.results_store import JsonlPredictionStore

H = Sha256Hasher()
EXAMPLES = [Example(f"toy:test:{i}", "toy", "test", f"text {i}", "a" if i % 2 else "b", 0 if i % 2 else 1)
            for i in range(4)]
CANDS = {e.id: CandidateSet(e.id, ("a", "b"), 0 if e.label == "a" else 1) for e in EXAMPLES}
SNAP = {"configs/models.yaml": "abc", "code:src/x.py": "c1"}


def _pred(e, model="m", probs=(0.7, 0.3), status="ok", extra=""):
    return Prediction(e.id, model, "toy", "test", ("a", "b"), probs if status == "ok" else (),
                      ("a" if probs[0] >= probs[1] else "b") if status == "ok" else None, e.label,
                      status=status, model_version="jev-1.13.0" if model == "jev" else None, raw={"x": extra})


def _run(tmp: Path, preds=None, model="m", env=None, run_id="run1", env_name=None) -> Path:
    run = tmp / run_id
    store = JsonlPredictionStore(run, H)
    store.write(model, "toy", "test", preds if preds is not None else [_pred(e, model) for e in EXAMPLES])
    store.write_environment(env if env is not None else {"limit": None, "config_snapshot": SNAP},
                            name=env_name or f"{model}/toy_test")
    return run


def _uc(secrets=(), snapshot=SNAP, cands=CANDS):
    return VerifyResults(H, lambda d, s: EXAMPLES, lambda d, s: cands, snapshot, secrets)


def test_clean_run_passes(tmp_path):
    rep = _uc(["x" * 12]).execute(_run(tmp_path))
    assert rep.ok, rep.errors


def test_incomplete_run_and_config_mismatch_fail(tmp_path):
    rep = _uc().execute(_run(tmp_path, preds=[_pred(e) for e in EXAMPLES[:3]],
                             env={"limit": None, "config_snapshot": {**SNAP, "configs/models.yaml": "zzz"}}))
    assert any("incomplete full run" in e for e in rep.errors)
    assert any("configs/models.yaml differs" in e for e in rep.errors)


def test_code_difference_is_warning_and_empty_snapshot_is_error(tmp_path):
    rep = _uc().execute(_run(tmp_path, env={"limit": None, "config_snapshot": {**SNAP, "code:src/x.py": "c2"}}))
    assert rep.ok and any("code:src/x.py differs" in w for w in rep.warnings)
    rep = _uc(snapshot={}).execute(_run(tmp_path / "b", env={"limit": None, "config_snapshot": {}}))
    assert any("empty" in e for e in rep.errors)


def test_secret_value_and_pattern_detected_but_not_printed(tmp_path):
    secret = "sk-very-secret-token-123"
    rep = _uc([secret]).execute(_run(tmp_path, preds=[_pred(e, extra=secret) for e in EXAMPLES]))
    assert any("contains a secret value" in e for e in rep.errors)
    assert not any(secret in e for e in rep.errors + rep.warnings)
    token = "Bearer " + "Ab3" * 10
    rep = _uc().execute(_run(tmp_path / "p", preds=[_pred(e, extra=token) for e in EXAMPLES]))
    assert any("secret-like pattern" in e for e in rep.errors)
    assert any("value scan skipped" in w for w in rep.warnings)


def test_unlisted_and_forbidden_files(tmp_path):
    run = _run(tmp_path)
    (run / "predictions" / "m" / "extra.jsonl").write_text("", encoding="utf-8")
    (run / ".env").write_text("A=1", encoding="utf-8")
    (run / "primary").mkdir()
    (run / "primary" / "p1_toy.json").write_text("{}", encoding="utf-8")
    rep = _uc().execute(run)
    assert any("extra.jsonl: not in MANIFEST.json" in e for e in rep.errors)
    assert any(".env: must never be packaged" in e for e in rep.errors)
    assert any("primary/p1_toy.json: not in MANIFEST.json" in w for w in rep.warnings)


def test_budget_model_uses_train_environment_and_must_be_complete(tmp_path):
    env = {"datasets": ["toy"], "config_snapshot": SNAP, "model": {"name": "m"}}
    full = _run(tmp_path, model="m@b100s1001", env=env, env_name="m/train_toy_20261007T000000000000Z")
    assert _uc().execute(full).ok, _uc().execute(full).errors
    short = _run(tmp_path / "s", preds=[_pred(e, "m@b100s1001") for e in EXAMPLES[:2]], model="m@b100s1001",
                 env=env, env_name="m/train_toy_20261007T000000000000Z")
    assert any("incomplete full run" in e for e in _uc().execute(short).errors)


def test_nan_unknown_status_missing_candidates_and_tolerance(tmp_path):
    preds = [_pred(EXAMPLES[0], probs=(float("nan"), 0.3)), _pred(EXAMPLES[1], status="error"),
             _pred(EXAMPLES[2], probs=(0.6, 0.38)), _pred(EXAMPLES[3])]
    rep = _uc(cands={k: v for k, v in CANDS.items() if k != EXAMPLES[3].id}).execute(_run(tmp_path, preds=preds))
    errs = " | ".join(rep.errors)
    assert "2 ok rows with bad probability vectors" in errs      # NaN row + sum 0.98 (non-Jev tolerance 1e-4)
    assert "unknown statuses ['error']" in errs
    assert "1 ids missing from data/candidates" in errs
    jev = [_pred(e, "jev", probs=(0.6, 0.39)) for e in EXAMPLES]   # Jev 2-decimal rounding: allowed
    assert _uc().execute(_run(tmp_path / "j", preds=jev, model="jev")).ok


def test_seeded_subset_must_have_exactly_limit_rows(tmp_path):
    env = {"limit": 3, "config_snapshot": SNAP}
    ok = _run(tmp_path, preds=[_pred(e) for e in EXAMPLES[:3]], env=env)
    assert _uc().execute(ok).ok
    short = _run(tmp_path / "s", preds=[_pred(e) for e in EXAMPLES[:2]], env=env)
    assert any("seeded subset incomplete" in e for e in _uc().execute(short).errors)


def test_smoke_run_outside_dev_run_id_is_error(tmp_path):
    env = {"datasets": ["toy"], "config_snapshot": SNAP, "model": {"dev_smoke": True}}
    run = _run(tmp_path, env=env, env_name="m/train_toy_1")
    assert any("dev_smoke run outside" in e for e in _uc().execute(run).errors)


def test_legacy_raw_manifest_key_is_warning_but_new_key_mismatch_is_error(tmp_path):
    old = {**SNAP, "data/MANIFEST.json": "with-timestamps"}
    rep = _uc().execute(_run(tmp_path, env={"limit": None, "config_snapshot": old}))
    assert rep.ok and any("old-format snapshot of data/MANIFEST.json" in w for w in rep.warnings)
    here = {**SNAP, "data/MANIFEST.json#files": "f1"}
    rep = _uc(snapshot=here).execute(_run(tmp_path / "b", env={"limit": None,
                                                               "config_snapshot": {**SNAP, "data/MANIFEST.json#files": "f2"}}))
    assert any("data/MANIFEST.json#files differs" in e for e in rep.errors)


def test_raw_manifest_hash_ignores_download_timestamps(tmp_path):
    import json as _json
    from s1bench.infrastructure.storage.results_store import raw_manifest_sha256

    def write(name, when, sha="aa"):
        p = tmp_path / name
        p.write_text(_json.dumps({"schema": 1, "datasets": {"d": {"downloaded_at": when, "files": [
            {"path": "x", "sha256": sha, "size_bytes": 1}]}}}), encoding="utf-8")
        return raw_manifest_sha256(p)

    assert write("a.json", "2026-10-06") == write("b.json", "2026-10-07")
    assert write("c.json", "2026-10-06") != write("d.json", "2026-10-06", sha="bb")


def test_frozen_snapshot_difference_is_error_even_for_code(tmp_path):
    frozen = {**SNAP, "code:src/x.py": "c1"}
    uc = VerifyResults(H, lambda d, s: EXAMPLES, lambda d, s: CANDS, frozen, (), frozen_snapshot=frozen)
    assert uc.execute(_run(tmp_path, env={"limit": None, "config_snapshot": frozen})).ok
    rep = uc.execute(_run(tmp_path / "b", env={"limit": None, "config_snapshot": {**SNAP, "code:src/x.py": "c2"}}))
    assert any("differs from the FREEZE file" in e for e in rep.errors)


def test_freeze_differences_detects_changed_file(tmp_path):
    from s1bench.infrastructure.storage.results_store import freeze_differences, freeze_record

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "models.yaml").write_text("a: 1\n")
    frozen = freeze_record(tmp_path)
    assert freeze_differences(frozen, tmp_path) == []
    (tmp_path / "configs" / "models.yaml").write_text("a: 2\n")
    assert freeze_differences(frozen, tmp_path) == ["configs/models.yaml: hash differs"]


def test_router_candidates_dir_expects_listed_examples_only(tmp_path):
    """v1.3 (G3): candidates read from the environment's candidates_dir (sha256 checked), listed examples only."""
    root = tmp_path / "proj"
    (root / "alt" / "toy").mkdir(parents=True)
    f = root / "alt" / "toy" / "test.jsonl"
    f.write_text("listed\n", encoding="utf-8")
    listed = {EXAMPLES[0].id: CandidateSet(EXAMPLES[0].id, ("b", "a"), -1)}
    env = {"limit": None, "config_snapshot": SNAP, "candidates_dir": "alt", "candidates_file_sha256": H.sha256(f)}
    pred = Prediction(EXAMPLES[0].id, "m", "toy", "test", ("b", "a"), (0.6, 0.4), "b", EXAMPLES[0].label)

    def uc():
        return VerifyResults(H, lambda d, s: EXAMPLES, lambda d, s: CANDS, SNAP, (),
                             examples_from=lambda r, d, s: EXAMPLES, candidates_from=lambda r, d, s: listed,
                             project_root=root)

    assert uc().execute(_run(tmp_path, preds=[pred], env=env)).ok
    bad = {**env, "candidates_file_sha256": "0" * 64}
    rep = uc().execute(_run(tmp_path, preds=[pred], env=bad, run_id="run2"))
    assert any("sha256 differs" in e for e in rep.errors)
    rep = _uc().execute(_run(tmp_path, preds=[pred], env=env, run_id="run3"))    # no reader for the folder
    assert not rep.ok
