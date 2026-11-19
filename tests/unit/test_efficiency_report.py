from __future__ import annotations

from s1bench.application import efficiency_report
from s1bench.domain.predictions import Prediction
from s1bench.infrastructure.hashing import Sha256Hasher
from s1bench.infrastructure.storage.results_store import JsonlPredictionStore


def _write(run, gpu):
    store = JsonlPredictionStore(run, Sha256Hasher())
    preds = [Prediction(f"toy:test:{i}", "m", "toy", "test", ("a", "b"), (0.6, 0.4), "a", "a",
                        latency_ms=1000.0 if i < efficiency_report.WARMUP else float(i), input_tokens=100)
             for i in range(110)]
    store.write("m", "toy", "test", preds)
    store.write_environment({"started_at": "2026-10-07T00:00:00+00:00", "finished_at": "2026-10-07T00:00:11+00:00",
                             "model": {"device": "cuda", "gpu": gpu, "peak_vram_mb": 9000.0}}, name="m/toy_test")


def test_warmup_dropped_and_eligibility(tmp_path):
    _write(tmp_path / "paper_run", "NVIDIA GeForce RTX 3060")
    (row,) = efficiency_report.efficiency_rows(tmp_path / "paper_run")
    assert row["n_latency_warm"] == 100 and row["p50_ms"] == 59.5 and row["paper_eligible"]
    assert abs(row["throughput_ex_per_s_wall"] - 10.0) < 1e-9 and row["gpu_seconds_per_1000"] == 59.5
    _write(tmp_path / "dev_x", "NVIDIA GeForce RTX 4050 Laptop GPU")
    (row,) = efficiency_report.efficiency_rows(tmp_path / "dev_x")
    assert not row["paper_eligible"] and set(row["not_eligible_because"]) == {"dev run", "laptop GPU"}


def test_warmup_counts_file_rows_and_train_environment_found(tmp_path):
    run = tmp_path / "paper_run"
    store = JsonlPredictionStore(run, Sha256Hasher())
    preds = [Prediction(f"toy:test:{i}", "m", "toy", "test", ("a", "b"), () if i < 5 else (0.6, 0.4),
                        None if i < 5 else "a", "a", status="failed" if i < 5 else "ok",
                        latency_ms=None if i < 5 else 1.0) for i in range(30)]
    store.write("m@b100s1", "toy", "test", preds)
    store.write_environment({"datasets": ["toy"], "model": {"device": "cuda", "gpu": "RTX 3060"}},
                            name="m/train_toy_1")
    (row,) = efficiency_report.efficiency_rows(run)
    assert row["n_latency_warm"] == 20          # first 10 FILE rows dropped (5 failed + 5 ok)
    assert "no environment file" not in row["not_eligible_because"] and row["device"] == "cuda"


def test_classifier_timed_pass_and_prefix_cache_merged(tmp_path):
    import json

    run = tmp_path / "paper_run"
    store = JsonlPredictionStore(run, Sha256Hasher())
    store.write("clf@b100s1", "toy", "test", [Prediction(f"toy:test:{i}", "clf@b100s1", "toy", "test", ("a", "b"),
                                                         (0.6, 0.4), "a", "a") for i in range(20)])
    store.write_environment({"datasets": ["toy"], "model": {}}, name="clf/train_toy_1")
    (run / "train_summary.json").write_text(json.dumps([
        {"model": "clf@b100s1", "dataset": "toy", "latency": {"n": 200, "p50_ms": 9.0, "p95_ms": 12.0,
                                                               "mean_ms": 10.0, "device": "cuda", "gpu": "RTX 3060"}}]))
    _write(run, "RTX 3060")
    store.write_json("latency/m_toy_dev.json", {"model": "m", "dataset": "toy", "prefix_tokens": 44,
                                                "cache_off": {"p50_ms": 60.0}, "cache_on": {"p50_ms": 50.0, "p95_ms": 55.0}})
    rows = {r["model"]: r for r in efficiency_report.efficiency_rows(run)}
    clf = rows["clf@b100s1"]
    assert clf["p50_ms"] == 9.0 and clf["gpu_seconds_per_1000"] == 10.0 and clf["paper_eligible"]
    assert clf["latency_source"].startswith("timed pass") and clf["prefix_cache_on_p50_ms"] is None
    assert rows["m"]["prefix_cache_on_p50_ms"] == 50.0 and rows["m"]["prefix_tokens"] == 44


def test_timing_only_run_gets_a_row(tmp_path):
    run = tmp_path / "paper_run"
    JsonlPredictionStore(run, Sha256Hasher()).write_json("latency/q_toy_dev.json", {
        "model": "q", "dataset": "toy", "split": "dev", "n_examples": 210, "prefix_tokens": 44, "mean_input_tokens": 300,
        "cache_off": {"p50_ms": 60.0, "p95_ms": 70.0, "mean_ms": 61.0, "n": 200},
        "cache_on": {"p50_ms": 50.0, "p95_ms": 55.0},
        "environment": {"model": {"device": "cuda", "gpu": "RTX 4070 Ti", "peak_vram_mb": 8000.0}}})
    (row,) = efficiency_report.efficiency_rows(run)
    assert row["p50_ms"] == 60.0 and row["prefix_cache_on_p50_ms"] == 50.0 and row["paper_eligible"]
    assert row["gpu_seconds_per_1000"] == 61.0 and row["split"] == "dev"
