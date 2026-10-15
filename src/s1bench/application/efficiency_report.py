"""Use case: efficiency table from saved runs (spec sec 9 'Efficiency'). No new inference.

Per model x dataset x split: p50/p95 per-example latency (warm: the first WARMUP rows in FILE order are
dropped, whatever their status), mean input tokens, GPU-seconds per 1,000 examples (local models: sum of forward latencies),
wall-clock throughput (environment start/finish), peak VRAM and GPU name (from the environment file),
API cost per 1,000 (Jev: from a configured price, else null).
v1.1: trained classifiers predict in batches, so their latency comes from the timed per-example pass recorded
in train_summary.json (`latency`, dev subset, batch size 1); local decoders additionally get prefix-cache on/off
latency from latency/<model>_<dataset>_<split>.json when `predict --time-prefix-cache` was run in the same run.
Rows from dev runs or a laptop GPU are marked paper_eligible = False (laptop latency is never reported,
spec sec 3).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from s1bench.domain.predictions import STATUS_OK

WARMUP = 10


def _seconds(env: dict) -> float | None:
    try:
        return (datetime.fromisoformat(env["finished_at"]) - datetime.fromisoformat(env["started_at"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


def _classifier_latency(run_dir: Path) -> dict[tuple[str, str], dict]:
    path = run_dir / "train_summary.json"
    if not path.exists():
        return {}
    out = {}
    for r in json.loads(path.read_text(encoding="utf-8")):   # later entries (re-runs) win
        if r.get("latency"):
            out[(r["model"], r["dataset"])] = r["latency"]   # r["model"] = prediction folder name
    return out


def _prefix_cache(run_dir: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for f in sorted(run_dir.glob("latency/*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[(d["model"], d["dataset"])] = d
    return out


def efficiency_rows(run_dir: Path, api_cost_per_1000: dict[str, float] | None = None) -> list[dict[str, object]]:
    api_cost_per_1000 = api_cost_per_1000 or {}
    clf_latency, prefix = _classifier_latency(run_dir), _prefix_cache(run_dir)
    rows = []
    for f in sorted(run_dir.glob("predictions/*/*.jsonl")):
        model = f.parent.name
        if "_" not in f.stem:
            continue
        dataset, split = f.stem.rsplit("_", 1)
        preds = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        env_path = run_dir / "environment" / model / f"{dataset}_{split}.json"
        env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
        if not env:   # trained classifiers: environment/<base model>/train_*.json listing the dataset
            for tp in sorted((run_dir / "environment" / model.split("@")[0]).glob("train_*.json")):
                cand = json.loads(tp.read_text(encoding="utf-8"))
                if dataset in (cand.get("datasets") or []):
                    env = cand
        desc = env.get("model", {}) if isinstance(env.get("model"), dict) else {}
        ok = [p for p in preds if p["status"] == STATUS_OK]
        lat = np.array([p["latency_ms"] for p in preds[WARMUP:]
                        if p["status"] == STATUS_OK and p.get("latency_ms") is not None], dtype=float)
        tokens = [p["input_tokens"] for p in ok if p.get("input_tokens") is not None]
        wall = _seconds(env)
        gpu = str(desc.get("gpu", ""))
        local = str(desc.get("device", "")).startswith("cuda")
        timed = clf_latency.get((model, dataset)) if len(lat) == 0 else None
        if timed:   # trained classifier: separate timed per-example pass (batch size 1, dev subset)
            gpu, local = str(timed.get("gpu", "")), str(timed.get("device", "")).startswith("cuda")
        pc = prefix.get((model, dataset))
        reasons = []
        if run_dir.name.startswith("dev_"):
            reasons.append("dev run")
        if "laptop" in gpu.lower():
            reasons.append("laptop GPU")
        if not env:
            reasons.append("no environment file")
        if len(lat) == 0 and not timed:
            reasons.append("no latency")
        rows.append({
            "run": run_dir.name, "model": model, "dataset": dataset, "split": split, "n": len(preds), "n_ok": len(ok),
            "n_latency_warm": int(len(lat)) if not timed else timed["n"],
            "latency_source": ("timed pass, dev subset, batch 1" if timed else "prediction run" if len(lat) else None),
            "p50_ms": float(np.percentile(lat, 50)) if len(lat) else (timed["p50_ms"] if timed else None),
            "p95_ms": float(np.percentile(lat, 95)) if len(lat) else (timed["p95_ms"] if timed else None),
            "mean_input_tokens": float(np.mean(tokens)) if tokens else None,
            # mean ms per example = GPU-seconds per 1,000 examples
            "gpu_seconds_per_1000": (float(lat.mean()) if len(lat) else timed["mean_ms"] if timed else None)
            if local else None,
            "prefix_cache_off_p50_ms": pc["cache_off"]["p50_ms"] if pc else None,
            "prefix_cache_on_p50_ms": pc["cache_on"]["p50_ms"] if pc else None,
            "prefix_cache_on_p95_ms": pc["cache_on"]["p95_ms"] if pc else None,
            "prefix_tokens": pc["prefix_tokens"] if pc else None,
            "throughput_ex_per_s_wall": (len(preds) / wall) if wall else None,
            "peak_vram_mb": desc.get("peak_vram_mb"), "peak_vram_reserved_mb": desc.get("peak_vram_reserved_mb"),
            "gpu": gpu or None, "device": desc.get("device"), "workers": env.get("workers"),
            "quantization": desc.get("quantization"),
            "api_cost_per_1000": api_cost_per_1000.get(model),
            "paper_eligible": not reasons, "not_eligible_because": reasons,
        })
    # Timing-only runs (predict --time-prefix-cache) have no prediction file: one row per latency file.
    covered = {(r["model"], r["dataset"]) for r in rows}
    for (model, dataset), pc in sorted(prefix.items()):
        if (model, dataset) in covered:
            continue
        desc = (pc.get("environment") or {}).get("model") or {}
        gpu = str(desc.get("gpu", ""))
        local = str(desc.get("device", "")).startswith("cuda")
        reasons = (["dev run"] if run_dir.name.startswith("dev_") else []) + (
            ["laptop GPU"] if "laptop" in gpu.lower() else []) + ([] if local else ["not on GPU"])
        rows.append({
            "run": run_dir.name, "model": model, "dataset": dataset, "split": pc.get("split"),
            "n": pc.get("n_examples"), "n_ok": None, "n_latency_warm": pc["cache_off"].get("n"),
            "latency_source": "prefix-cache timing run (cache off = predict path)",
            "p50_ms": pc["cache_off"]["p50_ms"], "p95_ms": pc["cache_off"].get("p95_ms"),
            "mean_input_tokens": pc.get("mean_input_tokens"),
            "gpu_seconds_per_1000": pc["cache_off"].get("mean_ms") if local else None,
            "prefix_cache_off_p50_ms": pc["cache_off"]["p50_ms"], "prefix_cache_on_p50_ms": pc["cache_on"]["p50_ms"],
            "prefix_cache_on_p95_ms": pc["cache_on"].get("p95_ms"), "prefix_tokens": pc["prefix_tokens"],
            "throughput_ex_per_s_wall": None, "peak_vram_mb": desc.get("peak_vram_mb"),
            "peak_vram_reserved_mb": desc.get("peak_vram_reserved_mb"), "gpu": gpu or None,
            "device": desc.get("device"), "workers": None, "quantization": desc.get("quantization"),
            "api_cost_per_1000": None, "paper_eligible": not reasons, "not_eligible_because": reasons,
        })
    return rows
