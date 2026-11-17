"""Exploratory (added after the results): robustness of the calibration comparisons to the calibration metric.

Metrics, all on the candidate-restricted probabilities: top-label ECE with 15 equal-width bins (the paper's metric);
top-label ECE with 15 equal-mass bins; debiased top-label calibration error (Kumar et al. 2019: square root of the
debiased estimate of the squared error, 15 equal-mass bins); class-wise ECE (Kull et al. 2019: for each class, ECE of
its predicted probability over the examples that offer it, 15 equal-width bins, averaged over classes); Brier score.

(1) No shift: the paired differences behind RQ1 (Tev1-4B - Qwen3.5-4B, Nimble-9B - Qwen3.5-9B) and Jev - DeBERTa.
(2) Shift: change of each metric at alpha = 1 (mean over the 200 with-replacement draws of Exp 2, raw probabilities)
minus the resampled no-shift control, Jev - DeBERTa-v3-large (seed 1001). The draws are regenerated with the frozen Exp 2
seed and order and checked against the stored Exp 2 rows (top-label ECE of every draw must match).

Writes results/final_analysis_calibrob/*.json, paper/tables/tab_calib_robust_pairs.tex and tab_calib_robust_shift.tex.
Run: .venv/Scripts/python scripts/paper/calibration_robustness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from s1bench.application.prevalence_shift import to_matrices  # noqa: E402
from s1bench.domain import label_shift as ls  # noqa: E402
from s1bench.infrastructure.storage.processed_store import JsonlProcessedReader, YamlLabelRepository  # noqa: E402
from s1bench.infrastructure.storage.results_store import read_predictions  # noqa: E402

RES = ROOT / "results"
OUT = RES / "final_analysis_calibrob"
DATASETS = ["clinc150", "hwu64", "fpb", "hatexplain", "trec6"]
DS = {"clinc150": "CLINC150", "hwu64": "HWU64", "fpb": "FPB", "hatexplain": "HateXplain", "trec6": "TREC-6"}
RUN = {"jev": "final_jev", "deberta-v3-large": "final_train", "tfidf-lr": "final_train", "qwen3.5-4b": "final_local",
       "tev1-4b": "final_local", "qwen3.5-9b": "final_local", "nimble-9b": "final_local"}
METRICS = [("ece", "Top-label ECE, equal width (paper)"), ("ece_mass", "Top-label ECE, equal mass"),
           ("ece_debiased", "Debiased calibration error"), ("ece_classwise", "Class-wise ECE"), ("brier", "Brier score")]
PAIRS = [("tev1-4b", "qwen3.5-4b", "Tev1-4B $-$ Qwen3.5-4B"), ("nimble-9b", "qwen3.5-9b", "Nimble-9B $-$ Qwen3.5-9B"),
         ("jev", "deberta-v3-large", "Jev $-$ DeBERTa-v3-large")]
N_BINS = 15


def equal_mass_bins(conf: np.ndarray) -> list[np.ndarray]:
    return [b for b in np.array_split(np.argsort(conf, kind="stable"), N_BINS) if len(b)]


def metrics(probs: np.ndarray, mask: np.ndarray, gold: np.ndarray, positions: np.ndarray) -> dict[str, float]:
    mm = ls.matrix_metrics(probs, gold, positions=positions)
    pred = ls.argmax_by_position(probs, positions)
    conf = probs.max(axis=1)
    corr = (pred == gold).astype(float)
    n = len(gold)
    ece_mass, sq = 0.0, 0.0
    for b in equal_mass_bins(conf):
        a, c, m = corr[b].mean(), conf[b].mean(), len(b)
        ece_mass += m / n * abs(a - c)
        if m > 1:
            sq += m / n * ((a - c) ** 2 - a * (1 - a) / (m - 1))
    cw = []
    for k in range(probs.shape[1]):
        sel = mask[:, k]
        if not sel.any():
            continue
        p, y = probs[sel, k], (gold[sel] == k).astype(float)
        idx = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
        cnt = np.bincount(idx, minlength=N_BINS)
        gap = np.abs(np.bincount(idx, weights=y, minlength=N_BINS) - np.bincount(idx, weights=p, minlength=N_BINS))
        cw.append(gap.sum() / cnt.sum())
    return {"ece": mm.ece, "ece_mass": ece_mass, "ece_debiased": float(np.sqrt(max(sq, 0.0))),
            "ece_classwise": float(np.mean(cw)), "brier": mm.brier}


def matrices(model: str, dataset: str, split: str):
    labels = YamlLabelRepository(ROOT / "data" / "labels").load(dataset)
    m = to_matrices(read_predictions(RES / RUN[model], model, dataset, split), labels)
    if m.n_excluded:
        raise SystemExit(f"{model}/{dataset}: non-ok rows")
    return m, labels


def no_shift() -> dict:
    exp1 = {}
    for f in ("exp1_test.json", "exp1_test_jev.json"):
        for r in json.loads((RES / "final_analysis" / f).read_text(encoding="utf-8")):
            if r["split"] == "test":
                exp1[(r["model"], r["dataset"])] = r
    out = {}
    for ds in DATASETS:
        for model in RUN:
            m, _ = matrices(model, ds, "test")
            v = metrics(m.probs, m.mask, m.gold, m.positions)
            for key in ("ece", "brier"):
                if abs(v[key] - exp1[(model, ds)][key]) > 1e-9:
                    raise SystemExit(f"{model}/{ds}: {key} {v[key]} differs from Exp 1 {exp1[(model, ds)][key]}")
            out[f"{model}|{ds}"] = v
    return out


def shift_effect(model: str, dataset: str) -> dict[str, float]:
    """Mean metric at alpha = 1 minus mean over the resampled no-shift control, raw probabilities, Exp 2 draws."""
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / "exp2.yaml").read_text(encoding="utf-8"))
    test, labels = matrices(model, dataset, "test")
    report = json.loads((ROOT / "data" / "processed" / dataset / "report.json").read_text(encoding="utf-8"))
    counts = report["label_counts"]["train"]
    train_prior = np.array([counts.get(lid, 0) for lid in labels.ids()], dtype=float)
    train_prior /= train_prior.sum()
    index = {lid: i for i, lid in enumerate(labels.ids())}
    full = JsonlProcessedReader(ROOT / "data" / "processed").read_split(dataset, "test")
    full_gold = np.array([index[e.label] for e in full])
    row_of = {eid: r for r, eid in enumerate(test.ids)}
    to_row = np.array([row_of[e.id] for e in full])
    stored = [json.loads(line) for line in
              (RES / RUN[model] / "exp2" / model / f"{dataset}.jsonl").read_text(encoding="utf-8").splitlines()]
    stored_ece = {(r["alpha"], r["draw"]): r["ece"] for r in stored
                  if r["arm"] == "none" and r["sampling"] == "with_replacement"}
    rng = np.random.default_rng(cfg["seed"])
    draws: dict[str, list[np.ndarray]] = {"1": [], "noshift_resampled": []}
    for alpha in cfg["alphas"]:
        for _ in range(cfg["draws_per_alpha"]):
            target = ls.dirichlet_target_prior(train_prior, alpha, rng)
            idx = ls.resample_with_replacement(full_gold, target, len(full_gold), rng)
            ls.resample_without_replacement(full_gold, target, rng)   # consumed in Exp 2; not used here
            if str(alpha) == "1":
                draws["1"].append(to_row[idx])
    full_prior = np.bincount(full_gold, minlength=len(index)) / len(full_gold)
    for _ in range(cfg["draws_per_alpha"]):
        draws["noshift_resampled"].append(to_row[ls.resample_with_replacement(full_gold, full_prior, len(full_gold), rng)])
    mean = {}
    for key, rows in draws.items():
        vals = []
        for d, r in enumerate(rows, 1):
            v = metrics(test.probs[r], test.mask[r], test.gold[r], test.positions[r])
            if round(v["ece"], 6) != stored_ece[(key, d)]:
                raise SystemExit(f"{model}/{dataset} {key} draw {d}: regenerated sample does not match Exp 2")
            vals.append(v)
        mean[key] = {k: float(np.mean([v[k] for v in vals])) for k, _ in METRICS}
    return {k: mean["1"][k] - mean["noshift_resampled"][k] for k, _ in METRICS}


def pts(x: float) -> str:
    return f"{100 * x:+.1f}".replace("-", "$-$").replace("+0.0", "0.0").replace("$-$0.0", "0.0")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ns = no_shift()
    (OUT / "no_shift.json").write_text(json.dumps(ns, indent=1) + "\n", encoding="utf-8")
    lines = ["\\begin{tabular}{ll" + "r" * len(DATASETS) + "}", "\\toprule",
             "Comparison & Metric & " + " & ".join(DS[d] for d in DATASETS) + " \\\\", "\\midrule"]
    for a, b, name in PAIRS:
        for j, (key, label) in enumerate(METRICS):
            cells = [pts(ns[f"{a}|{d}"][key] - ns[f"{b}|{d}"][key]) for d in DATASETS]
            lines.append(f"{name if j == 0 else ''} & {label} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if name != PAIRS[-1][2] else "\\bottomrule")
    lines.append("\\end{tabular}")
    (ROOT / "paper" / "tables" / "tab_calib_robust_pairs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sh = {f"{m}|{d}": shift_effect(m, d) for d in DATASETS for m in ("jev", "deberta-v3-large")}
    (OUT / "shift_alpha1.json").write_text(json.dumps(sh, indent=1) + "\n", encoding="utf-8")
    lines = ["\\begin{tabular}{l" + "r" * len(DATASETS) + "}", "\\toprule",
             "Metric & " + " & ".join(DS[d] for d in DATASETS) + " \\\\", "\\midrule"]
    for key, label in METRICS:
        cells = [pts(sh[f"jev|{d}"][key] - sh[f"deberta-v3-large|{d}"][key]) for d in DATASETS]
        print(f"{label:38s} " + " ".join(f"{100 * (sh[f'jev|{d}'][key] - sh[f'deberta-v3-large|{d}'][key]):+6.2f}"
                                         for d in DATASETS))
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper" / "tables" / "tab_calib_robust_shift.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
