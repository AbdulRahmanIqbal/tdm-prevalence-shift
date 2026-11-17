"""EXPLORATORY (post hoc): shift-distribution average of the PA1-PA3 statistics.

The confirmatory PA estimand is conditional on one fixed set of 50 target priors. Here the same statistics are averaged
over many independent target priors q ~ Dirichlet(alpha * K * pi_train) at alpha = 1 (one single-level resample of the
full test set per prior, with a resampled no-shift control for PA1, as in the PA design but without the bootstrap base).
This estimates the population (shift-distribution) mean and the spread across priors.

Writes results/final_analysis/pa_population.json and paper/tables/tab_pa_population.tex.
Run: .venv/Scripts/python scripts/paper/pa_population.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from s1bench.application.primary_tests import _ece, _risk, aligned_matrices  # noqa: E402
from s1bench.domain import label_shift as ls  # noqa: E402
from s1bench.infrastructure.storage.processed_store import YamlLabelRepository  # noqa: E402
from s1bench.infrastructure.storage.results_store import read_predictions  # noqa: E402

DATASETS = ["clinc150", "hwu64", "fpb", "hatexplain", "trec6"]
DS = {"clinc150": "CLINC150", "hwu64": "HWU64", "fpb": "FPB", "hatexplain": "HateXplain", "trec6": "TREC-6"}
N_PRIORS = 2000
SEED = 777


def pts(x: float) -> str:
    s = f"{100 * x:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s.replace("-", "$-$")


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/experiments/exp2.yaml").read_text(encoding="utf-8"))
    holm = {(r["test"], r["dataset"]): r for r in
            json.loads((ROOT / "results/final_analysis/primary/holm_family.json").read_text(encoding="utf-8"))["results"]}
    out = {}
    for d in DATASETS:
        labels = YamlLabelRepository(ROOT / "data/labels").load(d)
        load = lambda run, m, s: read_predictions(ROOT / "results" / run, m, d, s)  # noqa: E731
        a_dev, b_dev, _ = aligned_matrices(load("final_jev", "jev", "dev"), load("final_train", "deberta-v3-large", "dev"), labels)
        a, b, _ = aligned_matrices(load("final_jev", "jev", "test"), load("final_train", "deberta-v3-large", "test"), labels)
        counts = json.loads((ROOT / "data/processed" / d / "report.json").read_text(encoding="utf-8"))["label_counts"]["train"]
        tp = np.array([counts.get(lid, 0) for lid in labels.ids()], dtype=float)
        tp /= tp.sum()
        n = len(a.gold)
        q0 = np.bincount(a.gold, minlength=len(tp)) / n
        tau_a = ls.coverage_threshold(a_dev.probs.max(axis=1), cfg["coverage"])
        tau_b = ls.coverage_threshold(b_dev.probs.max(axis=1), cfg["coverage"])
        pi_a = ls.implicit_prior(a_dev.probs)
        base = np.arange(n)
        ra0, rb0 = _risk(a, base, tau_a), _risk(b, base, tau_b)
        rng = np.random.default_rng(SEED)
        v = {"PA1": [], "PA2": [], "PA3": []}
        for _ in range(N_PRIORS):
            q = ls.dirichlet_target_prior(tp, 1.0, rng)
            s = ls.resample_with_replacement(a.gold, q, n, rng)
            c = ls.resample_with_replacement(a.gold, q0, n, rng)
            ea, eb = _ece(a, s), _ece(b, s)
            v["PA1"].append((ea - _ece(a, c)) - (eb - _ece(b, c)))
            v["PA2"].append((_risk(a, s, tau_a) - ra0) - (_risk(b, s, tau_b) - rb0))
            em, _, _ = ls.em_prior_adjust(a.probs[s], a.mask[s], pi_a)
            v["PA3"].append(_ece(a, s, em) - ea)
        out[d] = {t: {"mean": float(np.mean(x)), "se": float(np.std(x) / np.sqrt(len(x))), "sd": float(np.std(x)),
                      "median": float(np.median(x))} for t, x in v.items()}
        print(d, {t: f"{100 * o['mean']:+.2f} (SE {100 * o['se']:.2f}, SD {100 * o['sd']:.1f})" for t, o in out[d].items()})
    (ROOT / "results/final_analysis/pa_population.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    lines = ["\\begin{tabular}{ll" + "c" * len(DATASETS) + "}", "\\toprule",
             "Test & Estimate & " + " & ".join(DS[d] for d in DATASETS) + " \\\\", "\\midrule"]
    for k, t in enumerate(("PA1", "PA2", "PA3")):
        if k:
            lines.append("\\midrule")
        frozen = []
        for d in DATASETS:
            f = holm[(t, d)]
            frozen.append(("\\textbf{" + pts(f["estimate"]) + "}$^\\dagger$") if f["p_holm"] < 0.05 else pts(f["estimate"]))
        lines.append(f"{t} & frozen 50 priors & " + " & ".join(frozen) + " \\\\")
        lines.append(" & 2{,}000 priors: mean (SE) & " + " & ".join(
            f"{pts(out[d][t]['mean'])} ({100 * out[d][t]['se']:.2f})" for d in DATASETS) + " \\\\")
        lines.append(" & SD across priors & " + " & ".join(f"{100 * out[d][t]['sd']:.1f}" for d in DATASETS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tables/tab_pa_population.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
