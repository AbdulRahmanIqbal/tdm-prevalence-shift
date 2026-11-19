"""EXPLORATORY (post hoc): why does the confirmatory PA estimate differ from the Exp 2 (200-draw) analogue on FPB?

Hypothesis: the PA estimate is the MEAN over bootstrap replicates of a statistic computed on samples that are resampled
twice (class-stratified bootstrap base, then shifted samples drawn from that base). Top-label ECE is biased upward on
duplicated data, and the duplication grows with shift strength, so double resampling can inflate the difference.

Check: on the frozen 50 priors (same seed and draw order as primary_tests.pa_tests), compute the same PA1/PA2/PA3
statistics WITHOUT the bootstrap base (base = the full test set), averaging over R repeated single-level resamples.
If these plug-in values match the Exp 2 analogue rather than the PA estimate, the gap is procedural.

Writes results/final_analysis/pa_procedure_check.json.
Run: .venv/Scripts/python scripts/paper/pa_procedure_check.py [datasets...]
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

R = 20  # repeated single-level resamples per prior


def main(datasets: list[str]) -> None:
    cfg = yaml.safe_load((ROOT / "configs/experiments/exp2.yaml").read_text(encoding="utf-8"))
    out = {}
    for d in datasets:
        labels = YamlLabelRepository(ROOT / "data/labels").load(d)
        load = lambda run, m, s: read_predictions(ROOT / "results" / run, m, d, s)  # noqa: E731
        a_dev, b_dev, _ = aligned_matrices(load("final_jev", "jev", "dev"), load("final_train", "deberta-v3-large", "dev"), labels)
        a_t, b_t, _ = aligned_matrices(load("final_jev", "jev", "test"), load("final_train", "deberta-v3-large", "test"), labels)
        rep = json.loads((ROOT / "data/processed" / d / "report.json").read_text(encoding="utf-8"))
        counts = rep["label_counts"]["train"]
        train_prior = np.array([counts.get(lid, 0) for lid in labels.ids()], dtype=float)
        train_prior /= train_prior.sum()
        n = len(a_t.gold)
        rng = np.random.default_rng(int(cfg["seed"]))
        priors = [ls.dirichlet_target_prior(train_prior, 1.0, rng) for _ in range(50)]   # identical to pa_tests
        tau_a = ls.coverage_threshold(a_dev.probs.max(axis=1), cfg["coverage"])
        tau_b = ls.coverage_threshold(b_dev.probs.max(axis=1), cfg["coverage"])
        pi_a = ls.implicit_prior(a_dev.probs)
        base = np.arange(n)
        q0 = np.bincount(a_t.gold, minlength=len(train_prior)) / n
        risk_a0, risk_b0 = _risk(a_t, base, tau_a), _risk(b_t, base, tau_b)
        d1, d2, d3 = [], [], []
        rng2 = np.random.default_rng(12345)
        for q in priors:
            for _ in range(R):
                s_idx = ls.resample_with_replacement(a_t.gold, q, n, rng2)
                n_idx = ls.resample_with_replacement(a_t.gold, q0, n, rng2)
                ea, eb = _ece(a_t, s_idx), _ece(b_t, s_idx)
                d1.append((ea - _ece(a_t, n_idx)) - (eb - _ece(b_t, n_idx)))
                d2.append((_risk(a_t, s_idx, tau_a) - risk_a0) - (_risk(b_t, s_idx, tau_b) - risk_b0))
                em, _, _ = ls.em_prior_adjust(a_t.probs[s_idx], a_t.mask[s_idx], pi_a)
                d3.append(_ece(a_t, s_idx, em) - ea)
        out[d] = {"pa1_plugin": float(np.mean(d1)), "pa2_plugin": float(np.mean(d2)), "pa3_plugin": float(np.mean(d3)),
                  "R_per_prior": R}
        print(d, {k: round(100 * v, 2) for k, v in out[d].items() if k != "R_per_prior"})
    p = ROOT / "results/final_analysis/pa_procedure_check.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    old.update(out)
    p.write_text(json.dumps(old, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["fpb", "hatexplain", "trec6"])
