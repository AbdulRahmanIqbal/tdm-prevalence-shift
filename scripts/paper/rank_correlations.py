"""DESCRIPTIVE (post hoc): do accuracy, MCC and NLL rank the seven analysed systems as Macro-F1 does?

Spearman rho between Macro-F1 and each other metric across the seven systems of the main comparison (NLL negated so
that higher is better), per dataset, from the no-shift test results. Writes results/final_analysis/rank_correlations.json.
Run: .venv/Scripts/python scripts/paper/rank_correlations.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = ["jev", "tev1-4b", "nimble-9b", "qwen3.5-4b", "qwen3.5-9b", "deberta-v3-large", "tfidf-lr"]
DATASETS = ["clinc150", "hwu64", "fpb", "hatexplain", "trec6"]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    rows = json.loads((ROOT / "results/final_analysis/exp1_test.json").read_text(encoding="utf-8"))
    rows += json.loads((ROOT / "results/final_analysis/exp1_test_jev.json").read_text(encoding="utf-8"))
    by = {(r["model"], r["dataset"]): r for r in rows}
    out = {}
    for d in DATASETS:
        f1 = np.array([by[(m, d)]["macro_f1"] for m in SYSTEMS])
        out[d] = {k: spearman(f1, sign * np.array([by[(m, d)][k] for m in SYSTEMS]))
                  for k, sign in (("accuracy", 1), ("mcc", 1), ("nll", -1))}
    (ROOT / "results/final_analysis/rank_correlations.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for d, v in out.items():
        print(d, {k: round(x, 2) for k, x in v.items()})


if __name__ == "__main__":
    main()
