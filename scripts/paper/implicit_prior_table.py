"""Descriptive: distance between each zero-shot system's implicit prior (mean development-set prediction, smoothed as in
Exp 2) and the true development-set class distribution. Reads final dev predictions only.
Writes paper/tables/tab_implicit_prior.tex and results/final_analysis/implicit_prior_tv.json.

Run: .venv/Scripts/python scripts/paper/implicit_prior_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from s1bench.application.prevalence_shift import to_matrices  # noqa: E402
from s1bench.domain import label_shift as ls  # noqa: E402
from s1bench.infrastructure.storage.processed_store import YamlLabelRepository  # noqa: E402
from s1bench.infrastructure.storage.results_store import read_predictions  # noqa: E402

DATASETS = ["clinc150", "hwu64", "fpb", "hatexplain", "trec6"]
DS = {"clinc150": "CLINC150", "hwu64": "HWU64", "fpb": "FPB", "hatexplain": "HateXplain", "trec6": "TREC-6"}
SYSTEMS = [("jev", "final_jev", "Jev"), ("tev1-4b", "final_local", "Tev1-4B"), ("nimble-9b", "final_local", "Nimble-9B"),
           ("qwen3.5-4b", "final_local", "Qwen3.5-4B"), ("qwen3.5-9b", "final_local", "Qwen3.5-9B")]


def main() -> None:
    tv = {}
    for d in DATASETS:
        labels = YamlLabelRepository(ROOT / "data" / "labels").load(d)
        for m, run, _ in SYSTEMS:
            dev = to_matrices(read_predictions(ROOT / "results" / run, m, d, "dev"), labels)
            true = np.bincount(dev.gold, minlength=dev.probs.shape[1]) / len(dev.gold)
            tv[(m, d)] = float(0.5 * np.abs(ls.implicit_prior(dev.probs) - true).sum())
    lines = ["\\begin{tabular}{l" + "r" * len(DATASETS) + "}", "\\toprule",
             "System & " + " & ".join(DS[d] for d in DATASETS) + " \\\\", "\\midrule"]
    for m, _, lab in SYSTEMS:
        lines.append(f"{lab} & " + " & ".join(f"{tv[(m, d)]:.3f}" for d in DATASETS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tables/tab_implicit_prior.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "results/final_analysis/implicit_prior_tv.json").write_text(
        json.dumps({f"{m}|{d}": v for (m, d), v in tv.items()}, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
