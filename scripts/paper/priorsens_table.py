"""EXPLORATORY post-hoc prior-set sensitivity for PA1-PA3 (added after the confirmatory results were seen; outside Holm).

The confirmatory PA estimand is conditional on one fixed set of 50 target priors (seed 20261006, B = 10,000). Here the
same test is rerun with five other prior sets (seeds 1-5, configs/experiments/sensitivity/, B = 1,000). Reads
results/final_analysis/primary/holm_family.json and results/final_analysis_priorsens_s{1..5}/. Writes
paper/tables/tab_pa_sensitivity.tex and results/final_analysis/pa_prior_sensitivity.json.

Run: .venv/Scripts/python scripts/paper/priorsens_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["clinc150", "hwu64", "fpb", "hatexplain", "trec6"]
DS = {"clinc150": "CLINC150", "hwu64": "HWU64", "fpb": "FPB", "hatexplain": "HateXplain", "trec6": "TREC-6"}
SEEDS = [1, 2, 3, 4, 5]


def pts(x: float) -> str:
    s = f"{100 * x:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s.replace("-", "$-$")


def main() -> None:
    holm = {(r["test"], r["dataset"]): r for r in
            json.loads((ROOT / "results/final_analysis/primary/holm_family.json").read_text(encoding="utf-8"))["results"]}
    sens: dict = {}
    for s in SEEDS:
        for d in DATASETS:
            j = json.loads((ROOT / f"results/final_analysis_priorsens_s{s}/primary/pa_jev_vs_deberta-v3-large_{d}.json")
                           .read_text(encoding="utf-8"))
            if j["b"] != 1000 or j["n_priors"] != 50 or j["dropped_test"] != 0:
                raise SystemExit(f"unexpected settings in seed {s}, {d}")
            for r in j["results"]:
                sens[(r["test"], d, s)] = r
    summary = {}
    lines = ["\\begin{tabular}{ll" + "c" * len(DATASETS) + "}", "\\toprule",
             "Test & Prior set & " + " & ".join(DS[d] for d in DATASETS) + " \\\\", "\\midrule"]
    for k, t in enumerate(("PA1", "PA2", "PA3")):
        if k:
            lines.append("\\midrule")
        row = []
        for d in DATASETS:
            f = holm[(t, d)]
            row.append(("\\textbf{" + pts(f["estimate"]) + "}$^\\dagger$") if f["p_holm"] < 0.05 else pts(f["estimate"]))
        lines.append(f"{t} & frozen (confirmatory) & " + " & ".join(row) + " \\\\")
        rng_cells, excl_cells = [], []
        for d in DATASETS:
            est = [sens[(t, d, s)]["estimate"] for s in SEEDS]
            excl = sum(sens[(t, d, s)]["ci_low"] > 0 or sens[(t, d, s)]["ci_high"] < 0 for s in SEEDS)
            same = sum((e > 0) == (holm[(t, d)]["estimate"] > 0) for e in est)
            summary[f"{t}|{d}"] = {"frozen": holm[(t, d)]["estimate"], "seeds": est, "ci_excludes_zero": excl,
                                   "same_sign_as_frozen": same}
            rng_cells.append(f"[{pts(min(est))}, {pts(max(est))}]")
            excl_cells.append(f"{excl}/5")
        lines.append(" & 5 other sets: range & " + " & ".join(rng_cells) + " \\\\")
        lines.append(" & CI excludes 0 & " + " & ".join(excl_cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tables/tab_pa_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "results/final_analysis/pa_prior_sensitivity.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                                           encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
