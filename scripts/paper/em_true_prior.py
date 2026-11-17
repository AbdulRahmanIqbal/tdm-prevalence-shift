"""Exploratory (added after the results): EM prior correction of the zero-shot systems with the implicit prior (the
paper's arm) versus the dataset's true class prior (the training-split class distribution, which is also the centre of
the shift distribution). Same Exp 2 draws at alpha = 1 (with replacement, 200 draws), raw probabilities, EM settings of
configs/experiments/exp2.yaml.

Check: the regenerated 'none' and 'em' (implicit prior) arms must reproduce the stored Exp 2 rows (top-label ECE of
every draw). Writes results/final_analysis_calibrob/em_true_prior.json and paper/tables/tab_em_true_prior.tex.
Run: .venv/Scripts/python scripts/paper/em_true_prior.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
from calibration_robustness import DATASETS, DS, OUT, RES, RUN, matrices  # noqa: E402
from s1bench.domain import label_shift as ls  # noqa: E402
from s1bench.infrastructure.storage.processed_store import JsonlProcessedReader  # noqa: E402

SYSTEMS = [("jev", "Jev"), ("tev1-4b", "Tev1-4B"), ("nimble-9b", "Nimble-9B"), ("qwen3.5-4b", "Qwen3.5-4B"),
           ("qwen3.5-9b", "Qwen3.5-9B")]
ARMS = [("none", "none"), ("em", "EM, implicit prior"), ("em_true", "EM, true prior")]


def run(model: str, dataset: str) -> dict[str, dict[str, float]]:
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / "exp2.yaml").read_text(encoding="utf-8"))
    em = (cfg["em"]["max_iter"], cfg["em"]["tol"])
    test, labels = matrices(model, dataset, "test")
    dev, _ = matrices(model, dataset, "dev")
    report = json.loads((ROOT / "data" / "processed" / dataset / "report.json").read_text(encoding="utf-8"))
    counts = report["label_counts"]["train"]
    train_prior = np.array([counts.get(lid, 0) for lid in labels.ids()], dtype=float)
    train_prior /= train_prior.sum()
    pi_implicit = ls.implicit_prior(dev.probs)
    index = {lid: i for i, lid in enumerate(labels.ids())}
    full = JsonlProcessedReader(ROOT / "data" / "processed").read_split(dataset, "test")
    full_gold = np.array([index[e.label] for e in full])
    row_of = {eid: r for r, eid in enumerate(test.ids)}
    to_row = np.array([row_of[e.id] for e in full])
    stored = {(r["arm"], r["draw"]): r for r in
              (json.loads(line) for line in
               (RES / RUN[model] / "exp2" / model / f"{dataset}.jsonl").read_text(encoding="utf-8").splitlines())
              if r["alpha"] == "1" and r["sampling"] == "with_replacement"}
    rng = np.random.default_rng(cfg["seed"])
    res = {a: {"ece": [], "macro_f1": []} for a, _ in ARMS}
    for alpha in cfg["alphas"]:
        for d in range(1, cfg["draws_per_alpha"] + 1):
            target = ls.dirichlet_target_prior(train_prior, alpha, rng)
            idx = to_row[ls.resample_with_replacement(full_gold, target, len(full_gold), rng)]
            ls.resample_without_replacement(full_gold, target, rng)   # consumed in Exp 2
            if str(alpha) != "1":
                continue
            p, mk, g, pos = test.probs[idx], test.mask[idx], test.gold[idx], test.positions[idx]
            arms = {"none": p, "em": ls.em_prior_adjust(p, mk, pi_implicit, *em)[0],
                    "em_true": ls.em_prior_adjust(p, mk, train_prior, *em)[0]}
            for a, probs in arms.items():
                mm = ls.matrix_metrics(probs, g, positions=pos)
                if a != "em_true" and round(mm.ece, 6) != stored[(a, d)]["ece"]:
                    raise SystemExit(f"{model}/{dataset} arm {a} draw {d}: does not reproduce Exp 2")
                res[a]["ece"].append(mm.ece)
                res[a]["macro_f1"].append(mm.macro_f1)
    return {a: {k: float(np.mean(v)) for k, v in m.items()} for a, m in res.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = {f"{m}|{d}": run(m, d) for m, _ in SYSTEMS for d in DATASETS}
    (OUT / "em_true_prior.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    lines = ["\\begin{tabular}{ll" + "r" * len(DATASETS) + "}", "\\toprule",
             "System & ECE under shift & " + " & ".join(DS[d] for d in DATASETS) + " \\\\", "\\midrule"]
    for m, name in SYSTEMS:
        for j, (a, label) in enumerate(ARMS):
            shade = "\\rowcolor{rowhl} " if a == "em" else ""   # the paper's arm (EM with the implicit prior) shaded
            lines.append(f"{shade}{name if j == 0 else ''} & {label} & "
                         + " & ".join(f"{out[f'{m}|{d}'][a]['ece']:.3f}" for d in DATASETS) + " \\\\")
            print(f"{m:11s} {label:20s} " + " ".join(f"{out[f'{m}|{d}'][a]['ece']:.3f}/{out[f'{m}|{d}'][a]['macro_f1']:.3f}"
                                                   for d in DATASETS))
        lines.append("\\midrule" if m != SYSTEMS[-1][0] else "\\bottomrule")
    lines.append("\\end{tabular}")
    (ROOT / "paper" / "tables" / "tab_em_true_prior.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
