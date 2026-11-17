"""Exploratory (added after the results): seed sensitivity of the Jev - DeBERTa shift comparison. The confirmatory
tests use DeBERTa-v3-large seed 1001; this runs the descriptive shift experiment (Exp 2, same code, configuration and
random numbers as `s1bench exp2`) for all five full-data seeds and compares the shift-induced change in ECE and the
excess risk of the 80%-coverage threshold at alpha = 1 with Jev's.

Check: the rows for seed 1001 must equal results/final_train/exp2/deberta-v3-large/<dataset>.jsonl exactly.
Writes results/final_analysis_debseeds/<model>/<dataset>.jsonl (skipped if present) and
paper/tables/tab_deberta_seed_shift.tex.

Run: .venv/Scripts/python scripts/paper/deberta_seed_shift.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from s1bench.application.prevalence_shift import PrevalenceShift, to_matrices  # noqa: E402
from s1bench.infrastructure.storage.processed_store import JsonlProcessedReader, YamlLabelRepository  # noqa: E402
from s1bench.infrastructure.storage.results_store import read_predictions  # noqa: E402

RES = ROOT / "results"
OUT = RES / "final_analysis_debseeds"
DATASETS = ["clinc150", "hwu64", "fpb", "hatexplain", "trec6"]
DS = {"clinc150": "CLINC150", "hwu64": "HWU64", "fpb": "FPB", "hatexplain": "HateXplain", "trec6": "TREC-6"}
SEEDS = ["deberta-v3-large"] + [f"deberta-v3-large@fulls{s}" for s in (1002, 1003, 1004, 1005)]


def exp2_rows(model: str, dataset: str) -> list[dict]:
    """Same steps as `s1bench exp2` (cli/main.py) for a trained classifier (training prior as EM source prior)."""
    path = OUT / model / f"{dataset}.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / "exp2.yaml").read_text(encoding="utf-8"))
    labels = YamlLabelRepository(ROOT / "data" / "labels").load(dataset)
    report = json.loads((ROOT / "data" / "processed" / dataset / "report.json").read_text(encoding="utf-8"))
    counts = report["label_counts"]["train"]
    train_prior = np.array([counts.get(lid, 0) for lid in labels.ids()], dtype=float)
    train_prior /= train_prior.sum()
    kind = cfg["source_prior"].get("deberta-v3-large", cfg["source_prior"]["default"])
    exp = PrevalenceShift(cfg["alphas"], cfg["draws_per_alpha"], cfg["seed"], cfg["coverage"],
                          cfg["em"]["max_iter"], cfg["em"]["tol"])
    index = {lid: i for i, lid in enumerate(labels.ids())}
    run = RES / "final_train"
    dev_m = to_matrices(read_predictions(run, model, dataset, "dev"), labels)
    test_m = to_matrices(read_predictions(run, model, dataset, "test"), labels)
    full = JsonlProcessedReader(ROOT / "data" / "processed").read_split(dataset, "test")
    full_test = ([e.id for e in full], np.array([index[e.label] for e in full]))
    rows, _ = exp.run(dev_m, test_m, train_prior, kind, full_test=full_test)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    # read back, so that NaN risks (no example accepted) compare equal with the stored rows, as both come from JSON
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def shift_effects(rows: list[dict]) -> tuple[float, float]:
    """(mean ECE at alpha=1 minus mean ECE of the resampled no-shift control, mean excess risk at alpha=1), arm none,
    with-replacement draws, in points."""
    sel = lambda a: [r for r in rows if r["arm"] == "none" and r["alpha"] == a and r["sampling"] == "with_replacement"]
    d_ece = np.mean([r["ece"] for r in sel("1")]) - np.mean([r["ece"] for r in sel("noshift_resampled")])
    exc = np.mean([r["excess_risk"] for r in sel("1") if r["excess_risk"] is not None])
    return 100 * float(d_ece), 100 * float(exc)


def main() -> None:
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             " & \\multicolumn{3}{c}{$\\Delta$ECE under shift} & \\multicolumn{3}{c}{Excess risk} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Dataset & Jev & DeBERTa, seed 1001 & DeBERTa, 5 seeds & Jev & DeBERTa, seed 1001 & DeBERTa, 5 seeds \\\\",
             "\\midrule"]
    for ds in DATASETS:
        stored = [json.loads(line) for line in
                  (RES / "final_train" / "exp2" / "deberta-v3-large" / f"{ds}.jsonl").read_text(encoding="utf-8").splitlines()]
        if exp2_rows("deberta-v3-large", ds) != stored:
            raise SystemExit(f"{ds}: recomputed seed-1001 rows differ from the stored Exp 2 rows")
        jev = shift_effects([json.loads(line) for line in
                             (RES / "final_jev" / "exp2" / "jev" / f"{ds}.jsonl").read_text(encoding="utf-8").splitlines()])
        deb = [shift_effects(exp2_rows(m, ds)) for m in SEEDS]
        e = [d[0] for d in deb]
        x = [d[1] for d in deb]
        print(f"{ds:10s} dECE jev {jev[0]:+.2f} deb {e[0]:+.2f} [{min(e):+.2f},{max(e):+.2f}] | "
              f"excess jev {jev[1]:+.2f} deb {x[0]:+.2f} [{min(x):+.2f},{max(x):+.2f}]")
        f = lambda v: f"{v:+.2f}".replace("-", "$-$")
        lines.append(f"{DS[ds]} & {f(jev[0])} & {f(e[0])} & {f(min(e))} to {f(max(e))} & {f(jev[1])} & {f(x[0])} & "
                     f"{f(min(x))} to {f(max(x))} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper" / "tables" / "tab_deberta_seed_shift.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
