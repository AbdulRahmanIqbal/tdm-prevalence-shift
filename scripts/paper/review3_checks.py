"""EXPLORATORY / DESCRIPTIVE checks added after the results were known (third external review).

(1) EM without shift: ECE change from EM (implicit prior for zero-shot systems, training prior for trained ones) on the
    resampled no-shift control and on the original test set, next to the change at alpha = 1 (raw final Exp 2 rows).
(2) FPB confusion matrices for Nimble-9B and Qwen3.5-9B (final test predictions; rows gold, columns predicted).

Writes results/final_analysis/review3_checks.json and paper/tables/tab_em_noshift.tex, tab_fpb_confusion.tex.
Run: .venv/Scripts/python scripts/paper/review3_checks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
from exp2_descriptives import DATASETS, DS, SYSTEMS, load, mean, p  # noqa: E402

OUT = ROOT / "paper" / "tables"
FPB_LABELS = ["negative", "neutral", "positive"]   # ids 0, 1, 2 (data/labels/fpb.yaml)


def em_noshift() -> dict:
    res = {}
    for m, run, _ in SYSTEMS:
        for d in DATASETS:
            r = load(m, run, d)
            ns = mean(r, "noshift_resampled", "em") - mean(r, "noshift_resampled", "none")
            orig = mean(r, "inf", "em", sampling="original") - mean(r, "inf", "none", sampling="original")
            a1 = mean(r, "1", "em") - mean(r, "1", "none")
            res[f"{m}|{d}"] = {"noshift_resampled": ns, "original": orig, "alpha1": a1}
    return res


def confusion(model: str) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    with (ROOT / "results" / "final_local" / "predictions" / model / "fpb_test.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["status"] == "ok":
                cm[int(r["gold"]), int(r["predicted"])] += 1
    return cm


def main() -> None:
    em = em_noshift()
    lines = ["\\begin{tabular}{l" + "rr" * 5 + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS[d]}}}" for d in DATASETS) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(5)),
             "System & " + " & ".join(["no shift & $\\alpha$=1"] * 5) + " \\\\", "\\midrule"]
    for m, _, lab in SYSTEMS:
        cells = []
        for d in DATASETS:
            v = em[f"{m}|{d}"]
            cells += [p(v["noshift_resampled"]), p(v["alpha1"])]
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_em_noshift.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cms = {m: confusion(m) for m in ("qwen3.5-9b", "nimble-9b")}
    lines = ["\\begin{tabular}{l" + "rrr" * 2 + "}", "\\toprule",
             " & \\multicolumn{3}{c}{Qwen3.5-9B predicted} & \\multicolumn{3}{c}{Nimble-9B predicted} \\\\",
             "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
             "Gold & " + " & ".join(FPB_LABELS * 2) + " \\\\", "\\midrule"]
    for i, g in enumerate(FPB_LABELS):
        lines.append(f"{g} ({cms['qwen3.5-9b'][i].sum()}) & "
                     + " & ".join(str(x) for x in cms["qwen3.5-9b"][i]) + " & "
                     + " & ".join(str(x) for x in cms["nimble-9b"][i]) + " \\\\")
    lines += ["\\midrule", "Total predicted & " + " & ".join(str(x) for x in cms["qwen3.5-9b"].sum(0)) + " & "
              + " & ".join(str(x) for x in cms["nimble-9b"].sum(0)) + " \\\\", "\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_fpb_confusion.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = {"em_effect": em, "fpb_confusion": {m: c.tolist() for m, c in cms.items()}, "labels": FPB_LABELS}
    (ROOT / "results" / "final_analysis" / "review3_checks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for m in ("jev", "tev1-4b", "nimble-9b", "qwen3.5-4b", "qwen3.5-9b", "deberta-v3-large"):
        print(m, {d: {k: round(100 * v, 2) for k, v in em[f"{m}|{d}"].items()} for d in ("fpb", "hatexplain", "trec6")})
    for m, c in cms.items():
        print(m, "\n", c, "recall", np.round(np.diag(c) / c.sum(1), 3))


if __name__ == "__main__":
    main()
