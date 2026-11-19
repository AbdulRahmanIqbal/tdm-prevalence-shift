"""Descriptive Exp 2 summaries for the paper appendix (no new inference; reads final Exp 2 rows and final predictions).

Writes paper/tables/tab_exp2_delta.tex, tab_exp2_arms.tex, tab_pa_vs_exp2.tex and
results/final_analysis/exp2_descriptives.json, and prints the numbers quoted in the text.

Run: .venv/Scripts/python scripts/paper/exp2_descriptives.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from s1bench.domain import metrics as mt  # noqa: E402

OUT = ROOT / "paper" / "tables"
DATASETS = ["clinc150", "hwu64", "fpb", "hatexplain", "trec6"]
DS = {"clinc150": "CLINC150", "hwu64": "HWU64", "fpb": "FPB", "hatexplain": "HateXplain", "trec6": "TREC-6"}
SYSTEMS = [("jev", "final_jev", "Jev"), ("tev1-4b", "final_local", "Tev1-4B"), ("nimble-9b", "final_local", "Nimble-9B"),
           ("qwen3.5-4b", "final_local", "Qwen3.5-4B"), ("qwen3.5-9b", "final_local", "Qwen3.5-9B"),
           ("deberta-v3-large", "final_train", "DeBERTa-v3-large"), ("tfidf-lr", "final_train", "TF-IDF + LR")]
ARMS = [("none", "none"), ("temperature", "temp."), ("em", "EM"), ("bcts_em", "BCTS+EM"), ("oracle", "oracle")]


def load(model: str, run: str, ds: str, stem: str | None = None) -> dict:
    rows = defaultdict(list)
    with (ROOT / "results" / run / "exp2" / model / f"{stem or ds}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[(r["alpha"], r["sampling"], r["arm"])].append(r)
    return rows


def mean(rows, alpha, arm, key="ece", sampling="with_replacement"):
    v = [r[key] for r in rows[(alpha, sampling, arm)] if r[key] is not None]
    return float(np.mean(v)) if v else float("nan")


def p(x: float) -> str:
    """points, signed, 1 decimal, proper minus."""
    s = f"{100 * x:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s.replace("-", "$-$")


def main() -> None:
    data = {(m, ds): load(m, run, ds) for m, run, _ in SYSTEMS for ds in DATASETS}
    out: dict = {}

    # (1) Delta ECE vs resampled no-shift control, raw probabilities, alpha 1 and 0.1, all systems
    delta = {}
    lines = ["\\begin{tabular}{l" + "rr" * 5 + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS[d]}}}" for d in DATASETS) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(5)),
             "System & " + " & ".join(["$\\alpha$=1 & 0.1"] * 5) + " \\\\", "\\midrule"]
    for m, _, lab in SYSTEMS:
        cells = []
        for d in DATASETS:
            r = data[(m, d)]
            base = mean(r, "noshift_resampled", "none")
            d1, d01 = mean(r, "1", "none") - base, mean(r, "0.1", "none") - base
            delta[(m, d)] = (d1, d01)
            cells += [p(d1), p(d01)]
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_exp2_delta.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # (2) ECE by arm at alpha = 1, all systems
    lines = ["\\begin{tabular}{ll" + "r" * len(ARMS) + "}", "\\toprule",
             "Dataset & System & " + " & ".join(lab for _, lab in ARMS) + " \\\\", "\\midrule"]
    arms_tab = {}
    for i, d in enumerate(DATASETS):
        if i:
            lines.append("\\midrule")
        for j, (m, _, lab) in enumerate(SYSTEMS):
            vals = [mean(data[(m, d)], "1", a) for a, _ in ARMS]
            arms_tab[(m, d)] = vals
            best = int(np.argmin(vals))
            cells = [(f"\\textbf{{{v:.3f}}}" if k == best else f"{v:.3f}") for k, v in enumerate(vals)]
            lines.append(f"{DS[d] if j == 0 else ''} & {lab} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_exp2_arms.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # (3) Exp 2 (200 draws) analogue of PA1-PA3 next to the PA estimates
    holm = {(r["test"], r["dataset"]): r for r in
            json.loads((ROOT / "results/final_analysis/primary/holm_family.json").read_text(encoding="utf-8"))["results"]}
    lines = ["\\begin{tabular}{l" + "c" * 5 + "}", "\\toprule",
             "Quantity (points) & " + " & ".join(DS[d] for d in DATASETS) + " \\\\", "\\midrule"]
    analog = {}
    for d in DATASETS:
        j, b = data[("jev", d)], data[("deberta-v3-large", d)]
        pa1 = (mean(j, "1", "none") - mean(j, "noshift_resampled", "none")) - \
              (mean(b, "1", "none") - mean(b, "noshift_resampled", "none"))
        pa2 = mean(j, "1", "none", "excess_risk") - mean(b, "1", "none", "excess_risk")
        pa3 = mean(j, "1", "em") - mean(j, "1", "none")
        analog[d] = (pa1, pa2, pa3)
    for t, k, name in (("PA1", 0, "PA1 ($\\Delta$ECE, Jev $-$ DeBERTa)"), ("PA2", 1, "PA2 (excess risk, Jev $-$ DeBERTa)"),
                       ("PA3", 2, "PA3 (ECE, Jev EM $-$ none)")):
        lines.append(f"{name}: 50 fixed priors & " + " & ".join(p(holm[(t, d)]["estimate"]) for d in DATASETS) + " \\\\")
        lines.append(f"\\quad 200 descriptive draws &" + " & ".join(p(analog[d][k]) for d in DATASETS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_pa_vs_exp2.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # (4) per-draw excess risk at alpha = 1 (raw)
    ex = {}
    for m, _, _ in SYSTEMS:
        for d in DATASETS:
            v = np.array([r["excess_risk"] for r in data[(m, d)][("1", "with_replacement", "none")]
                          if r["excess_risk"] is not None])
            ex[(m, d)] = (float(np.mean(v)), float(np.percentile(v, 95)), float(v.max()))
    worst = max(ex.items(), key=lambda kv: kv[1][2])
    # (5) resampling inflation: noshift_resampled vs original (inf), raw
    infl = {(m, d): mean(data[(m, d)], "noshift_resampled", "none") - mean(data[(m, d)], "inf", "none", sampling="original")
            for m, _, _ in SYSTEMS for d in DATASETS}
    # (6) without replacement: sign agreement of Jev - DeBERTa Delta ECE at alpha = 1 with the with-replacement value
    worep = {}
    for d in DATASETS:
        j, b = data[("jev", d)], data[("deberta-v3-large", d)]
        w = (mean(j, "1", "none", sampling="without_replacement") - mean(j, "inf", "none", sampling="original")) - \
            (mean(b, "1", "none", sampling="without_replacement") - mean(b, "inf", "none", sampling="original"))
        n_small = float(np.mean([r["n"] < 200 for r in j[("1", "without_replacement", "none")]]))
        worep[d] = (w, n_small)
    # (7) Jev across-run ECE (500-example repeats)
    rep_ece = {}
    for d in DATASETS:
        es = []
        for run in ("final_jev_rep1", "final_jev_rep2", "final_jev_rep3"):
            rows = [json.loads(x) for x in (ROOT / "results" / run / "predictions/jev" / f"{d}_test.jsonl").open(encoding="utf-8")]
            sc = [mt.Scored(tuple(r["candidates"]), tuple(v / sum(r["probabilities"]) for v in r["probabilities"]), r["gold"])
                  for r in rows if r["status"] == "ok"]
            es.append(mt.ece(sc))
        rep_ece[d] = (min(es), max(es))
    # (8) Jev retries in the full run
    retries = sum(json.loads(x)["raw"].get("attempts", 1) > 1
                  for f in (ROOT / "results/final_jev/predictions/jev").glob("*.jsonl") for x in f.open(encoding="utf-8"))
    # (9) EPS sensitivity for Jev: temperature arm ECE at alpha 1 for eps 1e-3 / default / 1e-12
    eps = {d: tuple(mean(load("jev", "final_jev", d, stem), "1", "temperature")
                    for stem in (f"{d}_eps0.001", d, f"{d}_eps1e-12")) for d in DATASETS}

    # (10) per-draw excess risk table (alpha = 1, raw): mean / 95th percentile / max, points
    lines = ["\\begin{tabular}{l" + "c" * 5 + "}", "\\toprule",
             "System & " + " & ".join(DS[d] for d in DATASETS) + " \\\\", "\\midrule"]
    for m, _, lab in SYSTEMS:
        lines.append(f"{lab} & " + " & ".join(
            f"{p(ex[(m, d)][0])} / {100 * ex[(m, d)][1]:.1f} / {100 * ex[(m, d)][2]:.1f}" for d in DATASETS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_excess_risk.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # (11) Brier and NLL change vs the resampled no-shift control (raw), alpha 1 and 0.1
    for key, name in (("brier", "tab_exp2_brier"), ("nll", "tab_exp2_nll")):
        lines = ["\\begin{tabular}{l" + "rr" * 5 + "}", "\\toprule",
                 " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS[d]}}}" for d in DATASETS) + " \\\\",
                 " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(5)),
                 "System & " + " & ".join(["$\\alpha$=1 & 0.1"] * 5) + " \\\\", "\\midrule"]
        for m, _, lab in SYSTEMS:
            cells = []
            for d in DATASETS:
                r = data[(m, d)]
                base = mean(r, "noshift_resampled", "none", key)
                cells += [p(mean(r, "1", "none", key) - base), p(mean(r, "0.1", "none", key) - base)]
            lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = {"delta_ece": {f"{m}|{d}": v for (m, d), v in delta.items()},
           "pa_analog_200": analog, "excess_risk_alpha1": {f"{m}|{d}": v for (m, d), v in ex.items()},
           "resampling_inflation": {f"{m}|{d}": v for (m, d), v in infl.items()},
           "without_replacement_pa1_analog": worep, "jev_repeat_ece_range": rep_ece, "jev_retried_calls": retries,
           "jev_temperature_ece_eps": eps}
    (ROOT / "results/final_analysis/exp2_descriptives.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("TF-IDF vs Jev Delta ECE at 0.1:", {d: (round(100 * delta[("tfidf-lr", d)][1], 2), round(100 * delta[("jev", d)][1], 2)) for d in DATASETS})
    print("BCTS+EM shift change (own control) HateXplain:",
          {m: round(100 * (mean(data[(m, 'hatexplain')], "1", "bcts_em") - mean(data[(m, 'hatexplain')], "noshift_resampled", "bcts_em")), 2) for m, _, _ in SYSTEMS})
    print("EM - none at alpha 1 for zero-shot systems on FPB/HateXplain:",
          {f"{m}|{d}": round(100 * (arms_tab[(m, d)][2] - arms_tab[(m, d)][0]), 2) for m, _, _ in SYSTEMS[:5] for d in ("fpb", "hatexplain")})
    print("excess risk alpha=1 worst draw:", worst[0], [round(100 * x, 1) for x in worst[1]])
    print("excess risk Jev HateXplain mean/p95/max:", [round(100 * x, 1) for x in ex[("jev", "hatexplain")]])
    print("max mean excess risk:", round(100 * max(v[0] for v in ex.values()), 2))
    print("resampling inflation range (points):", round(100 * min(infl.values()), 2), round(100 * max(infl.values()), 2),
          "Jev FPB", round(100 * infl[("jev", "fpb")], 2))
    print("without replacement PA1 analog (points, share n<200):", {d: (round(100 * w, 2), round(s, 2)) for d, (w, s) in worep.items()})
    print("Jev repeat ECE range:", {d: (round(a, 3), round(b, 3)) for d, (a, b) in rep_ece.items()})
    print("Jev retried calls:", retries)
    print("Jev temperature-arm ECE (eps 1e-3 / 1e-6 / 1e-12):", {d: tuple(round(x, 3) for x in v) for d, v in eps.items()})


if __name__ == "__main__":
    main()
