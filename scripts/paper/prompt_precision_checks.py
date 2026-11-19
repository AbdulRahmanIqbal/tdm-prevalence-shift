"""Exploratory (revision protocol v1.3, G5 + G6; added after the results): is the paired TDM-vs-backbone result on
FPB a property of the TDM prompt format or of 8-bit precision?

G5 (prompt sensitivity): the four local models scored in two generic formats that belong to neither TDM
('plain': plain-text multiple choice, letter readout; 'plain_names': option-likelihood readout), run id
final_promptsens, model names '<id>+<variant>'. The paired differences P1 (Tev1-4B - Qwen3.5-4B) and P2 (Nimble-9B -
Qwen3.5-9B) are recomputed in each format with the frozen bootstrap (B = 10,000, seed 20261006); "TDM format" is the
frozen run. Also: the backbone in its best format (chosen on test, so optimistic for the backbone) vs the TDM in its
own format.
G6 (precision): Qwen3.5-9B and Nimble-9B in bf16 with CPU offload (run id final_bf16, names '<id>+bf16') vs the frozen
8-bit runs: per-model Macro-F1 / ECE / agreement, and P2 in bf16.

Check: the TDM-format P1/P2 estimates and CIs must equal results/final_analysis/primary/p{1,2}_fpb.json.
Writes results/final_analysis_calibrob/prompt_precision.json and paper/tables/tab_prompt_precision.tex.
Run: .venv/Scripts/python scripts/paper/prompt_precision_checks.py [--g5-run final_promptsens] [--g6-run final_bf16]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from s1bench.application.prevalence_shift import to_matrices  # noqa: E402
from s1bench.application.primary_tests import fast_ece, fast_macro_f1, pair_tests  # noqa: E402
from s1bench.domain import label_shift as ls  # noqa: E402
from s1bench.infrastructure.storage.processed_store import YamlLabelRepository  # noqa: E402
from s1bench.infrastructure.storage.results_store import read_predictions  # noqa: E402

RES = ROOT / "results"
OUT = RES / "final_analysis_calibrob"
DATASET = "fpb"
PAIRS = [("P1", "tev1-4b", "qwen3.5-4b", "Tev1-4B $-$ Qwen3.5-4B"),
         ("P2", "nimble-9b", "qwen3.5-9b", "Nimble-9B $-$ Qwen3.5-9B")]
FORMATS = [("", "TDM format (frozen)"), ("+plain", "plain, letters"), ("+plain_names", "plain, option likelihood")]


def load(run: str, name: str):
    labels = YamlLabelRepository(ROOT / "data" / "labels").load(DATASET)
    preds = read_predictions(RES / run, name, DATASET, "test")
    m = to_matrices(preds, labels)
    if m.n_excluded:
        raise SystemExit(f"{run}/{name}: {m.n_excluded} non-ok rows")
    return m


def aligned(a, b) -> None:
    if a.ids != b.ids or not np.array_equal(a.gold, b.gold):
        raise SystemExit("prediction files are not aligned (same examples, same order)")


def single(m) -> dict[str, float]:
    pred = ls.argmax_by_position(m.probs, m.positions)
    k = m.probs.shape[1]
    return {"macro_f1": fast_macro_f1(pred, m.gold, k),
            "ece": fast_ece(m.probs.max(axis=1), (pred == m.gold).astype(float)),
            "pred": pred}


def pair(test: str, a, b) -> dict[str, dict[str, float]]:
    aligned(a, b)
    rows = pair_tests(test, DATASET, a, b)
    return {r.test.split("-")[1]: {"estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high,
                                   "p_value": r.p_value} for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g5-run", default="final_promptsens")
    ap.add_argument("--g6-run", default="final_bf16")
    ap.add_argument("--skip-g6", action="store_true")
    args = ap.parse_args()
    out: dict[str, object] = {"g5_run": args.g5_run, "g6_run": None if args.skip_g6 else args.g6_run}

    # G5
    g5: dict[str, object] = {}
    for test, tdm, base, _ in PAIRS:
        g5[test] = {}
        per_model = {}
        for suffix, _ in FORMATS:
            run = "final_local" if suffix == "" else args.g5_run
            a, b = load(run, tdm + suffix), load(run, base + suffix)
            res = pair(test, a, b)
            if suffix == "":
                frozen = json.loads((RES / "final_analysis" / "primary" / f"{test.lower()}_{DATASET}.json")
                                    .read_text(encoding="utf-8"))
                for r in frozen["results"]:
                    mine = res[r["test"].split("-")[1]]
                    if any(abs(mine[k] - r[e]) > 1e-12 for k, e in (("estimate", "estimate"), ("ci_low", "ci_low"),
                                                                   ("ci_high", "ci_high"))):
                        raise SystemExit(f"{test}: TDM-format arm does not reproduce the frozen {r['test']}")
            per_model[tdm + suffix], per_model[base + suffix] = single(a), single(b)
            g5[test][suffix or "tdm"] = res
        best_base = max((base + s for s, _ in FORMATS), key=lambda n: per_model[n]["macro_f1"])
        g5[test]["models"] = {n: {k: v for k, v in d.items() if k != "pred"} for n, d in per_model.items()}
        g5[test]["best_base_format"] = best_base
        g5[test]["tdm_own_minus_best_base_f1"] = per_model[tdm]["macro_f1"] - per_model[best_base]["macro_f1"]
    out["g5"] = g5

    # G6
    if not args.skip_g6:
        g6: dict[str, object] = {"models": {}}
        mats = {}
        for name in ("qwen3.5-9b", "nimble-9b"):
            q8, bf = load("final_local", name), load(args.g6_run, name + "+bf16")
            aligned(q8, bf)
            s8, sb = single(q8), single(bf)
            mats[name] = bf
            g6["models"][name] = {
                "8bit": {k: v for k, v in s8.items() if k != "pred"},
                "bf16": {k: v for k, v in sb.items() if k != "pred"},
                "argmax_agreement": float(np.mean(s8["pred"] == sb["pred"])),
                "max_abs_prob_difference": float(np.abs(q8.probs - bf.probs).max())}
        g6["P2_bf16"] = pair("P2", mats["nimble-9b"], mats["qwen3.5-9b"])
        out["g6"] = g6

    final = args.g5_run == "final_promptsens" and (args.skip_g6 or args.g6_run == "final_bf16")
    dest = OUT / "prompt_precision.json" if final else ROOT / "_scratch" / "prompt_precision_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)     # pipeline checks (other runs) write to _scratch/ only
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    def ci(d: dict[str, float]) -> str:
        return f"{100 * d['estimate']:+.1f} [{100 * d['ci_low']:+.1f}, {100 * d['ci_high']:+.1f}]"

    lines = ["\\begin{tabular}{llrrrr}", "\\toprule",
             "Pair & Format & TDM F1 & Base F1 & $\\Delta$Macro-F1 [95\\% CI] & $\\Delta$ECE [95\\% CI] \\\\",
             "\\midrule"]
    for test, tdm, base, label in PAIRS:
        for j, (suffix, flabel) in enumerate(FORMATS):
            r, mm = g5[test][suffix or "tdm"], g5[test]["models"]
            shade = "\\rowcolor{rowhl} " if j == 0 else ""   # the frozen confirmatory row (TDM format) shaded
            lines.append(f"{shade}{label if j == 0 else ''} & {flabel} & {100 * mm[tdm + suffix]['macro_f1']:.1f} & "
                         f"{100 * mm[base + suffix]['macro_f1']:.1f} & {ci(r['F1'])} & {ci(r['ECE'])} \\\\")
            print(f"{test} {flabel:26s} F1 {ci(r['F1'])}  ECE {ci(r['ECE'])}")
        if test == "P2" and not args.skip_g6:
            r, mm = out["g6"]["P2_bf16"], out["g6"]["models"]
            lines.append(f" & Nimble format, bf16 & {100 * mm['nimble-9b']['bf16']['macro_f1']:.1f} & "
                         f"{100 * mm['qwen3.5-9b']['bf16']['macro_f1']:.1f} & {ci(r['F1'])} & {ci(r['ECE'])} \\\\")
            print(f"P2 bf16                        F1 {ci(r['F1'])}  ECE {ci(r['ECE'])}")
        lines.append("\\midrule" if test != PAIRS[-1][0] else "\\bottomrule")
    lines.append("\\end{tabular}")
    if final:
        (ROOT / "paper" / "tables" / "tab_prompt_precision.tex").write_text("\n".join(lines) + "\n",
                                                                           encoding="utf-8")


if __name__ == "__main__":
    main()
