"""Spec 4.5 (required sensitivity): HWU64 results without the 614 test examples that overlap MASSIVE.

Recomputes Exp 1 metrics for every system and the P1/P2 paired bootstrap (same B and seed as the primary tests) on the
non-overlapping subset. Descriptive: outside the Holm family. Reads only final runs and data/overlap/flagged_hwu64.json.
Writes results/final_analysis/hwu64_nonoverlap.json and paper/tables/tab_hwu64_nonoverlap.tex.

Run: .venv/Scripts/python scripts/paper/hwu64_overlap_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from s1bench.application.exp1_metrics import exp1_row  # noqa: E402
from s1bench.application.primary_tests import aligned_matrices, pair_tests  # noqa: E402
from s1bench.infrastructure.storage.processed_store import YamlLabelRepository  # noqa: E402
from s1bench.infrastructure.storage.results_store import read_predictions  # noqa: E402

RUNS = {"jev": "final_jev", "tev1-4b": "final_local", "nimble-9b": "final_local", "qwen3.5-4b": "final_local",
        "qwen3.5-9b": "final_local", "deberta-v3-large": "final_train", "tfidf-lr": "final_train"}
LABEL = {"jev": "Jev", "tev1-4b": "Tev1-4B", "nimble-9b": "Nimble-9B", "qwen3.5-4b": "Qwen3.5-4B",
         "qwen3.5-9b": "Qwen3.5-9B", "deberta-v3-large": "DeBERTa-v3-large", "tfidf-lr": "TF-IDF + LR"}


def main() -> None:
    flagged = set(json.loads((ROOT / "data" / "overlap" / "flagged_hwu64.json").read_text(encoding="utf-8")))
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / "primary.yaml").read_text(encoding="utf-8"))
    labels = YamlLabelRepository(ROOT / "data" / "labels").load("hwu64")
    preds = {m: read_predictions(ROOT / "results" / run, m, "hwu64", "test") for m, run in RUNS.items()}
    keep = {m: [p for p in ps if p.example_id not in flagged] for m, ps in preds.items()}
    n_all, n_keep = len(preds["jev"]), len(keep["jev"])
    out = {"n_test": n_all, "n_flagged": len(flagged), "n_nonoverlap": n_keep, "exp1": {}, "pairs": []}
    for m, ps in keep.items():
        full, sub = exp1_row(preds[m]), exp1_row(ps)
        out["exp1"][m] = {"full": {k: full[k] for k in ("macro_f1", "ece")},
                          "nonoverlap": {k: sub[k] for k in ("macro_f1", "ece")}}
    for test, pair in cfg["pairs"].items():
        a, b, dropped = aligned_matrices(keep[pair["model"]], keep[pair["base"]], labels)
        for r in pair_tests(test, "hwu64", a, b, b=int(cfg["bootstrap_b"]), seed=int(cfg["seed"])):
            out["pairs"].append(r.__dict__)
    (ROOT / "results" / "final_analysis" / "hwu64_nonoverlap.json").write_text(json.dumps(out, indent=2) + "\n",
                                                                              encoding="utf-8")
    rows = ["\\begin{tabular}{lrrrr}", "\\toprule",
            f"System & F1 (all {n_all}) & F1 (non-overlap {n_keep}) & ECE (all) & ECE (non-overlap) \\\\", "\\midrule"]
    for m, v in out["exp1"].items():
        rows.append(f"{LABEL[m]} & {v['full']['macro_f1']:.3f} & {v['nonoverlap']['macro_f1']:.3f} & "
                    f"{v['full']['ece']:.3f} & {v['nonoverlap']['ece']:.3f} \\\\")
    rows.append("\\midrule")
    def pts(x: float) -> str:   # points; 2 decimals below 0.1 so that e.g. -0.03 is not printed as 0.0
        s = f"{100 * x:+.2f}" if 0 < abs(100 * x) < 0.095 else f"{100 * x:+.1f}"
        return s.replace("-", "$-$")
    for r in out["pairs"]:
        rows.append(f"{r['test']} (non-overlap) & \\multicolumn{{4}}{{l}}{{$\\Delta$ = {pts(r['estimate'])} points, "
                    f"95\\% CI [{pts(r['ci_low'])}, {pts(r['ci_high'])}], unadjusted $p$ = "
                    f"{'$<$0.0002' if r['p_value'] < 0.0002 else format(r['p_value'], '.3f')}}} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper" / "tables" / "tab_hwu64_nonoverlap.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_test", "n_flagged", "n_nonoverlap")}))
    for m, v in out["exp1"].items():
        print(f"{m:17s} F1 {v['full']['macro_f1']:.3f} -> {v['nonoverlap']['macro_f1']:.3f}   ECE {v['full']['ece']:.3f} -> {v['nonoverlap']['ece']:.3f}")
    for r in out["pairs"]:
        print(f"{r['test']:7s} {100*r['estimate']:+.1f} [{100*r['ci_low']:+.1f}, {100*r['ci_high']:+.1f}] p={r['p_value']:.4f}")


if __name__ == "__main__":
    main()
