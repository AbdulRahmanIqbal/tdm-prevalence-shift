"""Stage 2 paper tables (label budgets, Exp 3 and Exp 4 condition b) from the FINAL budget curves. Writes paper/tables/.

Inputs: results/final_budget_analysis/curves/*.json (written by `s1bench budget-curve`), data/processed/<d>/train.jsonl
(shots per class only). Never reads pilot or dev_* runs. Stage 2 is estimation: no p-values, no "significant".

Outputs: tab_crossover.tex (main text), tab_qlora_vs_base.tex (main text), tab_budget_f1.tex, tab_budget_ece.tex and
tab_shots_per_class.tex (appendix).

Run: .venv/Scripts/python scripts/paper/budget_tables.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from s1bench.domain.budgets import stratified_subset  # noqa: E402
from s1bench.infrastructure.storage.processed_store import JsonlProcessedReader  # noqa: E402

OUT = ROOT / "paper" / "tables"
CURVES = ROOT / "results" / "final_budget_analysis" / "curves"
# Stage 2 tables and figure order datasets by number of classes (the crossover rises with it)
DATASETS = ["hatexplain", "fpb", "trec6", "hwu64", "clinc150"]
DS_LABEL = {"clinc150": "CLINC150", "hwu64": "HWU64", "fpb": "FPB", "hatexplain": "HateXplain", "trec6": "TREC-6"}
N_CLASSES = {"hatexplain": 3, "fpb": 3, "trec6": 6, "hwu64": 64, "clinc150": 150}
SYSTEMS = [("deberta-v3-large", "DeBERTa-v3-large"), ("tfidf-lr", "TF-IDF + LR"), ("qwen3.5-4b-qlora", "Qwen3.5-4B + QLoRA")]
EXPECTED_SEEDS = {"deberta-v3-large": 5, "tfidf-lr": 5, "qwen3.5-4b-qlora": 3}
SEEDS = [1001, 1002, 1003, 1004, 1005]


def curve(system: str, ref: str, dataset: str) -> dict:
    refrun = "final_jev" if ref == "jev" else "final_local"
    d = json.loads((CURVES / f"{system}_vs_{ref}@{refrun}_{dataset}.json").read_text(encoding="utf-8"))
    for rows in (d["curves"][m]["rows"] for m in ("macro_f1", "ece")):
        for r in rows:
            full_tfidf = system == "tfidf-lr" and r["budget"] == "full"
            if r["n_seeds"] != (1 if full_tfidf else EXPECTED_SEEDS[system]):
                raise SystemExit(f"incomplete curve: {system} vs {ref} {dataset} b{r['budget']}")
    return d


def num(x: int | str) -> str:
    return "full" if x == "full" else f"{int(x):,}".replace(",", "{,}")


def signed1(x: float) -> str:
    """Signed points: 1 decimal, or 2 decimals below 0.1 in absolute value; typographic minus for LaTeX."""
    s = f"{x:+.2f}" if 0 < abs(x) < 0.095 else f"{x:+.1f}"
    if s in ("-0.0", "+0.0", "-0.00", "+0.00"):
        return "0.0"
    return s.replace("-", "$-$")


def delta_ci(r: dict) -> str:
    return f"{signed1(100 * r['estimate'])} [{signed1(100 * r['ci_low'])}, {signed1(100 * r['ci_high'])}]"


def crossover_cells(c: dict) -> tuple[str, str]:
    iv, pt = c["interval"], c["interpolated_point_estimate"]
    if c["crossover"] == "not reached":
        iv_s = "not reached"
    elif iv == "<= 100":
        iv_s = "$\\le$100"
    else:
        lo, hi = iv.strip("(]").split(", ")
        iv_s = f"({num(lo)}, {num(hi)}]"
    if isinstance(pt, str):
        pt_s = "$\\le$100" if pt == "<= 100" else "--"
    else:
        pt_s = num(int(float(f"{pt:.3g}")))  # 3 significant digits: the interpolation is a rough guide only
    return iv_s, pt_s


def table_crossover() -> str:
    out = ["\\begin{tabular}{lrcrcrcr}", "\\toprule",
           " & & \\multicolumn{2}{c}{DeBERTa-v3-large} & \\multicolumn{2}{c}{TF-IDF + LR} & "
           "\\multicolumn{2}{c}{Qwen3.5-4B + QLoRA} \\\\",
           "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}",
           "Dataset & Classes & Crossover & Interp. & Crossover & Interp. & Crossover & Interp. \\\\", "\\midrule"]
    for d in DATASETS:
        cells = []
        for s, _ in SYSTEMS:
            cells += crossover_cells(curve(s, "jev", d)["curves"]["macro_f1"]["crossover"])
        out.append(f"{DS_LABEL[d]} & {N_CLASSES[d]} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def table_budget(metric: str) -> str:
    """Per-budget paired difference vs Jev (points, 95% CI), one block per dataset; QLoRA only at 100/500/2,500."""
    out = ["\\begin{tabular}{lrrlll}", "\\toprule",
           "Dataset & Budget & $n$ & DeBERTa-v3-large & TF-IDF + LR & Qwen3.5-4B + QLoRA \\\\", "\\midrule"]
    for k, d in enumerate(DATASETS):
        if k:
            out.append("\\midrule")
        cs = {s: curve(s, "jev", d) for s, _ in SYSTEMS}
        rows = {s: {str(r["budget"]): r for r in c["curves"][metric]["rows"]} for s, c in cs.items()}
        sizes = cs["deberta-v3-large"]["sizes"]
        for j, b in enumerate(sizes):
            cells = [delta_ci(rows[s][b]) if b in rows[s] else "--" for s, _ in SYSTEMS]
            name = DS_LABEL[d] if j == 0 else ""
            out.append(f"{name} & {num(b)} & {num(sizes[b])} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def table_qlora_vs_base() -> str:
    """Exp 4: condition (b) task QLoRA vs (a) frozen Qwen3.5-4B and (c) Tev1-4B, Macro-F1 (descriptive)."""
    out = ["\\begin{tabular}{lrll}", "\\toprule",
           "Dataset & Budget & vs Qwen3.5-4B (a) & vs Tev1-4B (c) \\\\", "\\midrule"]
    for k, d in enumerate(DATASETS):
        if k:
            out.append("\\midrule")
        rows = {ref: {str(r["budget"]): r for r in curve("qwen3.5-4b-qlora", ref, d)["curves"]["macro_f1"]["rows"]}
                for ref in ("qwen3.5-4b", "tev1-4b")}
        for j, b in enumerate(("100", "500", "2500")):
            name = DS_LABEL[d] if j == 0 else ""
            out.append(f"{name} & {num(b)} & {delta_ci(rows['qwen3.5-4b'][b])} & {delta_ci(rows['tev1-4b'][b])} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def table_shots_per_class() -> str:
    """Training examples per class in the budget subsets (high-K datasets), over the 5 seeds."""
    reader = JsonlProcessedReader(ROOT / "data" / "processed")
    out = ["\\begin{tabular}{rrrrr}", "\\toprule",
           " & \\multicolumn{2}{c}{CLINC150 (150 classes)} & \\multicolumn{2}{c}{HWU64 (64 classes)} \\\\",
           "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
           "Budget & Classes seen & Per class (min--max) & Classes seen & Per class (min--max) \\\\", "\\midrule"]
    train = {d: reader.read_split(d, "train") for d in ("clinc150", "hwu64")}
    for b in (100, 250, 500, 1000, 2500, 5000, None):
        cells = []
        for d in ("clinc150", "hwu64"):
            seen, lo, hi = [], [], []
            for seed in SEEDS:
                cnt = Counter(e.label for e in stratified_subset(train[d], b, seed))
                seen.append(len(cnt))
                lo.append(min(cnt.values()) if len(cnt) == N_CLASSES[d] else 0)
                hi.append(max(cnt.values()))
            if len(set(seen)) != 1:
                raise SystemExit(f"classes seen differ across seeds: {d} b{b}")
            cells += [str(seen[0]), f"{min(lo)}--{max(hi)}"]
        out.append(f"{num(b if b else 'full')} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def table_deberta_confidence() -> str:
    """DeBERTa-v3-large on the high-K datasets: mean top-label probability vs accuracy (test, mean over 5 seeds).
    Shows the under-confidence behind the large mid-budget ECE (descriptive)."""
    pred = ROOT / "results" / "final_budget" / "predictions"
    full = ROOT / "results" / "final_train" / "predictions"
    full_runs = ["deberta-v3-large"] + [f"deberta-v3-large@fulls{s}" for s in SEEDS[1:]]
    out = ["\\begin{tabular}{rrrrr}", "\\toprule",
           " & \\multicolumn{2}{c}{CLINC150} & \\multicolumn{2}{c}{HWU64} \\\\",
           "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
           "Budget & Confidence & Accuracy & Confidence & Accuracy \\\\", "\\midrule"]
    for b in (500, 1000, 2500, 5000, "full"):
        cells = []
        for d in ("clinc150", "hwu64"):
            files = ([full / r / f"{d}_test.jsonl" for r in full_runs] if b == "full" else
                     [pred / f"deberta-v3-large@b{b}s{s}" / f"{d}_test.jsonl" for s in SEEDS])
            conf, acc = [], []
            for f in files:
                rows = [json.loads(line) for line in f.open(encoding="utf-8")]
                conf.append(statistics.mean(max(r["probabilities"]) for r in rows))
                acc.append(statistics.mean(r["predicted"] == r["gold"] for r in rows))
            cells += [f"{statistics.mean(conf):.3f}", f"{statistics.mean(acc):.3f}"]
        out.append(f"{num(b)} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("tab_crossover", table_crossover), ("tab_qlora_vs_base", table_qlora_vs_base),
                     ("tab_budget_f1", lambda: table_budget("macro_f1")),
                     ("tab_budget_ece", lambda: table_budget("ece")),
                     ("tab_shots_per_class", table_shots_per_class),
                     ("tab_deberta_confidence", table_deberta_confidence)):
        (OUT / f"{name}.tex").write_text(fn() + "\n", encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
