"""Paper tables (LaTeX, booktabs) from FINAL Stage 1 results and frozen data/configs. Writes paper/tables/*.tex.

Inputs: data/processed/<d>/report.json, data/candidates/<d>/test.jsonl, configs/models.yaml,
results/final_analysis/exp1_test*.json, results/final_analysis/primary/holm_family.json.
Never reads pilot or dev_* runs.

Run: .venv/Scripts/python scripts/paper/make_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "tables"
DATASETS = ["clinc150", "hwu64", "fpb", "hatexplain", "trec6"]
DS_LABEL = {"clinc150": "CLINC150", "hwu64": "HWU64", "fpb": "FPB", "hatexplain": "HateXplain", "trec6": "TREC-6"}
DOMAIN = {"clinc150": "intent (assistant)", "hwu64": "intent (home assistant)", "fpb": "financial sentiment",
          "hatexplain": "hate / offensive speech", "trec6": "question type"}
ALPHA_SIG = 0.05


def f3(x: float) -> str:
    return f"{x:.3f}"


def signed2(x: float) -> str:
    """Signed points with 2 decimals throughout (PA and all-tests tables, so precision is uniform)."""
    s = f"{x:+.2f}"
    return "0.00" if s in ("-0.00", "+0.00") else s.replace("-", "$-$")


def signed1(x: float) -> str:
    """Signed points: 1 decimal, or 2 decimals below 0.1 in absolute value (so a bound such as -0.04 is not shown as
    '0.0'); typographic minus for LaTeX."""
    s = f"{x:+.2f}" if 0 < abs(x) < 0.095 else f"{x:+.1f}"
    if s in ("-0.0", "+0.0", "-0.00", "+0.00"):
        return "0.0"
    return s.replace("-", "$-$")


def table_datasets() -> str:
    rows = []
    for d in DATASETS:
        rep = json.loads((ROOT / "data" / "processed" / d / "report.json").read_text(encoding="utf-8"))
        c = rep["counts"]
        n = lambda s: c[s] if isinstance(c[s], int) else c[s].get("after", c[s].get("n"))  # noqa: E731
        k = len(rep["label_counts"]["train"])
        sizes = sorted({len(json.loads(line)["candidates"])
                        for line in (ROOT / "data" / "candidates" / d / "test.jsonl").open(encoding="utf-8")})
        kc = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]}--{sizes[-1]}"
        rows.append(f"{DS_LABEL[d]} & {DOMAIN[d]} & {k} & {kc} & {n('train'):,} & {n('dev'):,} & {n('test'):,} \\\\")
    return "\n".join([
        "\\begin{tabular}{llrrrrr}", "\\toprule",
        "Dataset & Domain & Classes & Candidates $K$ & Train & Dev & Test \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}"])


def table_systems() -> str:
    m = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["models"]
    def rev(x):
        return f"\\texttt{{{x[:8]}}}"
    lines = [
        "\\textsc{Typed decision models} & & & & \\\\",
        "Jev & Cloudflare API (\\texttt{typesafe/jev}) & \\texttt{jev-1.13.0} & API & API probabilities \\\\",
        f"Tev1-4B & \\texttt{{{m['tev1-4b']['repo']}}} & {rev(m['tev1-4b']['revision'])} & bf16 & letters, Tev1 format \\\\",
        f"Nimble-9B & \\makecell[l]{{Qwen3.5-9B + LoRA \\\\ \\texttt{{{m['nimble-9b']['adapter_repo']}}}}} & "
        f"\\makecell[l]{{{rev(m['nimble-9b']['adapter_revision'])} \\\\ (\\texttt{{original-2676}})}} & 8-bit & letters, Nimble format \\\\",
        "\\midrule", "\\textsc{Base LLMs} & & & & \\\\",
        f"Qwen3.5-4B & \\texttt{{{m['qwen3.5-4b']['repo']}}} & {rev(m['qwen3.5-4b']['revision'])} & bf16 & as Tev1-4B \\\\",
        f"Qwen3.5-9B & \\texttt{{{m['qwen3.5-9b']['repo']}}} & {rev(m['qwen3.5-9b']['revision'])} & 8-bit & as Nimble-9B \\\\",
        "\\midrule", "\\textsc{Trained classifiers} & & & & \\\\",
        "DeBERTa-v3-large & fine-tuned on the full training set, 5 seeds & \\texttt{64a8c8ea} & fp16 & softmax \\\\",
        "TF-IDF + LR & word 1--2 grams, $C$ selected on dev & scikit-learn 1.9.1 & CPU & softmax \\\\",
    ]
    return "\n".join(["\\begin{tabular}{lllll}", "\\toprule",
                      "System & Source & Revision & Precision & Readout \\\\", "\\midrule", *lines,
                      "\\bottomrule", "\\end{tabular}"])


def table_exp1() -> str:
    rows = json.loads((ROOT / "results" / "final_analysis" / "exp1_test.json").read_text(encoding="utf-8"))
    rows += json.loads((ROOT / "results" / "final_analysis" / "exp1_test_jev.json").read_text(encoding="utf-8"))
    by = {(r["model"], r["dataset"]): r for r in rows}
    groups = [("Typed decision models", [("jev", "Jev"), ("tev1-4b", "Tev1-4B"), ("nimble-9b", "Nimble-9B")]),
              ("Base LLMs", [("qwen3.5-4b", "Qwen3.5-4B"), ("qwen3.5-9b", "Qwen3.5-9B")]),
              ("Trained classifiers (full data)", [("deberta-v3-large", "DeBERTa-v3-large"), ("tfidf-lr", "TF-IDF + LR")])]
    all_models = [m for _, ms in groups for m, _ in ms]
    best_f1 = {d: max(by[(m, d)]["macro_f1"] for m in all_models) for d in DATASETS}
    best_ece = {d: min(by[(m, d)]["ece"] for m in all_models) for d in DATASETS}
    out = ["\\begin{tabular}{l" + "rr" * len(DATASETS) + "}", "\\toprule",
           " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS_LABEL[d]}}}" for d in DATASETS) + " \\\\",
           " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DATASETS))),
           "System & " + " & ".join(["F1$\\uparrow$ & ECE$\\downarrow$"] * len(DATASETS)) + " \\\\", "\\midrule"]
    for gi, (gname, models) in enumerate(groups):
        if gi:
            out.append("\\midrule")
        out.append(f"\\multicolumn{{{1 + 2 * len(DATASETS)}}}{{l}}{{\\textsc{{{gname}}}}} \\\\")
        for m, label in models:
            cells = []
            for d in DATASETS:
                r = by[(m, d)]
                f1, ece = f3(r["macro_f1"]), f3(r["ece"])
                cells.append(f1)   # no bold: an untested lead must not look like a result (style guide)
                cells.append(ece)
            out.append(f"{label} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def table_pa() -> str:
    data = json.loads((ROOT / "results" / "final_analysis" / "primary" / "holm_family.json").read_text(encoding="utf-8"))
    if not data.get("complete"):
        raise SystemExit("Holm family incomplete")
    h = {(r["test"], r["dataset"]): r for r in data["results"]}
    names = {"PA1": "PA1: $\\Delta$ECE under shift, Jev $-$ DeBERTa",
             "PA2": "PA2: excess risk at $\\tau_{80}$, Jev $-$ DeBERTa",
             "PA3": "PA3: ECE, Jev EM $-$ none"}
    out = ["\\begin{tabular}{l" + "c" * len(DATASETS) + "}", "\\toprule",
           "Test (points) & " + " & ".join(DS_LABEL[d] for d in DATASETS) + " \\\\", "\\midrule"]
    for t in ("PA1", "PA2", "PA3"):
        cells = []
        for d in DATASETS:
            r = h[(t, d)]
            est = signed2(100 * r["estimate"])
            ci = f"[{signed2(100 * r['ci_low'])}, {signed2(100 * r['ci_high'])}]"
            shown = "\\textbf{" + est + "}$^\\dagger$" if r["p_holm"] < ALPHA_SIG else est
            cells.append("\\makecell{" + shown + "\\\\{\\scriptsize " + ci + "}}")
        out.append(f"{names[t]} & " + " & ".join(cells) + " \\\\[2pt]")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


SYSTEMS_FULL = [("jev", "Jev"), ("tev1-4b", "Tev1-4B"), ("nimble-9b", "Nimble-9B"),
                ("nimble-9b-main", "Nimble-9B (main)$^\\ast$"), ("qwen3.5-4b", "Qwen3.5-4B"),
                ("qwen3.5-9b", "Qwen3.5-9B"), ("deberta-v3-large", "DeBERTa-v3-large"), ("tfidf-lr", "TF-IDF + LR")]


def _exp1_rows() -> dict[tuple[str, str], dict]:
    rows = json.loads((ROOT / "results" / "final_analysis" / "exp1_test.json").read_text(encoding="utf-8"))
    rows += json.loads((ROOT / "results" / "final_analysis" / "exp1_test_jev.json").read_text(encoding="utf-8"))
    return {(r["model"], r["dataset"]): r for r in rows}


def table_exp1_full() -> str:
    by = _exp1_rows()
    out = ["\\begin{tabular}{llrrrrrrrr}", "\\toprule",
           "Dataset & System & Acc$\\uparrow$ & F1$\\uparrow$ & MCC$\\uparrow$ & ECE$\\downarrow$ & Brier$\\downarrow$ & "
           "NLL$\\downarrow$ & AURC$\\downarrow$ & Acc@80$\\uparrow$ \\\\", "\\midrule"]
    for i, d in enumerate(DATASETS):
        if i:
            out.append("\\midrule")
        for j, (m, label) in enumerate(SYSTEMS_FULL):
            r = by[(m, d)]
            cells = [f3(r[k]) for k in ("accuracy", "macro_f1", "mcc", "ece", "brier", "nll", "aurc", "acc_at_80")]
            out.append(f"{DS_LABEL[d] if j == 0 else ''} & {label} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def table_deberta_seeds() -> str:
    import statistics as st
    by = _exp1_rows()
    names = ["deberta-v3-large"] + [f"deberta-v3-large@fulls{s}" for s in (1002, 1003, 1004, 1005)]
    out = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
           "Dataset & seed 1001 & 1002 & 1003 & 1004 & 1005 & mean $\\pm$ SD \\\\", "\\midrule"]
    for d in DATASETS:
        v = [by[(n, d)]["macro_f1"] for n in names]
        out.append(f"{DS_LABEL[d]} & " + " & ".join(f3(x) for x in v) + f" & {st.mean(v):.3f} $\\pm$ {st.stdev(v):.3f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def table_jev_repeat() -> str:
    import itertools
    def load(run, d):
        p = ROOT / "results" / run / "predictions" / "jev" / f"{d}_test.jsonl"
        return {json.loads(line)["example_id"]: json.loads(line) for line in p.open(encoding="utf-8")}
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from s1bench.domain import metrics as mt

    def ece(run_rows, ids):
        sc = [mt.Scored(tuple(run_rows[i]["candidates"]),
                        tuple(v / sum(run_rows[i]["probabilities"]) for v in run_rows[i]["probabilities"]),
                        run_rows[i]["gold"]) for i in ids]
        return mt.ece(sc)

    out = ["\\begin{tabular}{lrrrrlc}", "\\toprule",
           "Dataset & $n$ & same prediction & same prob.\\ vector & max $|\\Delta p|$ & accuracy (3 repeats) & "
           "ECE range (4 runs) \\\\", "\\midrule"]
    for d in DATASETS:
        reps = [load(f"final_jev_rep{r}", d) for r in (1, 2, 3)]
        full = load("final_jev", d)
        ids = sorted(reps[0])
        runs = reps + [{i: full[i] for i in ids}]
        same_pred = sum(len({r[i]["predicted"] for r in runs}) == 1 for i in ids) / len(ids)
        same_prob = sum(len({tuple(r[i]["probabilities"]) for r in runs}) == 1 for i in ids) / len(ids)
        maxdp = max(max(abs(a - b) for a, b in zip(x[i]["probabilities"], y[i]["probabilities"]))
                    for x, y in itertools.combinations(runs, 2) for i in ids)
        accs = [sum(r[i]["predicted"] == r[i]["gold"] for i in ids) / len(ids) for r in reps]
        eces = [ece(r, ids) for r in runs]
        out.append(f"{DS_LABEL[d]} & {len(ids)} & {100 * same_pred:.1f}\\% & {100 * same_prob:.1f}\\% & {maxdp:.2f} & "
                   + " / ".join(f3(a) for a in accs) + f" & {f3(min(eces))}--{f3(max(eces))} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def table_pairs() -> str:
    data = json.loads((ROOT / "results" / "final_analysis" / "primary" / "holm_family.json").read_text(encoding="utf-8"))
    if not data.get("complete"):
        raise SystemExit("Holm family incomplete")
    h = {(r["test"], r["dataset"]): r for r in data["results"]}
    fp = lambda p: "$<$0.0002" if p < 0.0002 else f"{p:.3f}"  # noqa: E731
    # Holm bound: a raw p of 0 is only known to be < 2/B, so its Holm-adjusted value is < 35 * 2/B = 0.007
    fh = lambda p: "$<$0.007" if p < 0.0002 else f"{p:.3f}"  # noqa: E731
    out = ["\\begin{tabular}{llrcrr}", "\\toprule",
           "Test & Dataset & $\\Delta$ (points) & 95\\% CI & $p$ & Holm $p$ \\\\", "\\midrule"]
    for k, t in enumerate(("P1-F1", "P1-ECE", "P2-F1", "P2-ECE", "PA1", "PA2", "PA3")):
        if k:
            out.append("\\midrule")
        for d in DATASETS:
            r = h[(t, d)]
            shade = "\\rowcolor{rowhl} " if r["p_holm"] < ALPHA_SIG else ""   # Holm-significant rows shaded
            out.append(f"{shade}{t} & {DS_LABEL[d]} & {signed2(100 * r['estimate'])} & "
                       f"[{signed2(100 * r['ci_low'])}, {signed2(100 * r['ci_high'])}] & {fp(r['p_value'])} & "
                       f"{fh(r['p_holm'])} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("tab_datasets", table_datasets), ("tab_systems", table_systems),
                     ("tab_exp1", table_exp1), ("tab_pa", table_pa), ("tab_exp1_full", table_exp1_full),
                     ("tab_deberta_seeds", table_deberta_seeds), ("tab_jev_repeat", table_jev_repeat),
                     ("tab_all_tests", table_pairs)):
        (OUT / f"{name}.tex").write_text(fn() + "\n", encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
