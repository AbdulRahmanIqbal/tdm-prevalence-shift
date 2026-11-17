"""Exploratory (revision protocol v1.3, G4; added after the results): content-free calibration (Zhao et al. 2021)
of the zero-shot systems, compared with no correction and with EM under the implicit prior.

Inputs: the content-free probe predictions (`predict --split contentfree`, run id final_contentfree; probes from
`contentfree-probes`, see s1bench.domain.content_free) and the frozen test/dev predictions.

Arms
- none: raw probabilities.
- cf_label: q proportional to p / w, w = label-level content-free probability (rotation probes, all datasets).
- cf_matched: same with w of exactly the example's option order (FPB, HateXplain, TREC-6 only).
- em: EM with the implicit prior (the paper's arm in Exp 2).
- cf_label_em: cf_label first, then EM with the implicit prior of the cf_label-corrected dev predictions.

(1) No shift: full test split, Macro-F1 and top-label ECE.
(2) Shift: the 200 with-replacement draws at alpha = 1 of Exp 2, regenerated with its seed and order; the regenerated
'none' and 'em' arms must reproduce the stored Exp 2 rows (top-label ECE of every draw), as in em_true_prior.py.
Probes with a non-ok status are left out of w and counted in the output.

Writes results/final_analysis_calibrob/content_free.json and paper/tables/tab_content_free.tex.
Run: .venv/Scripts/python scripts/paper/content_free_calibration.py [--cf-run final_contentfree]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
from calibration_robustness import DATASETS, DS, OUT, RES, RUN, matrices  # noqa: E402
from s1bench.domain import content_free as cf  # noqa: E402
from s1bench.domain import label_shift as ls  # noqa: E402
from s1bench.infrastructure.storage.processed_store import JsonlProcessedReader  # noqa: E402
from s1bench.infrastructure.storage.results_store import read_predictions  # noqa: E402

SYSTEMS = [("jev", "Jev"), ("tev1-4b", "Tev1-4B"), ("nimble-9b", "Nimble-9B"), ("qwen3.5-4b", "Qwen3.5-4B"),
           ("qwen3.5-9b", "Qwen3.5-9B")]
ARMS = [("none", "none"), ("cf_label", "content-free, label level"), ("cf_matched", "content-free, order matched"),
        ("em", "EM, implicit prior"), ("cf_label_em", "content-free + EM")]


def order_of(positions: np.ndarray, mask: np.ndarray, label_ids: list[str]) -> list[tuple[str, ...]]:
    """Option order of every row of a label-indexed matrix (from the position matrix)."""
    out = []
    for pos, mk in zip(positions, mask):
        cols = np.flatnonzero(mk)
        out.append(tuple(label_ids[c] for c in cols[np.argsort(pos[cols], kind="stable")]))
    return out


def probe_weights(cf_run: Path, model: str, dataset: str, label_ids: list[str]):
    probes = read_predictions(cf_run, model, dataset, "contentfree")
    meta = {e.id: e.meta for e in JsonlProcessedReader(ROOT / "data" / "contentfree" / "processed")
            .read_split(dataset, "contentfree")}
    if {p.example_id for p in probes} != set(meta):
        raise SystemExit(f"{model}/{dataset}: probe predictions do not cover the probe file")
    ok = [p for p in probes if p.status == "ok" and p.probabilities]
    rot = [(p.candidates, p.probabilities) for p in ok if "rotation" in meta[p.example_id]["designs"]]
    mat = [(p.candidates, p.probabilities) for p in ok if "matched" in meta[p.example_id]["designs"]]
    w_label = cf.label_level_weights(rot, label_ids)
    by_text = {}
    for p in ok:
        by_text.setdefault(meta[p.example_id]["text_index"], []).append(p)
    info = {"n_probes": len(probes), "n_ok": len(ok),
            "n_not_ok_by_status": {s: sum(p.status == s for p in probes) for s in {p.status for p in probes}
                                   if s != "ok"},
            "n_ok_by_text": {cf.CONTENT_FREE_TEXTS[t]: len(v) for t, v in sorted(by_text.items())},
            "w_label": {lid: (None if np.isnan(v) else round(float(v), 6)) for lid, v in zip(label_ids, w_label)},
            "w_label_max_over_min": float(np.nanmax(w_label) / max(np.nanmin(w_label), cf.W_FLOOR))}
    return w_label, (cf.matched_weights(mat) if mat else None), info


def evaluate(model: str, dataset: str, cf_run: Path) -> dict[str, object]:
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / "exp2.yaml").read_text(encoding="utf-8"))
    em = (cfg["em"]["max_iter"], cfg["em"]["tol"])
    test, labels = matrices(model, dataset, "test")
    dev, _ = matrices(model, dataset, "dev")
    label_ids = labels.ids()
    w_label, w_matched, info = probe_weights(cf_run, model, dataset, label_ids)

    def corrected(m) -> dict[str, np.ndarray]:
        out = {"cf_label": cf.content_free_adjust(m.probs, m.mask, cf.weights_matrix_label(w_label, len(m.gold)))}
        if w_matched is not None:
            orders = order_of(m.positions, m.mask, label_ids)
            if all(o in w_matched for o in orders):
                out["cf_matched"] = cf.content_free_adjust(
                    m.probs, m.mask, cf.weights_matrix_matched(orders, w_matched, label_ids))
        return out

    test_cf = corrected(test)
    pi_implicit = ls.implicit_prior(dev.probs)
    pi_implicit_cf = ls.implicit_prior(corrected(dev)["cf_label"])

    no_shift = {}
    for arm, probs in {"none": test.probs, **test_cf}.items():
        mm = ls.matrix_metrics(probs, test.gold, positions=test.positions)
        no_shift[arm] = {"macro_f1": mm.macro_f1, "ece": mm.ece}

    report = json.loads((ROOT / "data" / "processed" / dataset / "report.json").read_text(encoding="utf-8"))
    counts = report["label_counts"]["train"]
    train_prior = np.array([counts.get(lid, 0) for lid in label_ids], dtype=float)
    train_prior /= train_prior.sum()
    index = {lid: i for i, lid in enumerate(label_ids)}
    full = JsonlProcessedReader(ROOT / "data" / "processed").read_split(dataset, "test")
    full_gold = np.array([index[e.label] for e in full])
    row_of = {eid: r for r, eid in enumerate(test.ids)}
    to_row = np.array([row_of[e.id] for e in full])
    stored = {(r["arm"], r["draw"]): r for r in
              (json.loads(line) for line in
               (RES / RUN[model] / "exp2" / model / f"{dataset}.jsonl").read_text(encoding="utf-8").splitlines())
              if r["alpha"] == "1" and r["sampling"] == "with_replacement"}
    rng = np.random.default_rng(cfg["seed"])
    shift: dict[str, dict[str, list[float]]] = {}
    for alpha in cfg["alphas"]:
        for d in range(1, cfg["draws_per_alpha"] + 1):
            target = ls.dirichlet_target_prior(train_prior, alpha, rng)
            idx = to_row[ls.resample_with_replacement(full_gold, target, len(full_gold), rng)]
            ls.resample_without_replacement(full_gold, target, rng)   # consumed in Exp 2
            if str(alpha) != "1":
                continue
            mk, g, pos = test.mask[idx], test.gold[idx], test.positions[idx]
            arms = {"none": test.probs[idx], "em": ls.em_prior_adjust(test.probs[idx], mk, pi_implicit, *em)[0]}
            arms.update({a: v[idx] for a, v in test_cf.items()})
            arms["cf_label_em"] = ls.em_prior_adjust(arms["cf_label"], mk, pi_implicit_cf, *em)[0]
            for a, probs in arms.items():
                mm = ls.matrix_metrics(probs, g, positions=pos)
                if a in ("none", "em") and round(mm.ece, 6) != stored[(a, d)]["ece"]:
                    raise SystemExit(f"{model}/{dataset} arm {a} draw {d}: does not reproduce Exp 2")
                shift.setdefault(a, {"ece": [], "macro_f1": []})
                shift[a]["ece"].append(mm.ece)
                shift[a]["macro_f1"].append(mm.macro_f1)
    return {"probes": info, "no_shift": no_shift,
            "shift_alpha1": {a: {k: float(np.mean(v)) for k, v in m.items()} for a, m in shift.items()},
            "n_draws": len(shift["none"]["ece"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-run", default="final_contentfree", help="results/<run id> with the probe predictions")
    ap.add_argument("--no-table", action="store_true", help="pipeline checks only: do not write the paper table")
    args = ap.parse_args()
    cf_run = RES / args.cf_run
    out = {f"{m}|{d}": evaluate(m, d, cf_run) for m, _ in SYSTEMS for d in DATASETS}
    final = args.cf_run == "final_contentfree"    # any other run (pipeline checks) writes to _scratch/ only
    dest = OUT / "content_free.json" if final else ROOT / "_scratch" / f"content_free_{Path(args.cf_run).name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"cf_run": args.cf_run, "results": out}, indent=1) + "\n", encoding="utf-8")

    def cell(r: dict, part: str, arm: str, key: str) -> str:
        v = r[part].get(arm)
        return "--" if v is None else f"{v[key]:.3f}"

    lines = ["\\begin{tabular}{ll" + "r" * 2 * len(DATASETS) + "}", "\\toprule",
             " & & \\multicolumn{" + str(len(DATASETS)) + "}{c}{Macro-F1, no shift} & \\multicolumn{"
             + str(len(DATASETS)) + "}{c}{ECE under shift ($\\alpha=1$)} \\\\",
             "System & Correction & " + " & ".join(DS[d] for d in DATASETS) * 1 + " & "
             + " & ".join(DS[d] for d in DATASETS) + " \\\\", "\\midrule"]
    for m, sname in SYSTEMS:
        for j, (a, label) in enumerate(ARMS):
            f1 = [cell(out[f"{m}|{d}"], "no_shift", a, "macro_f1") for d in DATASETS]
            ece = [cell(out[f"{m}|{d}"], "shift_alpha1", a, "ece") for d in DATASETS]
            lines.append(f"{sname if j == 0 else ''} & {label} & " + " & ".join(f1 + ece) + " \\\\")
            print(f"{m:11s} {label:28s} F1 " + " ".join(f1) + " | ECE shift " + " ".join(ece))
        lines.append("\\midrule" if m != SYSTEMS[-1][0] else "\\bottomrule")
    lines.append("\\end{tabular}")
    if final and not args.no_table:
        (ROOT / "paper" / "tables" / "tab_content_free.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
