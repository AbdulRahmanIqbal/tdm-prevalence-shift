"""Exploratory (revision protocol v1.3, G3): end-to-end accuracy AND calibration on CLINC150 and HWU64 when the
intent group is chosen by the learned router (src/s1bench/application/router_candidates.py) instead of being given.

- Correctly routed test examples: the stored Exp 1 prediction (same candidate set, same order).
- Misrouted test examples (data/candidates_router/<ds>/test.jsonl, gold absent): the new predictions of run
  `final_router`. Generative systems: predictions/<model>/<ds>_test.jsonl. DeBERTa and TF-IDF: the same full-data
  model retrained with --save-full-probs (full_probs/<model>/<ds>_test.json), restricted to the routed candidates
  exactly like the frozen runs (softmax renormalised over the candidates). Check first: restricted to the STORED
  candidates, the retrained probabilities must reproduce the stored test predictions (argmax agreement, max |dp|).
- Metrics over all test examples (Exp 1 conventions, spec 13): accuracy with blocked rows wrong; top-label ECE
  (15 bins) and Brier on ok rows. Brier adds the gold label's term (0 - 1)^2 = 1 when gold is not offered. NLL is
  not reported (infinite on misrouted examples). Also: mean top-label confidence on the misrouted examples.

Writes results/final_analysis_calibrob/router_calibration.json and paper/tables/tab_router_calib.tex.
Run (after final_router is copied back): .venv/Scripts/python scripts/paper/router_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from s1bench.application.train_classifier import restrict  # noqa: E402
from s1bench.domain import metrics as mt  # noqa: E402
from s1bench.domain.candidates import CandidateSet  # noqa: E402
from s1bench.domain.predictions import STATUS_BLOCKED, STATUS_OK  # noqa: E402
from s1bench.infrastructure.storage.processed_store import YamlLabelRepository  # noqa: E402
from s1bench.infrastructure.storage.results_store import JsonlCandidateReader, read_predictions  # noqa: E402

RES = ROOT / "results"
OUT = RES / "final_analysis_calibrob"
DATASETS = ["clinc150", "hwu64"]
DS = {"clinc150": "CLINC150", "hwu64": "HWU64"}
SYSTEMS = [("jev", "final_jev", "Jev"), ("tev1-4b", "final_local", "Tev1-4B"), ("nimble-9b", "final_local", "Nimble-9B"),
           ("qwen3.5-4b", "final_local", "Qwen3.5-4B"), ("qwen3.5-9b", "final_local", "Qwen3.5-9B"),
           ("deberta-v3-large", "final_train", "DeBERTa-v3-large"), ("tfidf-lr", "final_train", "TF-IDF + LR")]
TRAINED = {"deberta-v3-large", "tfidf-lr"}


def row(cands, probs, gold, status):
    """(Scored for probability metrics or None, correct, brier term or None)."""
    if status == STATUS_BLOCKED or not probs:
        return None, False, None
    total = sum(probs)
    p = tuple(x / total for x in probs) if total > 0 else tuple(probs)
    s = mt.Scored(tuple(cands), p, gold)
    b = sum((q - (1.0 if c == gold else 0.0)) ** 2 for c, q in zip(cands, p)) + (0.0 if gold in cands else 1.0)
    return s, s.correct, b


def trained_new(model: str, ds: str, stored, router_c, labels_ids, NEW: Path):
    """Retrained full-probability output -> (reproduction check, {id: (cands, probs)} on the routed sets)."""
    fp = json.loads((NEW / "full_probs" / model / f"{ds}_test.json").read_text(encoding="utf-8"))
    if fp["label_ids"] != labels_ids:
        raise SystemExit(f"{model}/{ds}: label order differs")
    col = {lid: i for i, lid in enumerate(fp["label_ids"])}
    probs = dict(zip(fp["example_ids"], np.array(fp["probabilities"], dtype=float)))
    agree, max_dp = 0, 0.0
    for p in stored:
        q = restrict(probs[p.example_id], CandidateSet(p.example_id, p.candidates, -1), col)
        agree += p.candidates[int(np.argmax(q))] == p.predicted
        max_dp = max(max_dp, float(np.max(np.abs(np.array(q) - np.array(p.probabilities)))))
    check = {"argmax_agreement": agree / len(stored), "max_abs_dp": max_dp}
    new = {eid: (cs.candidates, restrict(probs[eid], cs, col)) for eid, cs in router_c.items()}
    return check, new


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--new-run", type=Path, default=RES / "final_router")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--no-table", action="store_true", help="pipeline check: do not write the paper table")
    args = ap.parse_args()
    NEW, OUT_ = args.new_run, args.out_dir
    res = {}
    for ds in DATASETS:
        router_c = JsonlCandidateReader(ROOT / "data" / "candidates_router").read_split(ds, "test")
        labels_ids = list(YamlLabelRepository(ROOT / "data" / "labels").load(ds).ids())
        r = {"n_misrouted": len(router_c)}
        for m, run, _ in SYSTEMS:
            stored = read_predictions(RES / run, m, ds, "test")
            if m in TRAINED:
                check, new = trained_new(m, ds, stored, router_c, labels_ids, NEW)
                new = {eid: (c, p, STATUS_OK) for eid, (c, p) in new.items()}
            else:
                check = None
                preds = {p.example_id: p for p in read_predictions(NEW, m, ds, "test")}
                if set(preds) != set(router_c):
                    raise SystemExit(f"{m}/{ds}: new predictions do not cover exactly the misrouted examples")
                for eid, p in preds.items():
                    if tuple(p.candidates) != tuple(router_c[eid].candidates):
                        raise SystemExit(f"{m}/{ds}/{eid}: candidate order differs from the router file")
                new = {eid: (p.candidates, p.probabilities, p.status) for eid, p in preds.items()}
            rows = {"gold": [], "e2e": []}
            conf_mis = []
            for p in stored:
                rows["gold"].append(row(p.candidates, p.probabilities, p.gold, p.status))
                if p.example_id in new:
                    c, q, st = new[p.example_id]
                    rr = row(c, q, p.gold, st)
                    if rr[0] is not None:
                        conf_mis.append(rr[0].confidence)
                else:
                    rr = rows["gold"][-1]
                rows["e2e"].append(rr)
            out = {}
            for k, rs in rows.items():
                ok = [x for x in rs if x[0] is not None]
                out[k] = {"accuracy": float(np.mean([x[1] for x in rs])),
                          "ece": mt.ece([x[0] for x in ok]), "brier": float(np.mean([x[2] for x in ok])),
                          "n_ok": len(ok), "n": len(rs)}
            out["misrouted_mean_confidence"] = float(np.mean(conf_mis)) if conf_mis else None
            out["misrouted_n_ok"] = len(conf_mis)
            if check is not None:
                out["reproduction_check"] = check
            r[m] = out
        res[ds] = r
        print(ds, json.dumps({m: {k: (round(v["ece"], 4) if isinstance(v, dict) and "ece" in v else v)
                                  for k, v in r[m].items()} for m, _, _ in SYSTEMS}))
    OUT_.mkdir(parents=True, exist_ok=True)
    (OUT_ / "router_calibration.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")

    if args.no_table:
        return
    f = lambda x: f"{100 * x:.1f}"  # noqa: E731
    lines = ["\\begin{tabular}{l" + "rrr" * len(DATASETS) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DS[d]}}}" for d in DATASETS) + " \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(DATASETS))),
             "System & " + " & ".join("ECE gold $\\to$ routed & Brier gold $\\to$ routed & conf.\\ misrouted"
                                      for _ in DATASETS) + " \\\\", "\\midrule"]
    for m, _, name in SYSTEMS:
        cells = []
        for d in DATASETS:
            x = res[d][m]
            cells += [f"{f(x['gold']['ece'])} $\\to$ {f(x['e2e']['ece'])}",
                      f"{f(x['gold']['brier'])} $\\to$ {f(x['e2e']['brier'])}",
                      f(x["misrouted_mean_confidence"]) if x["misrouted_mean_confidence"] is not None else "--"]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper" / "tables" / "tab_router_calib.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
