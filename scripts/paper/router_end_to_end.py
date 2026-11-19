"""Exploratory (added after the results): end-to-end accuracy on CLINC150 and HWU64 when the intent group is chosen by
a learned router instead of being given (the paper's systems decide within the gold label's group).

Router: TF-IDF (word 1-2-grams, sublinear tf, as the TF-IDF baseline) + logistic regression trained on the training
split to predict the group of the gold label; C chosen on the development split by group accuracy. When the router
picks the gold label's group, the candidate set is exactly the stored one (same options, same order), so the system's
stored prediction applies; when it picks another group, the gold label is not offered and the decision is wrong.
End-to-end accuracy = share of test examples with a correct route and a correct stored prediction. Calibration of the
end-to-end decisions needs new predictions on the misrouted examples (not done here).

Writes results/final_analysis_calibrob/router.json and paper/tables/tab_router.tex.
Run: .venv/Scripts/python scripts/paper/router_end_to_end.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from s1bench.infrastructure.storage.processed_store import JsonlProcessedReader  # noqa: E402
from s1bench.infrastructure.storage.results_store import read_predictions  # noqa: E402

RES = ROOT / "results"
OUT = RES / "final_analysis_calibrob"
DATASETS = ["clinc150", "hwu64"]
DS = {"clinc150": "CLINC150", "hwu64": "HWU64"}
SYSTEMS = [("jev", "final_jev", "Jev"), ("tev1-4b", "final_local", "Tev1-4B"), ("nimble-9b", "final_local", "Nimble-9B"),
           ("qwen3.5-4b", "final_local", "Qwen3.5-4B"), ("qwen3.5-9b", "final_local", "Qwen3.5-9B"),
           ("deberta-v3-large", "final_train", "DeBERTa-v3-large"), ("tfidf-lr", "final_train", "TF-IDF + LR")]
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]


def label_groups(dataset: str) -> dict[str, int]:
    """Group of each label = the candidate set offered to the examples of that label (fixed groups)."""
    groups: dict[frozenset, int] = {}
    out: dict[str, int] = {}
    for split in ("dev", "test"):
        for line in (ROOT / "data" / "candidates" / dataset / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            key = frozenset(r["candidates"])
            gid = groups.setdefault(key, len(groups))
            gold = r["candidates"][r["gold_position"]]
            if out.setdefault(gold, gid) != gid:
                raise SystemExit(f"{dataset}: label {gold} in two groups")
    return out


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    OUT.mkdir(parents=True, exist_ok=True)
    reader = JsonlProcessedReader(ROOT / "data" / "processed")
    res = {}
    for ds in DATASETS:
        group = label_groups(ds)
        split = {s: reader.read_split(ds, s) for s in ("train", "dev", "test")}
        vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True, min_df=1)
        x = {s: (vec.fit_transform if s == "train" else vec.transform)([e.text for e in split[s]]) for s in split}
        y = {s: np.array([group[e.label] for e in split[s]]) for s in split}
        best = max(C_GRID, key=lambda c: (LogisticRegression(C=c, max_iter=5000).fit(x["train"], y["train"])
                                          .predict(x["dev"]) == y["dev"]).mean())
        clf = LogisticRegression(C=best, max_iter=5000).fit(x["train"], y["train"])
        routed_ok = dict(zip([e.id for e in split["test"]], clf.predict(x["test"]) == y["test"]))
        r = {"router": {"C": best, "accuracy": float(np.mean(list(routed_ok.values()))), "n_groups": len(set(group.values()))}}
        for m, run, _ in SYSTEMS:
            preds = read_predictions(RES / run, m, ds, "test")
            within = np.array([p.predicted == p.gold for p in preds])
            e2e = np.array([routed_ok[p.example_id] and p.predicted == p.gold for p in preds])
            r[m] = {"within_group": float(within.mean()), "end_to_end": float(e2e.mean())}
        res[ds] = r
        print(ds, json.dumps(r))
    (OUT / "router.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    lines = ["\\begin{tabular}{l" + "rr" * len(DATASETS) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS[d]}}}" for d in DATASETS) + " \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(DATASETS))),
             "Accuracy & " + " & ".join("gold group & routed" for _ in DATASETS) + " \\\\", "\\midrule",
             "Router (group accuracy) & " + " & ".join(f"-- & {res[d]['router']['accuracy']:.3f}" for d in DATASETS) + " \\\\",
             "\\midrule"]
    for m, _, name in SYSTEMS:
        lines.append(f"{name} & " + " & ".join(f"{res[d][m]['within_group']:.3f} & {res[d][m]['end_to_end']:.3f}"
                                               for d in DATASETS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper" / "tables" / "tab_router.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
