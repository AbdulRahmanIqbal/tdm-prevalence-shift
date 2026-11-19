"""Revision protocol (v1.3; exploratory, added after the results): candidate sets chosen by a learned intent-group
router instead of the gold label's group (CLINC150, HWU64).

Router: TF-IDF (word 1-2-grams, lowercase, sublinear tf) + logistic regression (lbfgs, max_iter 5000), trained on the
training split to predict the fixed label group (data/candidates/<ds>/label_groups.json) of the gold label; C chosen
on the development split by group accuracy (C_GRID, first best wins). Same router as
scripts/paper/router_end_to_end.py.

- Correctly routed test example: the candidate set is the stored one (same options, same order), so the stored
  prediction of every system applies unchanged. Nothing new is written for these examples.
- Misrouted test example: the candidate set is the routed group (the gold label is absent; gold_position = -1), in a
  seeded per-example order (sha256(candidate_seed:router:id)). Every system is then run on these sets only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from s1bench.domain.candidates import example_rng
from s1bench.domain.examples import Example

C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
ROUTER_DATASETS = ("clinc150", "hwu64")


def routed_order(group: Sequence[str], seed: int, example_id: str) -> list[str]:
    """Seeded per-example order of a routed group (deterministic, independent of the gold label)."""
    out = sorted(group)
    example_rng(seed, f"router:{example_id}").shuffle(out)
    return out


def train_router(train: Sequence[Example], dev: Sequence[Example], group_of: dict[str, int]):
    """Returns (predict(texts) -> group ids, info)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True, min_df=1)
    x_tr = vec.fit_transform([e.text for e in train])
    x_dev = vec.transform([e.text for e in dev])
    y_tr = np.array([group_of[e.label] for e in train])
    y_dev = np.array([group_of[e.label] for e in dev])
    dev_acc = {c: float((LogisticRegression(C=c, max_iter=5000).fit(x_tr, y_tr).predict(x_dev) == y_dev).mean())
               for c in C_GRID}
    best = max(C_GRID, key=lambda c: dev_acc[c])      # max() keeps the first of equal values
    clf = LogisticRegression(C=best, max_iter=5000).fit(x_tr, y_tr)
    return (lambda texts: clf.predict(vec.transform(list(texts)))), {"C": best, "dev_group_accuracy": dev_acc}


def build_router_candidates(dataset: str, data_dir: Path, out_dir: Path, seed: int) -> dict[str, object]:
    """Writes out_dir/<dataset>/test.jsonl (misrouted test examples only) and returns a report."""
    from s1bench.infrastructure.storage.processed_store import JsonlProcessedReader

    groups = json.loads((data_dir / "candidates" / dataset / "label_groups.json").read_text(encoding="utf-8"))["groups"]
    group_of = {lid: g for g, members in enumerate(groups) for lid in members}
    reader = JsonlProcessedReader(data_dir / "processed")
    train, dev, test = (reader.read_split(dataset, s) for s in ("train", "dev", "test"))
    stored = {}
    for line in (data_dir / "candidates" / dataset / "test.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        stored[r["id"]] = r
    predict, info = train_router(train, dev, group_of)
    routed = predict([e.text for e in test])
    rows = []
    for e, g in zip(test, routed):
        gold_group = group_of[e.label]
        if set(stored[e.id]["candidates"]) != set(groups[gold_group]):
            raise ValueError(f"{dataset}/{e.id}: stored candidate set is not the gold label's group")
        if int(g) == gold_group:
            continue
        rows.append({"id": e.id, "candidates": routed_order(groups[int(g)], seed, e.id), "gold_position": -1,
                     "routed_group": int(g), "gold_group": gold_group})
    path = out_dir / dataset / "test.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    return {"dataset": dataset, "router": info, "n_test": len(test), "n_misrouted": len(rows),
            "test_group_accuracy": round(1 - len(rows) / len(test), 6), "n_groups": len(groups),
            "file": path.relative_to(out_dir).as_posix(),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
