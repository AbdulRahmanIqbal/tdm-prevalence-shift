"""Revision protocol (v1.3; exploratory, G4): content-free probe files for content-free calibration
(see s1bench.domain.content_free).

Writes, for one dataset, a probe split named 'contentfree' that the ordinary `predict` runner can score:
- <out>/processed/<ds>/contentfree.jsonl   one Example per (option order, content-free text); label = ''
- <out>/candidates/<ds>/contentfree.jsonl  its candidate set (gold_position -1: there is no gold)
The probes use no test text and no test label; the ordering-matched design reads only the option orders of the
frozen test candidate file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from s1bench.domain.content_free import CONTENT_FREE_TEXTS, MAX_MATCHED_ORDERINGS, probe_orderings

SPLIT = "contentfree"


def probe_id(dataset: str, order_index: int, text_index: int) -> str:
    return f"{dataset}:{SPLIT}:o{order_index:04d}:t{text_index}"


def build_content_free_probes(dataset: str, data_dir: Path, out_dir: Path, seed: int) -> dict[str, object]:
    import yaml

    labels = yaml.safe_load((data_dir / "labels" / f"{dataset}.yaml").read_text(encoding="utf-8"))
    groups_file = data_dir / "candidates" / dataset / "label_groups.json"
    if groups_file.exists():
        groups = json.loads(groups_file.read_text(encoding="utf-8"))["groups"]
    else:
        groups = [sorted(str(lab["id"]) for lab in labels["labels"])]
    test_orders = [json.loads(line)["candidates"] for line in
                   (data_dir / "candidates" / dataset / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    for order in test_orders:   # every test set must be exactly one group (the gold label's group)
        if not any(set(order) == set(g) for g in groups):
            raise ValueError(f"{dataset}: a test candidate set is not one of the label groups")
    orders = probe_orderings(groups, test_orders, seed, dataset)

    examples, cands = [], []
    for o, row in enumerate(orders):
        for t, text in enumerate(CONTENT_FREE_TEXTS):
            eid = probe_id(dataset, o, t)
            examples.append({"dataset": dataset, "id": eid, "label": "", "label_index": -1,
                             "meta": {"order_index": o, "text_index": t, "group": row["group"],
                                      "designs": row["designs"]},
                             "split": SPLIT, "text": text})
            cands.append({"candidates": row["candidates"], "gold_position": -1, "id": eid})
    files = {}
    for sub, rows in (("processed", examples), ("candidates", cands)):
        path = out_dir / sub / dataset / f"{SPLIT}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        path.write_text(text, encoding="utf-8", newline="\n")
        files[path.relative_to(out_dir).as_posix()] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    n_rot = sum("rotation" in r["designs"] for r in orders)
    n_match = sum("matched" in r["designs"] for r in orders)
    return {"dataset": dataset, "n_groups": len(groups), "content_free_texts": list(CONTENT_FREE_TEXTS),
            "n_orders": len(orders), "n_rotation_orders": n_rot, "n_matched_orders": n_match,
            "n_distinct_test_orders": len({tuple(o) for o in test_orders}),
            "matched_design": n_match > 0, "max_matched_orderings": MAX_MATCHED_ORDERINGS,
            "n_probes": len(examples), "seed": seed, "files": files}
