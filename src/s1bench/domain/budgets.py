"""Label-budget subsets for Exp 3 (spec sec 10.3). Pure and deterministic."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

from s1bench.domain.examples import Example


def _key(seed: int, eid: str) -> bytes:
    return hashlib.sha256(f"{seed}:budget:{eid}".encode("utf-8")).digest()


def stratified_subset(examples: Sequence[Example], budget: int | None, seed: int) -> list[Example]:
    """Budget-sized subset of the training set, stratified by class.

    - budget None or >= len(examples): the full set.
    - Quotas proportional to class frequency (largest remainder). If the budget allows it, every class
      gets at least one example; if budget < number of classes, `budget` classes get one example each
      (chosen deterministically) — expected for CLINC150/HWU64 at 100 (spec 10.3).
    - Within a class, examples are taken in sha256(seed:id) order, so a given seed is reproducible and
      different seeds give different subsets.
    """
    if budget is None or budget >= len(examples):
        return list(examples)
    by_class: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        by_class[e.label].append(e)
    for rows in by_class.values():
        rows.sort(key=lambda e: _key(seed, e.id))
    classes = sorted(by_class, key=lambda c: _key(seed, c))

    if budget < len(classes):
        return [by_class[c][0] for c in classes[:budget]]

    n = len(examples)
    exact = {c: budget * len(by_class[c]) / n for c in classes}
    quota = {c: max(1, int(exact[c])) for c in classes}
    # fix the total to exactly `budget` (largest remainder up, smallest down)
    diff = budget - sum(quota.values())
    order_up = sorted(classes, key=lambda c: (-(exact[c] - int(exact[c])), _key(seed, c)))
    order_down = sorted(classes, key=lambda c: (exact[c] - int(exact[c]), _key(seed, c)))
    i = 0
    while diff > 0:
        c = order_up[i % len(classes)]
        if quota[c] < len(by_class[c]):
            quota[c] += 1
            diff -= 1
        i += 1
    i = 0
    while diff < 0:
        c = order_down[i % len(classes)]
        if quota[c] > 1:
            quota[c] -= 1
            diff += 1
        i += 1
    return [e for c in classes for e in by_class[c][: quota[c]]]
