"""Label-budget subsets for Exp 3 (spec sec 10.3). Pure and deterministic."""

from __future__ import annotations

import hashlib
import math
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


def realistic_steps(n_train: int, batch: int, epochs: int, min_steps: int, max_steps: int) -> int:
    """Revision protocol (v1.3) step rule: `epochs` passes over the training part, but at least `min_steps` and at
    most `max_steps` optimizer steps (no switch in the number of epochs between budgets)."""
    return min(max_steps, max(min_steps, epochs * math.ceil(n_train / batch)))


def _dev_key(seed: int, eid: str) -> bytes:
    return hashlib.sha256(f"{seed}:budget-dev:{eid}".encode("utf-8")).digest()


def split_budget(subset: Sequence[Example], dev_share: float, seed: int) -> tuple[list[Example], list[Example]]:
    """Revision protocol (v1.3): split a budget subset into training and development examples, so that model selection
    uses only the budget's own labels.

    - The development part has round(dev_share * len(subset)) examples, at least 1 (if the subset has 2 or more).
    - It is drawn first from classes with at least two examples, at most n_c - 1 per class (so those classes keep a
      training example), with quotas proportional to the class counts (largest remainder). Only if these classes cannot
      supply enough examples do classes with a single example contribute theirs (they then have no training example;
      expected for CLINC150 and HWU64 at the smallest budgets).
    - Within a class, examples are taken in sha256(seed:budget-dev:id) order; deterministic for a given seed.
    Returns (train, dev), each in the order of `subset`.
    """
    n_dev = max(1, round(dev_share * len(subset))) if len(subset) >= 2 else 0
    by_class: dict[str, list[Example]] = defaultdict(list)
    for e in subset:
        by_class[e.label].append(e)
    for rows in by_class.values():
        rows.sort(key=lambda e: _dev_key(seed, e.id))
    classes = sorted(by_class, key=lambda c: _dev_key(seed, c))
    multi = [c for c in classes if len(by_class[c]) >= 2]
    capacity = {c: len(by_class[c]) - 1 for c in multi}
    take = {c: 0 for c in classes}
    want = min(n_dev, sum(capacity.values()))
    if want:
        total = sum(len(by_class[c]) for c in multi)
        exact = {c: want * len(by_class[c]) / total for c in multi}
        for c in multi:
            take[c] = min(capacity[c], int(exact[c]))
        order = sorted(multi, key=lambda c: (-(exact[c] - int(exact[c])), _dev_key(seed, c)))
        i = 0
        while sum(take.values()) < want:
            c = order[i % len(order)]
            if take[c] < capacity[c]:
                take[c] += 1
            i += 1
    for c in classes:   # fallback: single-example classes, in sha256 order
        if sum(take.values()) >= n_dev:
            break
        if len(by_class[c]) == 1:
            take[c] = 1
    dev_ids = {e.id for c in classes for e in by_class[c][: take[c]]}
    return [e for e in subset if e.id not in dev_ids], [e for e in subset if e.id in dev_ids]
