from __future__ import annotations

from collections import Counter

from s1bench.domain.budgets import split_budget, stratified_subset
from s1bench.domain.examples import Example


def _ex(n_per_class):
    out = []
    for c, n in n_per_class.items():
        out += [Example(f"d:train:{c}{i}", "d", "train", f"t{c}{i}", c, 0) for i in range(n)]
    return out


def test_size_partition_and_proportional():
    subset = _ex({"a": 60, "b": 30, "c": 10})
    train, dev = split_budget(subset, 0.2, seed=1)
    assert len(dev) == 20 and len(train) == 80
    assert {e.id for e in train} | {e.id for e in dev} == {e.id for e in subset}
    assert not {e.id for e in train} & {e.id for e in dev}
    assert Counter(e.label for e in dev) == {"a": 12, "b": 6, "c": 2}


def test_multi_example_classes_keep_a_training_example():
    subset = _ex({"a": 2, "b": 2, "c": 2, "d": 2, "e": 2})   # capacity 1 per class
    train, dev = split_budget(subset, 0.2, seed=3)
    assert len(dev) == 2
    assert set(Counter(e.label for e in train)) == {"a", "b", "c", "d", "e"}


def test_singletons_used_only_when_needed():
    subset = _ex({f"s{i}": 1 for i in range(10)})          # CLINC150 at budget 100: one example per class
    train, dev = split_budget(subset, 0.2, seed=5)
    assert len(dev) == 2 and len(train) == 8
    mixed = _ex({"a": 3, **{f"s{i}": 1 for i in range(7)}})  # n=10, want 2: class a can give 2
    _, dev2 = split_budget(mixed, 0.2, seed=5)
    assert Counter(e.label for e in dev2) == {"a": 2}


def test_deterministic_and_seed_dependent_and_order_kept():
    subset = stratified_subset(_ex({"a": 50, "b": 50}), 40, seed=1)
    t1, d1 = split_budget(subset, 0.2, seed=1)
    t2, d2 = split_budget(subset, 0.2, seed=1)
    assert [e.id for e in d1] == [e.id for e in d2] and [e.id for e in t1] == [e.id for e in t2]
    _, d3 = split_budget(subset, 0.2, seed=2)
    assert {e.id for e in d1} != {e.id for e in d3}
    pos = {e.id: i for i, e in enumerate(subset)}
    assert [pos[e.id] for e in t1] == sorted(pos[e.id] for e in t1)


def test_tiny_subsets():
    assert split_budget(_ex({"a": 1}), 0.2, seed=1) == (_ex({"a": 1}), [])
    train, dev = split_budget(_ex({"a": 2}), 0.2, seed=1)
    assert len(dev) == 1 and len(train) == 1
