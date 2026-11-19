from __future__ import annotations

from collections import Counter

from s1bench.domain.budgets import stratified_subset
from s1bench.domain.examples import Example


def _ex(n_per_class):
    out = []
    for c, n in n_per_class.items():
        out += [Example(f"d:train:{c}{i}", "d", "train", f"t{c}{i}", c, 0) for i in range(n)]
    return out


def test_exact_size_proportional_and_every_class_present():
    rows = _ex({"a": 600, "b": 300, "c": 100})
    sub = stratified_subset(rows, 100, seed=1)
    counts = Counter(e.label for e in sub)
    assert len(sub) == 100 and counts == {"a": 60, "b": 30, "c": 10}


def test_budget_smaller_than_classes_gives_one_per_selected_class():
    rows = _ex({f"k{i}": 5 for i in range(150)})
    sub = stratified_subset(rows, 100, seed=2)
    assert len(sub) == 100 and len({e.label for e in sub}) == 100


def test_full_and_determinism_and_seed_variation():
    rows = _ex({"a": 50, "b": 50})
    assert len(stratified_subset(rows, None, 1)) == 100
    a = [e.id for e in stratified_subset(rows, 20, 1)]
    assert a == [e.id for e in stratified_subset(rows, 20, 1)]
    assert a != [e.id for e in stratified_subset(rows, 20, 2)]


def test_rare_class_keeps_minimum_one():
    rows = _ex({"a": 990, "b": 10})
    sub = stratified_subset(rows, 50, seed=3)
    assert len(sub) == 50 and Counter(e.label for e in sub)["b"] >= 1
