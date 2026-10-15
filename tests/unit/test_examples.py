from __future__ import annotations

from collections import Counter

import pytest

from s1bench.domain.examples import Label, LabelSet, RawExample, carve_dev, clean_splits, to_examples


def _rows(split, items):
    return [RawExample(split, i, text, label) for i, (text, label) in enumerate(items)]


def test_test_overlap_removed_from_train_and_dev_but_test_untouched():
    splits = {
        "train": _rows("train", [("Hello World", "a"), ("unique", "a")]),
        "dev": _rows("dev", [("hello   world", "a"), ("other", "b")]),
        "test": _rows("test", [("hello world", "a")]),
    }
    cleaned, report = clean_splits(splits)
    assert [r.text for r in cleaned["train"]] == ["unique"]
    assert [r.text for r in cleaned["dev"]] == ["other"]
    assert len(cleaned["test"]) == 1
    assert report.removed_overlap_with_test == {"train": 1, "dev": 1}


def test_conflicting_labels_removed_everywhere_in_train_dev_and_reported_in_test():
    splits = {
        "train": _rows("train", [("x", "a"), ("x", "b"), ("y", "a")]),
        "dev": _rows("dev", [("y", "b")]),
        "test": _rows("test", [("z", "a"), ("z", "b")]),
    }
    cleaned, report = clean_splits(splits)
    assert cleaned["train"] == [] and cleaned["dev"] == []
    assert report.conflicting_in_test == 1
    assert len(cleaned["test"]) == 2


def test_same_label_duplicates_and_dev_overlap_removed_from_train():
    splits = {
        "train": _rows("train", [("p", "a"), ("p", "a"), ("q", "b")]),
        "dev": _rows("dev", [("q", "b")]),
        "test": _rows("test", [("r", "a")]),
    }
    cleaned, report = clean_splits(splits)
    assert [r.text for r in cleaned["train"]] == ["p"]
    assert report.removed_same_label_duplicates_train == 1
    assert report.removed_train_overlap_with_dev == 1


def test_carve_dev_is_deterministic_stratified_and_disjoint():
    train = _rows("train", [(f"t{i}", "a" if i < 80 else "b") for i in range(100)])
    rest1, dev1 = carve_dev(train, 10, seed=7)
    rest2, dev2 = carve_dev(train, 10, seed=7)
    assert [r.source_index for r in dev1] == [r.source_index for r in dev2]
    assert Counter(r.label for r in dev1) == {"a": 8, "b": 2}
    assert not {r.source_index for r in dev1} & {r.source_index for r in rest1}
    assert len(rest1) + len(dev1) == 100


def test_to_examples_ids_and_unknown_label():
    labels = LabelSet("toy", "q?", (Label("a", "A", "d"), Label("b", "B", "d")))
    ex = to_examples("toy", "dev", [RawExample("train", 12, "t", "b")], labels)
    assert ex[0].id == "toy:dev:train000012" and ex[0].label_index == 1
    with pytest.raises(ValueError):
        to_examples("toy", "dev", [RawExample("train", 1, "t", "zzz")], labels)
