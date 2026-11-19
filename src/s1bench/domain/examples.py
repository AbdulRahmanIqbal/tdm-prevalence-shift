"""Entities and pure rules for processed data (spec sec 4). No I/O here."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class Label:
    id: str
    name: str
    description: str
    raw_name: str = ""


@dataclass(frozen=True)
class LabelSet:
    dataset: str
    task_question: str
    labels: tuple[Label, ...]

    def ids(self) -> list[str]:
        return [label.id for label in self.labels]

    def index_of(self, label_id: str) -> int:
        return self.ids().index(label_id)


@dataclass(frozen=True)
class RawExample:
    """One row as read from the raw source, before cleaning."""

    source_split: str  # split of the raw source the row came from (e.g. "train" for a carved dev row)
    source_index: int  # position in that raw source split
    text: str
    label: str  # label id as used in data/labels/<dataset>.yaml
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Example:
    id: str
    dataset: str
    split: str
    text: str
    label: str
    label_index: int
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass
class CleaningReport:
    before: dict[str, int] = field(default_factory=dict)
    after: dict[str, int] = field(default_factory=dict)
    removed_overlap_with_test: dict[str, int] = field(default_factory=dict)
    removed_train_overlap_with_dev: int = 0
    removed_conflicting: dict[str, int] = field(default_factory=dict)
    conflicting_in_test: int = 0
    removed_same_label_duplicates_train: int = 0

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def normalize(text: str) -> str:
    """Comparison-only normalization (spec 4.3, v1.0.4): lowercase, every run of non-letter/non-digit
    characters (any script) -> one space, so punctuation-only variants ("frozen" / "frozen!") count as the
    same text. Texts that would normalize to "" (only symbols/emoji) fall back to lowercase + collapsed
    whitespace. Models always see the original text."""
    t = text.strip().lower()
    key = re.sub(r"[\W_]+", " ", t).strip()
    return key if key else re.sub(r"\s+", " ", t)


def carve_dev(train: Sequence[RawExample], size: int, seed: int) -> tuple[list[RawExample], list[RawExample]]:
    """Split `size` dev examples off `train`, stratified by the train class distribution.

    Per-class quotas use largest-remainder rounding; within each class, rows are chosen with a
    seeded RNG over rows sorted by source_index, so the result is deterministic.
    """
    if not 0 < size < len(train):
        raise ValueError(f"dev size {size} must be between 0 and {len(train)}")
    by_label: dict[str, list[RawExample]] = defaultdict(list)
    for row in sorted(train, key=lambda r: r.source_index):
        by_label[row.label].append(row)

    exact = {label: size * len(rows) / len(train) for label, rows in by_label.items()}
    quota = {label: int(v) for label, v in exact.items()}
    remainder = size - sum(quota.values())
    for label in sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))[:remainder]:
        quota[label] += 1

    rng = random.Random(seed)
    dev_keys: set[int] = set()
    for label in sorted(by_label):
        rows = by_label[label]
        k = min(quota[label], len(rows) - 1)  # never empty a class in train
        dev_keys.update(r.source_index for r in rng.sample(rows, k))
    dev = [r for r in train if r.source_index in dev_keys]
    rest = [r for r in train if r.source_index not in dev_keys]
    return rest, [RawExample("train", r.source_index, r.text, r.label, r.meta) for r in dev]


def clean_splits(splits: Mapping[str, Sequence[RawExample]]) -> tuple[dict[str, list[RawExample]], CleaningReport]:
    """Apply spec sec 4.3 cleaning rules. The test split is never modified."""
    report = CleaningReport(before={s: len(splits.get(s, ())) for s in SPLITS})
    test = list(splits.get("test", ()))
    test_texts = {normalize(r.text) for r in test}

    # Rule 2: train/dev rows that exactly match a test text are removed from train/dev.
    kept: dict[str, list[RawExample]] = {}
    for s in ("train", "dev"):
        rows = list(splits.get(s, ()))
        kept[s] = [r for r in rows if normalize(r.text) not in test_texts]
        report.removed_overlap_with_test[s] = len(rows) - len(kept[s])

    # Rule 3: texts with conflicting labels within train+dev are removed from both.
    labels_by_text: dict[str, set[str]] = defaultdict(set)
    for s in ("train", "dev"):
        for r in kept[s]:
            labels_by_text[normalize(r.text)].add(r.label)
    conflicting = {t for t, labels in labels_by_text.items() if len(labels) > 1}
    for s in ("train", "dev"):
        before = len(kept[s])
        kept[s] = [r for r in kept[s] if normalize(r.text) not in conflicting]
        report.removed_conflicting[s] = before - len(kept[s])
    test_labels: dict[str, set[str]] = defaultdict(set)
    for r in test:
        test_labels[normalize(r.text)].add(r.label)
    report.conflicting_in_test = sum(1 for labels in test_labels.values() if len(labels) > 1)

    # Train rows identical to a dev text are removed from train (keeps the dev set unseen).
    dev_texts = {normalize(r.text) for r in kept["dev"]}
    before = len(kept["train"])
    kept["train"] = [r for r in kept["train"] if normalize(r.text) not in dev_texts]
    report.removed_train_overlap_with_dev = before - len(kept["train"])

    # Rule 4: same-label duplicates in train: keep the first occurrence.
    seen: set[str] = set()
    deduped = []
    for r in kept["train"]:
        key = normalize(r.text)
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    report.removed_same_label_duplicates_train = len(kept["train"]) - len(deduped)
    kept["train"] = deduped

    kept["test"] = test
    report.after = {s: len(kept[s]) for s in SPLITS}
    return kept, report


def to_examples(dataset: str, split: str, rows: Iterable[RawExample], labels: LabelSet) -> list[Example]:
    """Assign stable ids: <dataset>:<split>:<source_split><source_index:06d>."""
    known = set(labels.ids())
    out = []
    for r in rows:
        if r.label not in known:
            raise ValueError(f"{dataset}: label {r.label!r} not in label file")
        meta = {"source_split": r.source_split, "source_index": r.source_index, **r.meta}
        out.append(
            Example(
                id=f"{dataset}:{split}:{r.source_split}{r.source_index:06d}",
                dataset=dataset,
                split=split,
                text=r.text,
                label=r.label,
                label_index=labels.index_of(r.label),
                meta=meta,
            )
        )
    ids = [e.id for e in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{dataset}/{split}: duplicate example ids")
    return out


def label_counts(rows: Iterable[RawExample | Example]) -> dict[str, int]:
    return dict(sorted(Counter(r.label for r in rows).items()))
