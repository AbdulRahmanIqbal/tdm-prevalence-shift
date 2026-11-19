from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from s1bench.domain.candidates import (
    balanced_gold_positions,
    balanced_label_groups,
    build_candidate_set,
    example_rng,
    nearest_labels,
)

LABELS = [f"l{i:02d}" for i in range(30)]
GROUPS = [LABELS[:15], LABELS[15:]]


def test_large_label_set_uses_the_gold_group():
    cs = build_candidate_set("d:test:x1", "l05", LABELS, GROUPS, max_candidates=20, seed=1)
    assert sorted(cs.candidates) == GROUPS[0] and cs.candidates[cs.gold_position] == "l05"


def test_every_member_of_a_group_gets_the_same_option_set():
    """The fix for the text-free 'centrality' cue: the set never depends on which member is gold."""
    sets = {frozenset(build_candidate_set(f"d:test:{i}", g, LABELS, GROUPS, 20, 1).candidates)
            for i, g in enumerate(GROUPS[1])}
    assert sets == {frozenset(GROUPS[1])}


def test_small_label_set_is_all_labels_shuffled():
    labels = ["a", "b", "c"]
    cs = build_candidate_set("d:test:x2", "b", labels, None, max_candidates=20, seed=1)
    assert sorted(cs.candidates) == labels and cs.candidates[cs.gold_position] == "b"


def test_deterministic_per_example_and_varies_across_examples():
    a = build_candidate_set("d:test:x3", "l01", LABELS, GROUPS, 20, seed=1)
    assert a == build_candidate_set("d:test:x3", "l01", LABELS, GROUPS, 20, seed=1)
    others = {build_candidate_set(f"d:test:{i}", "l01", LABELS, GROUPS, 20, 1).gold_position for i in range(50)}
    assert len(others) > 5


def test_unknown_gold_or_missing_group_raises():
    with pytest.raises(ValueError):
        build_candidate_set("d:test:x4", "nope", LABELS, GROUPS, 20, 1)
    with pytest.raises(ValueError):
        build_candidate_set("d:test:x5", "l01", LABELS, [LABELS[2:]], 20, 1)


def test_balanced_label_groups_sizes_determinism_and_similarity():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 8))
    emb = np.vstack([c + 0.05 * rng.normal(size=(21, 8)) for c in centers])   # 3 clear clusters of 21
    ids = [f"c{i // 21}_{i:02d}" for i in range(63)]
    groups = balanced_label_groups(ids, emb.tolist(), target_size=20, seed=5, restarts=5)
    assert sorted(len(g) for g in groups) == [21, 21, 21]
    assert sorted(sorted({x.split("_")[0] for x in g}) for g in groups) == [["c0"], ["c1"], ["c2"]]
    assert groups == balanced_label_groups(ids, emb.tolist(), target_size=20, seed=5, restarts=5)
    uneven = balanced_label_groups(ids[:62], emb[:62].tolist(), target_size=20, seed=5, restarts=5)
    assert max(len(g) for g in uneven) - min(len(g) for g in uneven) <= 1 and sum(map(len, uneven)) == 62


def test_example_rng_is_process_independent():
    assert example_rng(7, "abc").random() == example_rng(7, "abc").random()


def test_nearest_labels_orders_by_similarity():
    n = nearest_labels(["x", "y", "z"], [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    assert n["x"] == ["y", "z"]


def test_balanced_gold_positions_are_exactly_balanced_and_deterministic():
    ids = [f"d:test:{i}" for i in range(103)]
    slots = balanced_gold_positions(ids, 15, seed=3)
    counts = Counter(slots.values())
    assert set(counts) == set(range(15)) and max(counts.values()) - min(counts.values()) <= 1
    assert slots == balanced_gold_positions(list(reversed(ids)), 15, seed=3)
    cs = build_candidate_set("d:test:5", "l03", LABELS, GROUPS, 20, 3, gold_position=slots["d:test:5"])
    assert cs.gold_position == slots["d:test:5"] and cs.candidates[cs.gold_position] == "l03"


def test_frozen_files_have_no_text_free_cue():
    """On the frozen CLINC150/HWU64 files every gold label maps to exactly one option set."""
    import json
    from pathlib import Path

    for d in ("clinc150", "hwu64"):
        root = Path("data")
        if not (root / "candidates" / d / "test.jsonl").exists():
            pytest.skip("frozen data not present")
        gold = {json.loads(x)["id"]: json.loads(x)["label"]
                for x in (root / "processed" / d / "test.jsonl").read_text(encoding="utf-8").splitlines()}
        sets: dict[str, set] = {}
        for line in (root / "candidates" / d / "test.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            sets.setdefault(gold[r["id"]], set()).add(frozenset(r["candidates"]))
        assert all(len(v) == 1 for v in sets.values())
