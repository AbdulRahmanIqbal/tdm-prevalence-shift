"""Revision protocol (v1.3, G4): content-free probes and content-free calibration (Zhao et al. 2021)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from s1bench.domain import content_free as cf


def test_rotations_put_every_label_at_every_position_once():
    rots = cf.rotation_orderings(["d", "a", "c", "b"], seed=5, tag="x:0")
    assert len(rots) == 4 and len(set(rots)) == 4
    for pos in range(4):
        assert sorted(r[pos] for r in rots) == ["a", "b", "c", "d"]
    assert rots == cf.rotation_orderings(["a", "b", "c", "d"], seed=5, tag="x:0")   # depends on the set only


def test_probe_orderings_dedupe_and_matched_limit():
    groups = [["0", "1", "2"]]
    test_orders = [("2", "1", "0"), ("0", "1", "2"), ("2", "1", "0")]
    rows = cf.probe_orderings(groups, test_orders, seed=1, dataset="toy")
    orders = [tuple(r["candidates"]) for r in rows]
    assert len(orders) == len(set(orders))                          # every order probed once
    assert sum("rotation" in r["designs"] for r in rows) == 3
    assert {tuple(r["candidates"]) for r in rows if "matched" in r["designs"]} == {("2", "1", "0"), ("0", "1", "2")}
    many = [tuple(map(str, np.random.default_rng(i).permutation(8))) for i in range(cf.MAX_MATCHED_ORDERINGS + 50)]
    rows = cf.probe_orderings([[str(i) for i in range(8)]], many, seed=1, dataset="toy")
    assert not any("matched" in r["designs"] for r in rows)        # too many distinct test orders


def test_label_level_weights_average_over_positions():
    probes = [(("a", "b"), (0.8, 0.2)), (("b", "a"), (0.4, 0.6))]
    w = cf.label_level_weights(probes, ["a", "b", "c"])
    assert np.allclose(w[:2], [0.7, 0.3]) and np.isnan(w[2])


def test_adjust_uniform_weights_is_identity_and_removes_bias():
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.0, 0.8]])
    mask = np.array([[True, True, True], [True, False, True]])
    assert np.allclose(cf.content_free_adjust(probs, mask, cf.weights_matrix_label(np.full(3, 1 / 3), 2)), probs)
    w = np.array([0.5, 0.25, 0.25])
    q = cf.content_free_adjust(probs, mask, cf.weights_matrix_label(w, 2))
    assert np.allclose(q.sum(1), 1) and q[1, 1] == 0
    assert np.allclose(q[0], np.array([0.6 / 0.5, 0.3 / 0.25, 0.1 / 0.25]) / (1.2 + 1.2 + 0.4))
    with pytest.raises(ValueError):                                 # offered label without a weight
        cf.content_free_adjust(probs, mask, cf.weights_matrix_label(np.array([0.5, 0.5, np.nan]), 2))


def test_matched_weights_follow_the_example_order():
    w = cf.matched_weights([(("a", "b"), (0.9, 0.1)), (("a", "b"), (0.7, 0.3)), (("b", "a"), (0.6, 0.4))])
    m = cf.weights_matrix_matched([("a", "b"), ("b", "a")], w, ["a", "b"])
    assert np.allclose(m, [[0.8, 0.2], [0.4, 0.6]])


def test_probe_files_have_no_text_and_no_gold(tmp_path):
    from s1bench.application.content_free_probes import build_content_free_probes

    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "toy.yaml").write_text(
        "dataset: toy\ntask_question: q\nlabels:\n" + "".join(
            f"  - id: {i}\n    name: n{i}\n    description: d{i}\n" for i in range(3)), encoding="utf-8")
    (tmp_path / "candidates" / "toy").mkdir(parents=True)
    (tmp_path / "candidates" / "toy" / "test.jsonl").write_text("".join(
        json.dumps({"id": f"t{i}", "candidates": c, "gold_position": 0}) + "\n"
        for i, c in enumerate([["2", "1", "0"], ["0", "2", "1"]])), encoding="utf-8")
    r1 = build_content_free_probes("toy", tmp_path, tmp_path / "out", seed=3)
    r2 = build_content_free_probes("toy", tmp_path, tmp_path / "out2", seed=3)
    assert r1["files"] == r2["files"]                               # deterministic
    assert r1["n_probes"] == r1["n_orders"] * len(cf.CONTENT_FREE_TEXTS) and r1["n_matched_orders"] == 2
    ex = [json.loads(x) for x in (tmp_path / "out/processed/toy/contentfree.jsonl").read_text("utf-8").splitlines()]
    ca = [json.loads(x) for x in (tmp_path / "out/candidates/toy/contentfree.jsonl").read_text("utf-8").splitlines()]
    assert {e["text"] for e in ex} == set(cf.CONTENT_FREE_TEXTS) and {e["label"] for e in ex} == {""}
    assert [e["id"] for e in ex] == [c["id"] for c in ca] and {c["gold_position"] for c in ca} == {-1}


def test_predict_contentfree_requires_both_dirs():
    from s1bench.cli.main import main

    assert main(["predict", "--model", "jev", "--dataset", "fpb", "--split", "contentfree"]) == 2
    assert main(["predict", "--model", "jev", "--dataset", "fpb", "--processed-dir", "x"]) == 2
