"""Revision protocol (v1.3): budget-internal development data, the realistic step rule, early stopping and the
full-probability output. The frozen behaviour (all options off) is covered by test_training_and_runner.py."""

from __future__ import annotations

import numpy as np

from s1bench.application.train_classifier import TrainAndPredict
from s1bench.domain.budgets import realistic_steps

from test_training_and_runner import Cands, Labels, Reader


class JsonStore:
    def __init__(self):
        self.written, self.json = {}, {}

    def write(self, model, dataset, split, preds):
        self.written[(model, split)] = preds

    def write_json(self, rel, payload):
        self.json[rel] = payload


class Recording:
    """Records the texts it was fitted and selected on."""
    calls: list[dict] = []

    def __init__(self, hp, seed):
        self.seed, self.failed, self.log = seed, False, {}

    def fit(self, texts, labels, dev_texts, dev_labels, ids, dev_candidates=None):
        Recording.calls.append({"train": list(texts), "dev": list(dev_texts)})
        return self

    def predict_proba(self, texts, ids):
        return np.tile([0.2, 0.5, 0.3], (len(texts), 1))

    def release(self):
        pass


def test_realistic_steps_rule():
    assert realistic_steps(80, 16, 20, 500, 2000) == 500       # small budget: minimum
    assert realistic_steps(800, 16, 20, 500, 2000) == 1000     # 20 epochs
    assert realistic_steps(4000, 16, 20, 500, 2000) == 2000    # capped


def test_budget_dev_used_for_selection_and_full_dev_written():
    Recording.calls = []
    store = JsonStore()
    r = TrainAndPredict(Recording, [1.0], Reader(), Cands(), Labels(), store, dev_share=0.2).execute(
        "m@b40s1001", "toy", 40, 1001)
    call = Recording.calls[0]
    assert len(call["train"]) == 32 and len(call["dev"]) == 8
    assert not set(call["train"]) & set(call["dev"])
    assert all(t.startswith("text") for t in call["dev"])
    train_texts = {e.text for e in Reader().data["train"]}
    assert set(call["dev"]) <= train_texts                      # selection data come from the budget
    assert len(store.written[("m@b40s1001", "dev")]) == 30      # written dev = the full development split
    assert r["n_train"] == 32 and r["n_dev_selection"] == 8


def test_frozen_mode_selects_on_full_dev_and_full_data_ignores_dev_share():
    Recording.calls = []
    TrainAndPredict(Recording, [1.0], Reader(), Cands(), Labels(), JsonStore()).execute("m", "toy", 40, 1001)
    assert len(Recording.calls[-1]["dev"]) == 30
    TrainAndPredict(Recording, [1.0], Reader(), Cands(), Labels(), JsonStore(), dev_share=0.2).execute(
        "m", "toy", None, 1001)
    assert len(Recording.calls[-1]["dev"]) == 30                # full data keeps the official dev split


def test_full_probs_written_for_test():
    store = JsonStore()
    TrainAndPredict(Recording, [1.0], Reader(), Cands(), Labels(), store, save_full_probs=True).execute(
        "m", "toy", None, 1001)
    payload = store.json["full_probs/m/toy_test.json"]
    assert payload["label_ids"] == ["a", "b", "c"] and len(payload["probabilities"]) == 30
    assert len(payload["example_ids"]) == 30


def test_router_order_is_seeded_and_gold_free():
    from s1bench.application.router_candidates import routed_order

    g = ["x", "y", "z", "w", "v"]
    a, b = routed_order(g, 7, "e1"), routed_order(list(reversed(g)), 7, "e1")
    assert a == b and sorted(a) == sorted(g)                     # depends only on the set, the seed and the id
    assert any(routed_order(g, 7, f"e{i}") != a for i in range(2, 10))


def test_runner_only_listed_scores_router_examples_only():
    from s1bench.application.run_predictions import RunPredictions
    from s1bench.domain.candidates import CandidateSet

    from test_training_and_runner import SpyModel, Store

    test_ids = [e.id for e in Reader().data["test"]][:4]

    class Listed:
        def read_split(self, dataset, split):
            return {eid: CandidateSet(eid, ("b", "c"), -1) for eid in test_ids}   # gold may be absent

    spy, store = SpyModel(), Store()
    RunPredictions(spy, Reader(), Listed(), Labels(), store, only_listed=True).execute("toy", "test", None, 1)
    written = store.written[("spy", "test")]
    assert [p.example_id for p in written] == test_ids
    assert {s[2] for s in spy.seen} == {-1}
