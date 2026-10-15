from __future__ import annotations

import numpy as np
import pytest

from s1bench.application.run_predictions import RunPredictions
from s1bench.application.train_classifier import TrainAndPredict
from s1bench.domain.candidates import CandidateSet
from s1bench.domain.examples import Example, Label, LabelSet
from s1bench.domain.predictions import Prediction

LABELS = LabelSet("toy", "q?", (Label("a", "alpha", "d"), Label("b", "beta", "d"), Label("c", "gamma", "d")))


def _ex(split, n):
    return [Example(f"toy:{split}:{i}", "toy", split, f"text {i}", "abc"[i % 3] if i % 5 else "a", 0) for i in range(n)]


class Reader:
    def __init__(self):
        self.data = {s: _ex(s, n) for s, n in (("train", 60), ("dev", 30), ("test", 30))}

    def read_split(self, dataset, split):
        return self.data[split]


class Cands:
    def read_split(self, dataset, split):
        return {e.id: CandidateSet(e.id, ("a", "b", "c"), "abc".index(e.label)) for e in Reader().data[split]}


class Labels:
    def load(self, dataset):
        return LABELS


class Store:
    def __init__(self):
        self.written = {}

    def write(self, model, dataset, split, preds):
        self.written[(model, split)] = preds


class FakeModel:
    """fails for the seeds in FAIL; records release() calls."""
    FAIL: set[int] = set()
    released: list[int] = []

    def __init__(self, hp, seed):
        self.seed, self.failed, self.log = seed, seed in self.FAIL, {"seed": seed}

    def fit(self, *a, **k):
        assert "dev_candidates" in k          # selection must be candidate-restricted
        return self

    def predict_proba(self, texts, ids):
        return np.tile([0.2, 0.5, 0.3], (len(texts), 1))

    def release(self):
        FakeModel.released.append(self.seed)


def _uc(store=None):
    return TrainAndPredict(FakeModel, [1.0], Reader(), Cands(), Labels(), store or Store(), reserve_seeds=(1006, 1007, 1008))


def test_reserve_seeds_shared_per_dataset_budget_max_three():
    FakeModel.FAIL, FakeModel.released = {1001, 1002, 1006, 1007}, []
    uc = _uc()
    r1 = uc.execute("m", "toy", 30, 1001)        # 1001 fails -> 1006 fails -> 1007 fails -> 1008 ok
    r2 = uc.execute("m", "toy", 30, 1002)        # pool exhausted: no restart, 1002 stays failed
    assert [x["restart_seed"] for x in r1["restarts"]] == [1006, 1007, 1008] and not r1["still_failed"]
    assert r2["restarts"] == [] and r2["still_failed"] and r2["scored_as"] == "majority_class"
    assert {1001, 1006, 1007} <= set(FakeModel.released)   # failed models were freed


def test_still_failed_seed_scored_as_majority_class():
    FakeModel.FAIL = {1001, 1006, 1007, 1008}
    store = Store()
    _uc(store).execute("m", "toy", 30, 1001)
    preds = store.written[("m", "test")]
    assert {p.predicted for p in preds} == {"a"}               # train majority class
    assert len({p.probabilities for p in preds}) == 1           # constant predictor


def test_budget_not_below_train_size_is_rejected():
    FakeModel.FAIL = set()
    with pytest.raises(ValueError):
        _uc().execute("m", "toy", 60, 1001)


class SpyModel:
    name = "spy"

    def __init__(self):
        self.seen = []

    def predict(self, ex, cs, labels):
        self.seen.append((ex.label, ex.label_index, cs.gold_position))
        return Prediction(ex.id, "spy", ex.dataset, ex.split, cs.candidates, (1 / 3,) * 3, cs.candidates[0], ex.label)

    def describe(self):
        return {}


def test_models_never_receive_gold():
    spy, store = SpyModel(), Store()
    RunPredictions(spy, Reader(), Cands(), Labels(), store).execute("toy", "test", None, 1)
    assert set(spy.seen) == {("", -1, -1)}
    assert [p.gold for p in store.written[("spy", "test")]] == [e.label for e in Reader().data["test"]]


def test_tfidf_converges_and_logs_iterations():
    from s1bench.infrastructure.models.tfidf_lr import TfidfLogReg

    texts = [f"word{i % 3} token{i}" for i in range(60)]
    m = TfidfLogReg(1000.0, 1).fit(texts, ["abc"[i % 3] for i in range(60)])
    assert m.log["converged"] and m.log["tol"] == 1e-8 and m.log["n_iter"] > 1


def test_jev_retries_backoff_and_errors(monkeypatch):
    from s1bench.infrastructure.models import jev_cloudflare as jc

    sleeps = []
    monkeypatch.setattr(jc.time, "sleep", lambda s: sleeps.append(s))
    m = jc.JevCloudflareModel("acct", "tok")
    calls = {"n": 0}

    def flaky(payload):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("HTTP 500: upstream")
        return {"success": True, "result": {"model": "jev-1.13.0", "result": {"model": "jev-1.13.0", "answers": {
            "label": {"choice": "beta", "confidence": 0.5,
                      "probabilities": {"alpha": 0.2, "beta": 0.5, "gamma": 0.3}}}}}}

    monkeypatch.setattr(m, "_post", flaky)
    ex = Example("toy:test:0", "toy", "test", "t", "b", 1)
    p = m.predict(ex, CandidateSet(ex.id, ("a", "b", "c"), 1), LABELS)
    assert p.status == "ok" and p.predicted == "b" and len(p.raw["attempt_errors"]) == 2
    assert sleeps == [1.5, 3.0]                                  # exponential
    monkeypatch.setattr(m, "_post", lambda payload: {"success": True, "result": {"unexpected": 1}})
    calls["n"], sleeps[:] = 0, []
    bad = m.predict(ex, CandidateSet(ex.id, ("a", "b", "c"), 1), LABELS)
    assert bad.status == "failed" and len(sleeps) == 1          # parse error: one retry only


def test_skip_test_never_reads_or_writes_test():
    FakeModel.FAIL = set()

    class NoTestReader(Reader):
        def read_split(self, dataset, split):
            assert split != "test", "test split must not be read with predict_test=False"
            return super().read_split(dataset, split)

    class NoTestCands(Cands):
        def read_split(self, dataset, split):
            assert split != "test"
            return super().read_split(dataset, split)

    store = Store()
    uc = TrainAndPredict(FakeModel, [1.0], NoTestReader(), NoTestCands(), Labels(), store)
    r = uc.execute("m", "toy", 30, 1001, predict_test=False)
    assert r["test_macro_f1"] is None and ("m", "test") not in store.written and ("m", "dev") in store.written


def test_selected_model_timed_on_dev_texts_only_after_warmup(monkeypatch):
    import s1bench.application.train_classifier as tc

    monkeypatch.setattr(tc, "TIMING_N", 15)
    seen = []

    class Timed(FakeModel):
        def time_single(self, texts):
            seen.extend(texts)
            return [100.0] * tc.TIMING_WARMUP + [float(i) for i in range(len(texts) - tc.TIMING_WARMUP)]

        def timing_info(self):
            return {"device": "cuda", "gpu": "RTX 3060"}

    Timed.FAIL = set()
    r = TrainAndPredict(Timed, [1.0], Reader(), Cands(), Labels(), Store()).execute("m", "toy", None, 1001)
    lat = r["latency"]
    assert lat["n"] == 15 and lat["split"] == "dev" and lat["p50_ms"] == 7.0 and lat["gpu"] == "RTX 3060"
    assert set(seen) <= {e.text for e in Reader().data["dev"]} and len(seen) == 25
    assert _uc().execute("m", "toy", None, 1001)["latency"] is None     # models without time_single


def test_tfidf_time_single_returns_one_latency_per_text():
    from s1bench.infrastructure.models.tfidf_lr import TfidfLogReg

    m = TfidfLogReg(1.0, 1).fit([f"word{i % 3}" for i in range(30)], ["abc"[i % 3] for i in range(30)])
    ms = m.time_single(["word1", "word2", "unseen"])
    assert len(ms) == 3 and all(x >= 0 for x in ms) and m.timing_info()["device"] == "cpu"


def test_jev_connection_refused_is_failed_not_blocked(monkeypatch):
    import urllib.error

    from s1bench.infrastructure.models import jev_cloudflare as jc

    monkeypatch.setattr(jc.time, "sleep", lambda s: None)
    m = jc.JevCloudflareModel("acct", "tok")
    ex = Example("toy:test:0", "toy", "test", "t", "b", 1)
    refused = ConnectionRefusedError(10061, "No connection could be made because the target machine actively refused it")

    def down(payload):
        raise urllib.error.URLError(refused)

    monkeypatch.setattr(m, "_post", down)
    p = m.predict(ex, CandidateSet(ex.id, ("a", "b", "c"), 1), LABELS)
    assert p.status == "failed" and "refused" in p.error          # transport error: 5 attempts, then failed
    assert p.error.count("|") == 4

    def blocked(payload):
        raise RuntimeError('HTTP 400: {"errors": [{"message": "request refused by content policy"}]}')

    monkeypatch.setattr(m, "_post", blocked)
    assert m.predict(ex, CandidateSet(ex.id, ("a", "b", "c"), 1), LABELS).status == "blocked"


def test_failed_rule_threshold_is_candidate_restricted():
    from s1bench.infrastructure.models.deberta_classifier import candidate_majority_accuracy

    col = {c: i for i, c in enumerate("abcdef")}
    y_train = [0] * 5 + [3] * 4 + [1, 2, 4, 5]          # a most frequent overall, d second
    # two candidate groups like CLINC label groups: {a,b,c} and {d,e,f}
    cands = [list("abc"), list("abc"), list("def"), list("def")]
    y_dev = [0, 1, 3, 4]
    # fallback predicts a in group 1 and d in group 2 -> 2/4 correct; all-label majority would be 1/4
    assert candidate_majority_accuracy(y_train, y_dev, cands, col) == 0.5
    tie = [list("bc")]                                    # b and c both 1 in train -> earliest position (b)
    assert candidate_majority_accuracy(y_train, [1], tie, col) == 1.0


def test_non_failed_model_preferred_over_failed_with_higher_dev_score():
    class ByLr(FakeModel):
        def __init__(self, hp, seed):
            super().__init__(hp, seed)
            self.hp, self.failed = hp, hp == 1.0            # LR 1.0 always fails (also after restarts)

        def predict_proba(self, texts, ids):              # failed LR has the HIGHER dev score
            row = [0.6, 0.2, 0.2] if self.hp == 1.0 else [0.2, 0.5, 0.3]
            return np.tile(row, (len(texts), 1))

    store = Store()
    r = TrainAndPredict(ByLr, [1.0, 2.0], Reader(), Cands(), Labels(), store).execute("m", "toy", None, 1001)
    assert r["selected_c"] == 2.0 and not r["still_failed"] and r["scored_as"] == "model"
    assert len(r["restarts"]) == 3 and all("failed_run" in x for x in r["restarts"])
