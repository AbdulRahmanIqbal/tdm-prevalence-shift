from __future__ import annotations

import math

import pytest

from s1bench.domain.metrics import (
    Scored,
    accuracy,
    accuracy_at_coverage,
    aurc,
    brier,
    ece,
    macro_f1,
    mcc,
    nll,
)


def s(probs, gold, cands=("a", "b", "c")):
    return Scored(tuple(cands), tuple(probs), gold)


def test_perfect_confident_predictions():
    rows = [s((1.0, 0.0, 0.0), "a"), s((0.0, 1.0, 0.0), "b"), s((0.0, 0.0, 1.0), "c")]
    assert accuracy(rows) == 1.0 and macro_f1(rows) == 1.0 and mcc(rows) == pytest.approx(1.0)
    assert ece(rows) == 0.0 and brier(rows) == 0.0 and nll(rows) == pytest.approx(0.0, abs=1e-9)
    assert aurc(rows) == 0.0


def test_ece_known_value():
    # 2 examples in bin of confidence 0.8, one correct -> |0.5 - 0.8| = 0.3
    rows = [s((0.8, 0.1, 0.1), "a"), s((0.8, 0.1, 0.1), "b")]
    assert ece(rows) == pytest.approx(0.3)


def test_brier_and_nll_known_values():
    r = s((0.5, 0.3, 0.2), "a")
    assert brier([r]) == pytest.approx(0.25 + 0.09 + 0.04)
    assert nll([r]) == pytest.approx(-math.log(0.5))


def test_macro_f1_and_mcc_on_imbalanced_example():
    rows = [s((0.9, 0.05, 0.05), "a")] * 3 + [s((0.9, 0.05, 0.05), "b")]
    assert accuracy(rows) == 0.75
    assert macro_f1(rows) == pytest.approx((2 * 3 / (2 * 3 + 1) + 0.0) / 2)
    assert mcc(rows) == 0.0  # constant predictor


def test_aurc_rewards_confidence_ranking_and_is_tie_order_invariant():
    good = [s((0.9, 0.05, 0.05), "a"), s((0.4, 0.35, 0.25), "b")]  # wrong one has low confidence
    bad = [s((0.9, 0.05, 0.05), "b"), s((0.4, 0.35, 0.25), "a")]   # wrong one has high confidence
    assert aurc(good) < aurc(bad)
    tied = [s((0.5, 0.3, 0.2), "a"), s((0.5, 0.3, 0.2), "b")]
    assert aurc(tied) == aurc(list(reversed(tied)))


def test_accuracy_at_coverage():
    rows = [s((0.9, 0.05, 0.05), "a"), s((0.6, 0.2, 0.2), "b")]
    assert accuracy_at_coverage(rows, 0.5) == 1.0 and accuracy_at_coverage(rows, 1.0) == 0.5


def test_macro_f1_and_mcc_match_sklearn_on_random_data():
    sk = pytest.importorskip("sklearn.metrics")
    import random

    rng = random.Random(0)
    labels = ["a", "b", "c", "d"]
    rows = []
    for _ in range(300):
        cands = tuple(rng.sample(labels, 4))
        raw = [rng.random() for _ in cands]
        tot = sum(raw)
        rows.append(Scored(cands, tuple(x / tot for x in raw), rng.choice(labels)))
    gold = [r.gold for r in rows]
    pred = [r.predicted for r in rows]
    assert macro_f1(rows) == pytest.approx(sk.f1_score(gold, pred, average="macro"))
    assert mcc(rows) == pytest.approx(sk.matthews_corrcoef(gold, pred))


def test_accuracy_at_coverage_is_order_independent_with_ties():
    from s1bench.domain.metrics import Scored, accuracy_at_coverage

    rows = [Scored(("a", "b"), (0.9, 0.1), "a"), Scored(("a", "b"), (0.6, 0.4), "a"),
            Scored(("a", "b"), (0.6, 0.4), "b"), Scored(("a", "b"), (0.6, 0.4), "b")]
    v1 = accuracy_at_coverage(rows, 0.5)
    v2 = accuracy_at_coverage(list(reversed(rows)), 0.5)
    assert v1 == v2 == (1 + 1 / 3) / 2
