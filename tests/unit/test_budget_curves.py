"""Exp 3 / Exp 4 budget curves (spec 10.3, 10.4; v1.2): name parsing, alignment, paired differences with a
bootstrap over test examples and seeds, crossover rule."""

from __future__ import annotations

import numpy as np
import pytest

from s1bench.application.budget_curves import align, crossover, curve_difference, parse_model_name
from s1bench.domain.examples import Label, LabelSet
from s1bench.domain.predictions import STATUS_FAILED, Prediction

LABELS = LabelSet("toy", "q?", (Label("a", "alpha", "d"), Label("b", "beta", "d"), Label("c", "gamma", "d")))
GOLD = ["abc"[i % 3] for i in range(60)]


def _preds(name, correct_every, seed=0):
    """Predictions that are right on examples i with (i + seed) % correct_every == 0 (always right if 1)."""
    out = []
    for i, g in enumerate(GOLD):
        right = (i + seed) % correct_every == 0
        pred = g if right else next(c for c in "abc" if c != g)
        probs = tuple(0.8 if c == pred else 0.1 for c in "abc")
        out.append(Prediction(f"toy:test:{i}", name, "toy", "test", ("a", "b", "c"), probs, pred, g))
    return out


def test_parse_model_name():
    assert parse_model_name("deberta-v3-large", "deberta-v3-large", 1001) == ("full", 1001)
    assert parse_model_name("deberta-v3-large@fulls1003", "deberta-v3-large", 1001) == ("full", 1003)
    assert parse_model_name("deberta-v3-large@b250s1002", "deberta-v3-large", 1001) == (250, 1002)
    assert parse_model_name("qwen3.5-4b-qlora@b100s1001", "qwen3.5-4b-qlora", 1001) == (100, 1001)
    assert parse_model_name("qwen3.5-4b", "qwen3.5-4b-qlora", 1001) is None
    assert parse_model_name("deberta-v3-large@b250s1002x", "deberta-v3-large", 1001) is None


def test_identical_system_gives_exactly_zero():
    ref = _preds("ref", 2)
    a = align({100: {1001: _preds("s", 2), 1002: _preds("s", 2)}}, ref, LABELS)
    (row,) = curve_difference(a, [100], "macro_f1", b=200, seed=1)
    assert row["estimate"] == 0.0 and row["ci_low"] == 0.0 and row["ci_high"] == 0.0


def test_better_system_positive_and_estimate_is_mean_over_seeds():
    ref = _preds("ref", 3)                                  # right on 1/3
    seeds = {1001: _preds("s", 1), 1002: _preds("s", 2)}     # always right / right on 1/2
    a = align({500: seeds}, ref, LABELS)
    (row,) = curve_difference(a, [500], "macro_f1", b=500, seed=1)
    assert row["ci_low"] > 0 and row["n_seeds"] == 2 and len(row["system_per_seed"]) == 2
    assert row["estimate"] == pytest.approx(np.mean(row["system_per_seed"]) - row["reference"])
    (ece,) = curve_difference(a, [500], "ece", b=100, seed=1)
    assert np.isfinite(ece["estimate"])


def test_missing_or_failed_rows_are_errors():
    ref = _preds("ref", 2)
    short = _preds("s", 1)[:-1]
    with pytest.raises(ValueError):
        align({100: {1001: short}}, ref, LABELS)
    failed = _preds("s", 1)
    failed[0] = Prediction(failed[0].example_id, "s", "toy", "test", ("a", "b", "c"), (), None, "a", status=STATUS_FAILED)
    with pytest.raises(ValueError):
        align({100: {1001: failed}}, ref, LABELS)
    blocked_ref = _preds("ref", 2)
    blocked_ref[3] = Prediction(blocked_ref[3].example_id, "ref", "toy", "test", ("a", "b", "c"), (), None, "a",
                                status=STATUS_FAILED)
    with pytest.raises(ValueError):
        align({100: {1001: _preds("s", 1)}}, blocked_ref, LABELS)


def test_crossover_rule_and_interpolation():
    sizes = {100: 100, 250: 250, 500: 500, "full": 3000}
    rows = [{"budget": 100, "estimate": -0.10, "ci_low": -0.15}, {"budget": 250, "estimate": -0.02, "ci_low": -0.06},
            {"budget": 500, "estimate": 0.03, "ci_low": 0.01}, {"budget": "full", "estimate": 0.2, "ci_low": 0.15}]
    c = crossover(rows, sizes)
    assert c["crossover"] == 500 and c["interval"] == "(250, 500]"
    assert c["interpolated_point_estimate"] == pytest.approx(250 + 0.02 * 250 / 0.05)
    never = [dict(r, ci_low=-1.0, estimate=-0.1) for r in rows]
    assert crossover(never, sizes) == {"crossover": "not reached", "interval": None,
                                       "interpolated_point_estimate": "not reached"}
    first = [dict(r, ci_low=0.01, estimate=0.05) for r in rows]
    assert crossover(first, sizes)["interval"] == "<= 100"
    with pytest.raises(ValueError):
        crossover(list(reversed(rows)), sizes)
