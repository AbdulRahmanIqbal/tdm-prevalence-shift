from __future__ import annotations

import numpy as np
import pytest

from s1bench.domain.stats import holm, paired_bootstrap


def test_identical_models_give_zero_difference_and_large_p():
    rng = np.random.default_rng(0)
    correct = rng.random(500) < 0.7
    acc = lambda idx: float(correct[idx].mean())  # noqa: E731
    r = paired_bootstrap(acc, acc, 500, b=2000)
    assert r.observed == 0.0 and r.ci_low == 0.0 and r.ci_high == 0.0 and r.p_value == 1.0


def test_clearly_better_model_is_significant_and_ci_excludes_zero():
    rng = np.random.default_rng(1)
    a = rng.random(800) < 0.85
    b = rng.random(800) < 0.65
    r = paired_bootstrap(lambda i: float(a[i].mean()), lambda i: float(b[i].mean()), 800, b=2000)
    assert r.ci_low > 0 and r.p_value < 0.01 and r.observed == pytest.approx(a.mean() - b.mean())


def test_holm_matches_textbook_example():
    adj = holm({"t1": 0.01, "t2": 0.04, "t3": 0.03, "t4": 0.005})
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert adj == pytest.approx({"t4": 0.02, "t1": 0.03, "t3": 0.06, "t2": 0.06})
