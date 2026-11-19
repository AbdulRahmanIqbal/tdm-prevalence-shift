from __future__ import annotations

import numpy as np
import pytest

from s1bench.domain import label_shift as ls
from s1bench.domain import metrics as m


def _calibrated_gaussian_data(n, prior, rng, sep=1.5):
    """1-D Gaussian classes; returns gold and the TRUE posterior under a uniform source prior."""
    k = len(prior)
    gold = rng.choice(k, size=n, p=prior)
    x = rng.normal(loc=gold * sep, scale=1.0)
    lik = np.exp(-0.5 * (x[:, None] - np.arange(k)[None, :] * sep) ** 2)
    post_uniform = lik / lik.sum(axis=1, keepdims=True)  # calibrated for a uniform source prior
    return gold, post_uniform


def test_dirichlet_centred_on_source_prior():
    rng = np.random.default_rng(0)
    pi_s = np.array([0.6, 0.3, 0.1])
    draws = np.array([ls.dirichlet_target_prior(pi_s, 10.0, rng) for _ in range(4000)])
    assert np.allclose(draws.mean(axis=0), pi_s, atol=0.01)


def test_resample_with_replacement_matches_target_counts():
    rng = np.random.default_rng(1)
    gold = np.repeat([0, 1, 2], [50, 30, 20])
    q = np.array([0.1, 0.1, 0.8])
    idx = ls.resample_with_replacement(gold, q, 10000, rng)
    assert np.allclose(np.bincount(gold[idx], minlength=3) / 10000, q, atol=0.02)


def test_resample_without_replacement_never_exceeds_available():
    rng = np.random.default_rng(2)
    gold = np.repeat([0, 1, 2], [50, 30, 20])
    q = np.array([0.1, 0.1, 0.8])
    idx = ls.resample_without_replacement(gold, q, rng)
    assert len(set(idx.tolist())) == len(idx)
    assert np.bincount(gold[idx], minlength=3)[2] <= 20


def test_em_recovers_target_prior_for_calibrated_model():
    rng = np.random.default_rng(3)
    uniform = np.full(3, 1 / 3)
    q = np.array([0.7, 0.2, 0.1])
    gold, probs = _calibrated_gaussian_data(20000, q, rng)
    mask = np.ones_like(probs, dtype=bool)
    _, q_hat, _ = ls.em_prior_adjust(probs, mask, uniform)
    assert np.allclose(q_hat, q, atol=0.02)


def test_em_respects_candidate_mask():
    probs = np.array([[0.6, 0.4, 0.0], [0.0, 0.3, 0.7]])
    mask = probs > 0
    adjusted, _, _ = ls.em_prior_adjust(probs, mask, np.full(3, 1 / 3))
    assert np.all(adjusted[~mask] == 0) and np.allclose(adjusted.sum(axis=1), 1)


def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(4)
    gold, probs = _calibrated_gaussian_data(20000, np.full(3, 1 / 3), rng)
    mask = np.ones_like(probs, dtype=bool)
    sharpened = ls.apply_temperature(probs, mask, 0.5)  # overconfident by factor 2
    # fitting must undo the sharpening: T_fit ≈ 1 / 0.5 = 2
    assert ls.fit_temperature(sharpened, mask, gold) == pytest.approx(2.0, rel=0.08)


def test_bcts_does_not_increase_dev_nll():
    rng = np.random.default_rng(5)
    gold, probs = _calibrated_gaussian_data(5000, np.array([0.5, 0.3, 0.2]), rng)
    mask = np.ones_like(probs, dtype=bool)
    biased = ls.apply_temperature(probs, mask, 0.6)
    params = ls.fit_bcts(biased, mask, gold)
    before = ls.matrix_metrics(biased, gold).nll
    after = ls.matrix_metrics(ls.apply_bcts(biased, mask, params), gold).nll
    assert after <= before + 1e-9


def test_matrix_metrics_agree_with_scalar_metrics():
    rng = np.random.default_rng(6)
    probs = rng.dirichlet(np.ones(4), size=300)
    gold = rng.integers(0, 4, size=300)
    rows = [m.Scored(("0", "1", "2", "3"), tuple(p), str(g)) for p, g in zip(probs, gold)]
    mm = ls.matrix_metrics(probs, gold)
    assert mm.accuracy == pytest.approx(m.accuracy(rows))
    assert mm.ece == pytest.approx(m.ece(rows))
    assert mm.brier == pytest.approx(m.brier(rows))
    assert mm.nll == pytest.approx(m.nll(rows))
    assert mm.aurc == pytest.approx(m.aurc(rows))
    assert mm.macro_f1 == pytest.approx(m.macro_f1(rows))


def test_coverage_threshold_and_selective_risk():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45], [0.51, 0.49]])
    gold = np.array([0, 0, 1, 0, 0])
    t = ls.coverage_threshold(probs.max(axis=1), 0.8)
    risk, cov = ls.selective_risk(probs, gold, t)
    assert t == pytest.approx(0.55) and cov == pytest.approx(0.8) and risk == pytest.approx(0.25)


def test_tie_break_by_candidate_position_matches_scalar_rule():
    probs = np.array([[0.5, 0.5, 0.0]])
    positions = np.array([[1, 0, 10**6]])  # class 1 is presented first
    assert ls.argmax_by_position(probs, positions)[0] == 1
    rows = [m.Scored(("1", "0", "2"), (0.5, 0.5, 0.0), "1")]
    assert rows[0].predicted == "1"


def test_stratified_bootstrap_keeps_every_class_count():
    from s1bench.domain.label_shift import stratified_bootstrap

    gold = np.array([0] * 50 + [1] * 3 + [2] * 9)
    rng = np.random.default_rng(0)
    for _ in range(200):
        idx = stratified_bootstrap(gold, rng)
        assert np.array_equal(np.bincount(gold[idx], minlength=3), [50, 3, 9])


def test_pa_tests_survive_tiny_classes():
    """Regression: an unstratified base sample lost 3-example classes (HWU64) and crashed."""
    from s1bench.application.prevalence_shift import Matrices
    from s1bench.application.primary_tests import pa_tests

    rng = np.random.default_rng(1)
    k = 12
    gold = np.concatenate([np.repeat(np.arange(k - 1), 20), [k - 1] * 3])
    p = rng.dirichlet(np.ones(k), size=len(gold))
    m = Matrices(p, np.ones_like(p, bool), gold, np.tile(np.arange(k), (len(gold), 1)), 0)
    res = pa_tests("toy", m, m, m, m, np.full(k, 1 / k), "implicit", b=60, n_priors=5)
    assert len(res) == 3


def test_implicit_prior_is_smoothed():
    from s1bench.domain.label_shift import implicit_prior

    pi = implicit_prior(np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]))
    assert pi.min() > 0 and abs(pi.sum() - 1) < 1e-12
