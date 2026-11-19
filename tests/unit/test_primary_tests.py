from __future__ import annotations

import numpy as np

from s1bench.application.primary_tests import aligned_matrices, pa_tests, with_holm
from s1bench.domain.examples import Label, LabelSet
from s1bench.domain.predictions import Prediction

LABELS = LabelSet("toy", "q", tuple(Label(str(i), f"c{i}", "d") for i in range(3)))


def _preds(model, probs, gold, status="ok"):
    out = []
    for i, (p, g) in enumerate(zip(probs, gold)):
        out.append(Prediction(example_id=f"toy:test:{i:05d}", model=model, dataset="toy", split="test",
                              candidates=("0", "1", "2"), probabilities=tuple(p), predicted=str(int(np.argmax(p))),
                              gold=str(g), status=status))
    return out


def _data(n, rng):
    gold = rng.integers(0, 3, size=n)
    logits = rng.normal(size=(n, 3)) + 2.0 * np.eye(3)[gold]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return probs, gold


def test_identical_models_give_zero_pa1_pa2():
    rng = np.random.default_rng(0)
    probs, gold = _data(300, rng)
    a, b, dropped = aligned_matrices(_preds("a", probs, gold), _preds("b", probs, gold), LABELS)
    res = pa_tests("toy", a, a, b, b, np.full(3, 1 / 3), "implicit", b=50)
    pa1, pa2 = res[0], res[1]
    assert dropped == 0
    assert pa1.estimate == 0 and pa1.p_value == 1.0 and pa2.estimate == 0 and pa2.p_value == 1.0


def test_alignment_drops_examples_failed_for_either_model():
    rng = np.random.default_rng(1)
    probs, gold = _data(10, rng)
    a = _preds("a", probs, gold)
    b = _preds("b", probs, gold)
    b[3] = Prediction(**{**b[3].__dict__, "status": "failed", "probabilities": ()})
    ma, mb, dropped = aligned_matrices(a, b, LABELS)
    assert len(ma.gold) == 9 and dropped == 1 and np.array_equal(ma.gold, mb.gold)


def test_holm_flags_incomplete_family():
    rng = np.random.default_rng(2)
    probs, gold = _data(200, rng)
    a, b, _ = aligned_matrices(_preds("a", probs, gold), _preds("b", probs, gold), LABELS)
    rows = with_holm(pa_tests("toy", a, a, b, b, np.full(3, 1 / 3), "implicit", b=30))
    assert all(r["holm_family_complete"] is False for r in rows) and len(rows) == 3


def test_fast_metrics_match_matrix_metrics():
    from s1bench.application.primary_tests import fast_ece, fast_macro_f1
    from s1bench.domain import label_shift as ls

    rng = np.random.default_rng(3)
    probs, gold = _data(500, rng)
    m = ls.matrix_metrics(probs, gold)
    pred = probs.argmax(axis=1)
    assert abs(fast_macro_f1(pred, gold, 3) - m.macro_f1) < 1e-12
    assert abs(fast_ece(probs.max(axis=1), (pred == gold).astype(float)) - m.ece) < 1e-12


def test_pair_tests_identical_models_and_direction():
    from s1bench.application.primary_tests import pair_tests

    rng = np.random.default_rng(4)
    probs, gold = _data(400, rng)
    a, b, _ = aligned_matrices(_preds("a", probs, gold), _preds("b", probs, gold), LABELS)
    f1, ece = pair_tests("P1", "toy", a, b, b=200)
    assert f1.test == "P1-F1" and ece.test == "P1-ECE"
    assert f1.estimate == 0 and f1.p_value == 1.0 and ece.estimate == 0
    # model A = perfect one-hot predictions -> higher Macro-F1 (delta > 0), significant
    perfect = np.eye(3)[gold] * 0.98 + 0.01
    a2, b2, _ = aligned_matrices(_preds("a", perfect, gold), _preds("b", probs, gold), LABELS)
    f1b, _ = pair_tests("P1", "toy", a2, b2, b=200)
    assert f1b.estimate > 0 and f1b.ci_low > 0 and f1b.p_value < 0.01


def test_holm_family_completeness_dev_checks_and_conservative_m():
    import pytest

    from s1bench.application.primary_tests import expected_family_keys, holm_family

    keys = expected_family_keys(["fpb", "trec6"])
    assert len(expected_family_keys(["clinc150", "hwu64", "fpb", "hatexplain", "trec6"])) == 35
    rows = [{"test": k.split(":")[0], "dataset": k.split(":")[1], "p_value": 0.01, "dev_check": False} for k in sorted(keys)]
    out, status = holm_family(rows, keys)
    assert status["complete"] and all(r["holm_family_complete"] for r in out)
    # partial family: m = full family size (conservative), never COMPLETE
    out, status = holm_family(rows[:2], keys)
    assert not status["complete"] and abs(out[0]["p_holm"] - 0.01 * len(keys)) < 1e-12 and len(status["missing"]) == len(keys) - 2
    # a dev-check file blocks completeness even with the exact key set
    out, status = holm_family([{**rows[0], "dev_check": True, "source_file": "x.json"}] + rows[1:], keys)
    assert not status["complete"] and status["dev_check_files"] == ["x.json"]
    with pytest.raises(ValueError):            # duplicates
        holm_family(rows + [rows[0]], keys)
    with pytest.raises(ValueError):            # outside the preregistered family
        holm_family(rows + [{"test": "P3-F1", "dataset": "fpb", "p_value": 0.5}], keys)


def test_input_check_flags_incomplete_failed_and_unknown_status():
    from s1bench.application.primary_tests import input_check

    rng = np.random.default_rng(5)
    probs, gold = _data(200, rng)
    preds = _preds("a", probs, gold)
    ids = [p.example_id for p in preds]
    assert input_check(preds, ids)["problems"] == []
    assert any("missing" in x for x in input_check(preds[:150], ids)["problems"])
    failed = [Prediction(**{**p.__dict__, "status": "failed", "probabilities": ()}) if i < 3 else p for i, p in enumerate(preds)]
    assert any("failed 3/200" in x for x in input_check(failed, ids)["problems"])
    blocked = [Prediction(**{**p.__dict__, "status": "blocked", "probabilities": ()}) if i < 30 else p for i, p in enumerate(preds)]
    chk = input_check(blocked, ids)
    assert chk["problems"] == [] and chk["status"]["blocked"] == 30   # reported, not a refusal
    weird = [Prediction(**{**preds[0].__dict__, "status": "error"})] + preds[1:]
    assert any("unknown statuses" in x for x in input_check(weird, ids)["problems"])


def test_pa1_resampled_control_removes_resampling_bias_without_shift():
    # No shift at all: alpha huge and train prior = test prior, so shifted and no-shift samples come from the
    # same distribution. With the resampled control (v1.1.1) PA1 is centred on 0 even for two different models.
    rng = np.random.default_rng(1)
    probs, gold = _data(600, rng)
    sharp = probs ** 3 / (probs ** 3).sum(axis=1, keepdims=True)        # overconfident second model
    a, b, _ = aligned_matrices(_preds("a", probs, gold), _preds("b", sharp, gold), LABELS)
    prior = np.bincount(gold, minlength=3) / len(gold)
    pa1 = pa_tests("toy", a, a, b, b, prior, "implicit", alpha=1e6, b=40, n_priors=10)[0]
    assert pa1.ci_low <= 0 <= pa1.ci_high and abs(pa1.estimate) < 0.01
