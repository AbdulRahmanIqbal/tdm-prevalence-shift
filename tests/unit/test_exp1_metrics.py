from __future__ import annotations

from s1bench.application.exp1_metrics import exp1_row
from s1bench.domain.predictions import Prediction


def _p(i, probs, gold, status="ok"):
    return Prediction(f"toy:test:{i}", "m", "toy", "test", ("a", "b"), probs if status == "ok" else (),
                      None if status != "ok" else ("a" if probs[0] >= probs[1] else "b"), gold, status=status)


def test_blocked_is_wrong_and_abstain_failed_excluded():
    preds = [_p(0, (0.9, 0.1), "a"), _p(1, (0.2, 0.8), "b"), _p(2, (), "a", "blocked"), _p(3, (), "b", "failed")]
    r = exp1_row(preds)
    assert r["n_scored"] == 3 and r["status"] == {"ok": 2, "blocked": 1, "failed": 1}
    assert abs(r["accuracy"] - 2 / 3) < 1e-12                       # blocked counts as wrong
    # class a: tp 1, fn 1 (blocked) -> F1 2/3; class b: tp 1 -> F1 1; abstain not averaged in
    assert abs(r["macro_f1"] - (2 / 3 + 1) / 2) < 1e-12
    assert abs(r["aurc"] - (0 + 0 + 1 / 3) / 3) < 1e-12              # abstain rejected first (confidence 0)
    assert abs(r["ece"] - ((1 - 0.9) + (1 - 0.8)) / 2) < 1e-12      # ok rows only
    assert "sensitivity_blocked_excluded" in r                     # blocked 25% > 1%
    assert r["sensitivity_blocked_excluded"]["accuracy"] == 1.0


def test_jev_style_rounded_probabilities_are_renormalised():
    r = exp1_row([_p(0, (0.6, 0.39), "a")])                        # sums to 0.99
    assert abs(r["ece"] - (1 - 0.6 / 0.99)) < 1e-12
