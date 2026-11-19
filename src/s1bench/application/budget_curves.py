"""Label-budget curves (Exp 3) and decision-training conditions (Exp 4): paired differences between a trained system
at each label budget and a reference system on the same test examples (spec 10.3, 10.4; v1.2).

For each budget b: Delta(b) = mean over training seeds of metric(system at b, seed) - metric(reference). The interval
comes from a bootstrap that resamples the test examples (shared by every budget and by the reference, so the
reference's own uncertainty is included) and, within each budget, the training seeds. Estimation, not hypothesis
tests: these comparisons are outside the 35-test Holm family.

Crossover (Exp 3, Macro-F1 against Jev): the smallest budget whose 95% interval lies above zero, reported as the
interval (previous budget, crossover budget]; "not reached" if no budget qualifies. Secondary point estimate: linear
interpolation of the point estimates at the first upward zero crossing, i.e. between the first budget whose estimate
is at or below zero and is followed by one above zero (Le Scao & Rush 2021); "<= first budget" if the first estimate
is already above zero.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from s1bench.application.prevalence_shift import to_matrices
from s1bench.application.primary_tests import fast_ece, fast_macro_f1
from s1bench.domain import label_shift as ls
from s1bench.domain.examples import LabelSet
from s1bench.domain.predictions import STATUS_OK, Prediction

METRICS = ("macro_f1", "ece")


def parse_model_name(name: str, prefix: str, first_seed: int) -> tuple[object, int] | None:
    """(budget, seed) of a trained-model prediction folder: `<prefix>@b<budget>s<seed>`, `<prefix>@fulls<seed>`,
    or the bare `<prefix>` (full data, first seed). None if the name belongs to another system."""
    if name == prefix:
        return "full", first_seed
    m = re.fullmatch(re.escape(prefix) + r"@(?:b(\d+)s(\d+)|fulls(\d+))", name)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2))) if m.group(1) else ("full", int(m.group(3)))


@dataclass(frozen=True)
class Aligned:
    ids: tuple[str, ...]
    gold: np.ndarray
    k: int
    ref: tuple[np.ndarray, np.ndarray]                                  # (pred, conf)
    points: dict[object, dict[int, tuple[np.ndarray, np.ndarray]]]     # budget -> seed -> (pred, conf)


def align(points: Mapping[object, Mapping[int, Sequence[Prediction]]], ref: Sequence[Prediction],
          labels: LabelSet) -> Aligned:
    """All systems on the same test examples, in the same order. Every reference row must be ok (Exp 1 scores a
    blocked row as wrong, so dropping it would change the reference's score), and every system must have an ok
    prediction for every reference example: a missing or failed row is an error, not a silent drop."""
    ok_ref = {p.example_id: p for p in ref if p.status == STATUS_OK and p.probabilities}
    if len(ok_ref) != len(ref):
        raise ValueError(f"reference: {len(ref) - len(ok_ref)} of {len(ref)} rows not ok (blocked, failed or duplicate)")
    ids = tuple(sorted(ok_ref))
    ref_m = to_matrices([ok_ref[i] for i in ids], labels)
    out: dict[object, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for budget, by_seed in points.items():
        for seed, preds in by_seed.items():
            ok = {p.example_id: p for p in preds if p.status == STATUS_OK and p.probabilities}
            missing = [i for i in ids if i not in ok]
            if missing:
                raise ValueError(f"budget {budget} seed {seed}: {len(missing)} reference examples missing or failed")
            m = to_matrices([ok[i] for i in ids], labels)
            if not np.array_equal(m.gold, ref_m.gold):
                raise ValueError(f"budget {budget} seed {seed}: gold labels differ from the reference")
            out.setdefault(budget, {})[seed] = (ls.argmax_by_position(m.probs, m.positions), m.probs.max(axis=1))
    ref_arr = (ls.argmax_by_position(ref_m.probs, ref_m.positions), ref_m.probs.max(axis=1))
    return Aligned(ids, ref_m.gold, ref_m.probs.shape[1], ref_arr, out)


def _metric(name: str, pred: np.ndarray, conf: np.ndarray, gold: np.ndarray, k: int) -> float:
    if name == "macro_f1":
        return fast_macro_f1(pred, gold, k)
    if name == "ece":
        return fast_ece(conf, (pred == gold).astype(float))
    raise ValueError(f"unknown metric {name!r}")


def curve_difference(a: Aligned, order: Sequence[object], metric: str, b: int = 10_000,
                     seed: int = 20261006) -> list[dict[str, object]]:
    """Point estimate, percentile 95% interval and per-seed values of Delta(b) for every budget in `order`."""
    rng = np.random.default_rng(seed)
    n = len(a.ids)
    ref_full = _metric(metric, *a.ref, a.gold, a.k)
    seeds = {p: sorted(a.points[p]) for p in order}
    per_seed = {p: [_metric(metric, *a.points[p][s], a.gold, a.k) for s in seeds[p]] for p in order}
    reps = np.empty((b, len(order)))
    for r in range(b):
        idx = rng.integers(0, n, n)
        g = a.gold[idx]
        ref_r = _metric(metric, a.ref[0][idx], a.ref[1][idx], g, a.k)
        for j, p in enumerate(order):
            pick = rng.integers(0, len(seeds[p]), len(seeds[p]))
            vals = [_metric(metric, a.points[p][seeds[p][q]][0][idx], a.points[p][seeds[p][q]][1][idx], g, a.k)
                    for q in pick]
            reps[r, j] = float(np.mean(vals)) - ref_r
    rows = []
    for j, p in enumerate(order):
        lo, hi = np.quantile(reps[:, j], [0.025, 0.975])
        rows.append({"budget": p, "n_seeds": len(seeds[p]), "seeds": seeds[p], "metric": metric,
                     "estimate": float(np.mean(per_seed[p]) - ref_full), "ci_low": float(lo), "ci_high": float(hi),
                     "system_mean": float(np.mean(per_seed[p])), "system_per_seed": per_seed[p],
                     "reference": ref_full, "n_test": n, "b": b})
    return rows


def crossover(rows: Sequence[Mapping[str, object]], sizes: Mapping[object, int]) -> dict[str, object]:
    """Crossover budget (smallest budget whose interval lies above zero) and the interpolated point estimate.
    `sizes` maps each budget label to its number of training examples ('full' -> train size)."""
    xs = [sizes[r["budget"]] for r in rows]
    if xs != sorted(xs):
        raise ValueError("rows must be ordered by budget size")
    hit = next((i for i, r in enumerate(rows) if r["ci_low"] > 0), None)
    if hit is None:
        result: dict[str, object] = {"crossover": "not reached", "interval": None}
    elif hit == 0:
        result = {"crossover": rows[0]["budget"], "interval": f"<= {xs[0]}"}
    else:
        result = {"crossover": rows[hit]["budget"], "interval": f"({xs[hit - 1]}, {xs[hit]}]"}
    est = [float(r["estimate"]) for r in rows]
    interp = None
    if est and est[0] > 0:
        interp = f"<= {xs[0]}"
    else:
        for i in range(1, len(est)):
            if est[i - 1] <= 0 < est[i]:
                interp = round(xs[i - 1] + (0 - est[i - 1]) * (xs[i] - xs[i - 1]) / (est[i] - est[i - 1]), 1)
                break
    result["interpolated_point_estimate"] = interp if interp is not None else "not reached"
    return result
