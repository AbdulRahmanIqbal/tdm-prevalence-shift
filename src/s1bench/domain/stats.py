"""Statistical tests (spec sec 11). numpy only.

- Paired bootstrap: resample the SAME example indices for both models; statistic = metric(A) - metric(B).
  95% percentile CI; two-sided p = 2 * min(P(delta* <= 0), P(delta* >= 0)), capped at 1 (spec 11.2).
- Holm step-down adjustment over the preregistered family (35 tests, spec 11.1).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedResult:
    observed: float     # metric(A) - metric(B) on the full sample
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    b: int


def paired_bootstrap(metric_a: Callable[[np.ndarray], float], metric_b: Callable[[np.ndarray], float],
                     n: int, b: int = 10_000, seed: int = 20261006, alpha: float = 0.05) -> PairedResult:
    """metric_x(idx) computes the metric of model x on examples `idx` (same idx for both models)."""
    rng = np.random.default_rng(seed)
    full = np.arange(n)
    observed = metric_a(full) - metric_b(full)
    deltas = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        deltas[i] = metric_a(idx) - metric_b(idx)
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
    return PairedResult(float(observed), float(lo), float(hi), float(p), n, b)


def holm(p_values: Mapping[str, float], m: int | None = None) -> dict[str, float]:
    """Holm-adjusted p-values (monotone step-down), keyed like the input. `m` = family size; when only part
    of a preregistered family is present, pass the full size (conservative: missing tests count as p = 1)."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = max(len(items), m or 0)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted
