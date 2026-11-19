"""Metrics (spec sec 9). Pure Python, operating on aligned (candidates, probabilities, gold) records.

Conventions (spec sec 9, frozen):
- Prediction = argmax over the candidate set; confidence = max candidate probability.
- ECE: top-label, 15 equal-width bins (Guo et al. 2017; same as Deusser et al. and Alexandari et al.).
- Brier: multi-class, sum over the example's candidates of (p_k - y_k)^2, averaged over examples.
- NLL: -log p(gold), probability clipped at 1e-12.
- AURC: area under the risk-coverage curve, examples sorted by decreasing confidence (ties kept
  together by averaging within tied groups), trapezoid-free discrete mean of selective risks.
- Macro-F1 over all classes present in gold or predictions; MCC multi-class (Gorodkin).
Records with status != ok are excluded by the caller (or counted as abstain, spec sec 13).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Scored:
    candidates: tuple[str, ...]
    probabilities: tuple[float, ...]
    gold: str

    @property
    def predicted(self) -> str:
        best = max(range(len(self.candidates)), key=lambda i: (self.probabilities[i], -i))
        return self.candidates[best]

    @property
    def confidence(self) -> float:
        return max(self.probabilities)

    @property
    def correct(self) -> bool:
        return self.predicted == self.gold

    def p_gold(self) -> float:
        return self.probabilities[self.candidates.index(self.gold)]


def accuracy(rows: Sequence[Scored]) -> float:
    return sum(r.correct for r in rows) / len(rows)


def macro_f1(rows: Sequence[Scored]) -> float:
    classes = sorted({r.gold for r in rows} | {r.predicted for r in rows})
    f1s = []
    for c in classes:
        tp = sum(1 for r in rows if r.predicted == c and r.gold == c)
        fp = sum(1 for r in rows if r.predicted == c and r.gold != c)
        fn = sum(1 for r in rows if r.predicted != c and r.gold == c)
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return sum(f1s) / len(f1s)


def mcc(rows: Sequence[Scored]) -> float:
    """Multi-class Matthews correlation (Gorodkin 2004)."""
    n = len(rows)
    correct = sum(r.correct for r in rows)
    t = Counter(r.gold for r in rows)
    p = Counter(r.predicted for r in rows)
    classes = set(t) | set(p)
    cov_tp = correct * n - sum(t[c] * p[c] for c in classes)
    cov_pp = n * n - sum(p[c] ** 2 for c in classes)
    cov_tt = n * n - sum(t[c] ** 2 for c in classes)
    if cov_pp == 0 or cov_tt == 0:
        return 0.0
    return cov_tp / math.sqrt(cov_pp * cov_tt)


def ece(rows: Sequence[Scored], n_bins: int = 15) -> float:
    bins: list[list[Scored]] = [[] for _ in range(n_bins)]
    for r in rows:
        idx = min(int(r.confidence * n_bins), n_bins - 1)
        bins[idx].append(r)
    total = len(rows)
    err = 0.0
    for b in bins:
        if b:
            conf = sum(r.confidence for r in b) / len(b)
            acc = sum(r.correct for r in b) / len(b)
            err += len(b) / total * abs(acc - conf)
    return err


def ece_bin_occupancy(rows: Sequence[Scored], n_bins: int = 15) -> list[int]:
    counts = [0] * n_bins
    for r in rows:
        counts[min(int(r.confidence * n_bins), n_bins - 1)] += 1
    return counts


def brier(rows: Sequence[Scored]) -> float:
    total = 0.0
    for r in rows:
        total += sum((p - (1.0 if c == r.gold else 0.0)) ** 2 for c, p in zip(r.candidates, r.probabilities))
    return total / len(rows)


def nll(rows: Sequence[Scored], eps: float = 1e-12) -> float:
    return -sum(math.log(max(r.p_gold(), eps)) for r in rows) / len(rows)


def aurc(rows: Sequence[Scored]) -> float:
    """Mean selective risk over coverages 1/n..n/n with examples sorted by decreasing confidence.
    Ties in confidence are handled by giving every member of a tied group the group's mean error,
    so the result does not depend on input order (Jev outputs sit on a 0.01 grid)."""
    ordered = sorted(rows, key=lambda r: -r.confidence)
    errors: list[float] = []
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j].confidence == ordered[i].confidence:
            j += 1
        group_err = sum(not r.correct for r in ordered[i:j]) / (j - i)
        errors.extend([group_err] * (j - i))
        i = j
    cum, risks = 0.0, []
    for k, e in enumerate(errors, 1):
        cum += e
        risks.append(cum / k)
    return sum(risks) / len(risks)


def accuracy_at_coverage(rows: Sequence[Scored], coverage: float) -> float:
    """Accuracy among the top `coverage` fraction of examples by confidence (audit item C2).
    k = ceil(coverage * n) (same rule as label_shift.coverage_threshold). The tied group that straddles the
    cut contributes its mean accuracy for the slots it fills, so the value does not depend on input order."""
    ordered = sorted(rows, key=lambda r: -r.confidence)
    k = max(1, math.ceil(coverage * len(ordered) - 1e-9))
    total, taken, i = 0.0, 0, 0
    while taken < k:
        j = i
        while j < len(ordered) and ordered[j].confidence == ordered[i].confidence:
            j += 1
        group_acc = sum(r.correct for r in ordered[i:j]) / (j - i)
        use = min(j - i, k - taken)
        total += group_acc * use
        taken += use
        i = j
    return total / k


def summary(rows: Sequence[Scored]) -> dict[str, float]:
    return {
        "n": len(rows), "accuracy": accuracy(rows), "macro_f1": macro_f1(rows), "mcc": mcc(rows),
        "ece": ece(rows), "brier": brier(rows), "nll": nll(rows), "aurc": aurc(rows),
        "acc_at_80": accuracy_at_coverage(rows, 0.8), "acc_at_50": accuracy_at_coverage(rows, 0.5),
    }
