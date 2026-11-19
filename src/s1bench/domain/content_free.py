"""Revision protocol (v1.3; exploratory, added after the results): content-free calibration (Zhao et al. 2021,
"Calibrate before use"). Pure: no I/O, no global randomness.

A content-free probe is the TDM prompt of a dataset with the example text replaced by a content-free string
("N/A", "", "[MASK]"; Zhao et al.). The probe's probabilities over the candidate options estimate the system's
bias with no evidence; calibration divides every prediction by it, q(y|x) proportional to p(y|x) / w(y), with no
labels used (W = diag(w)^-1, b = 0 in Zhao et al.'s notation).

Two estimates of w, both from the same probes:
- label level (all datasets): every label of a candidate group is shown once at every position (the cyclic rotations
  of one seeded order of the group); w(label) = mean probability of that label over the rotations and the
  content-free texts. Position bias averages out, as it does in the test sets (gold positions are counterbalanced).
- ordering matched (datasets whose test split has few distinct option orders: FPB, HateXplain, TREC-6): one probe
  per distinct test order; w(position) = mean probability over the content-free texts for exactly that order. This is
  Zhao et al.'s per-prompt estimate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from s1bench.domain.candidates import example_rng

CONTENT_FREE_TEXTS = ("N/A", "", "[MASK]")
MAX_MATCHED_ORDERINGS = 400     # ordering-matched probes only when the test split has at most this many orders
W_FLOOR = 1e-6                  # probabilities below this in w are raised to it before dividing


def rotation_orderings(group: Sequence[str], seed: int, tag: str) -> list[tuple[str, ...]]:
    """The k cyclic rotations of one seeded order of the group: every label appears once at every position."""
    base = sorted(group)
    example_rng(seed, f"contentfree:{tag}").shuffle(base)
    return [tuple(base[r:] + base[:r]) for r in range(len(base))]


def probe_orderings(groups: Sequence[Sequence[str]], test_orderings: Sequence[Sequence[str]], seed: int,
                    dataset: str) -> list[dict[str, object]]:
    """Unique option orders to probe, each tagged with the designs that use it ('rotation', 'matched').
    Rotations first (group by group), then the distinct test orders (sorted) if there are at most
    MAX_MATCHED_ORDERINGS of them; an order needed by both designs is probed once."""
    rows: dict[tuple[str, ...], dict[str, object]] = {}
    for g, group in enumerate(groups):
        for order in rotation_orderings(group, seed, f"{dataset}:{g}"):
            rows.setdefault(order, {"candidates": list(order), "group": g, "designs": []})["designs"].append("rotation")
    distinct = sorted({tuple(o) for o in test_orderings})
    if len(distinct) <= MAX_MATCHED_ORDERINGS:
        group_of = {lid: g for g, members in enumerate(groups) for lid in members}
        for order in distinct:
            row = rows.setdefault(order, {"candidates": list(order), "group": group_of[order[0]], "designs": []})
            row["designs"].append("matched")
    return list(rows.values())


def label_level_weights(probes: Sequence[tuple[Sequence[str], Sequence[float]]],
                        label_ids: Sequence[str]) -> np.ndarray:
    """w over the dataset's labels (index = position in label_ids) from rotation probes: (candidates, probabilities)
    pairs. Mean probability of each label over the probes that offer it; labels never offered stay NaN."""
    index = {lid: i for i, lid in enumerate(label_ids)}
    total, count = np.zeros(len(label_ids)), np.zeros(len(label_ids))
    for cands, probs in probes:
        p = np.asarray(probs, dtype=float)
        p = p / p.sum()
        for c, v in zip(cands, p):
            total[index[c]] += v
            count[index[c]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def matched_weights(probes: Sequence[tuple[Sequence[str], Sequence[float]]]) -> dict[tuple[str, ...], np.ndarray]:
    """Per option order: mean (renormalised) probability vector by position over its probes."""
    acc: dict[tuple[str, ...], list[np.ndarray]] = {}
    for cands, probs in probes:
        p = np.asarray(probs, dtype=float)
        acc.setdefault(tuple(cands), []).append(p / p.sum())
    return {k: np.mean(v, axis=0) for k, v in acc.items()}


def weights_matrix_label(w: np.ndarray, n: int) -> np.ndarray:
    """Label-level w broadcast to (n, L) for content_free_adjust."""
    return np.tile(w, (n, 1))


def weights_matrix_matched(candidates: Sequence[Sequence[str]], weights: Mapping[tuple[str, ...], np.ndarray],
                           label_ids: Sequence[str]) -> np.ndarray:
    """(n, L) w for examples with the given option orders (every order must have a matched probe)."""
    index = {lid: i for i, lid in enumerate(label_ids)}
    out = np.full((len(candidates), len(label_ids)), np.nan)
    for r, cands in enumerate(candidates):
        vec = weights[tuple(cands)]
        out[r, [index[c] for c in cands]] = vec
    return out


def content_free_adjust(probs: np.ndarray, mask: np.ndarray, w: np.ndarray) -> np.ndarray:
    """q proportional to p / w on each example's candidates (label-indexed (n, L) arrays); zero elsewhere."""
    if np.isnan(np.where(mask, w, 0.0)).any():
        raise ValueError("content-free weight missing for an offered label")
    scores = np.where(mask, probs / np.maximum(np.where(mask, w, 1.0), W_FLOOR), 0.0)
    return scores / scores.sum(axis=1, keepdims=True)
