"""Candidate-set rules (spec sec 6). Pure: no I/O, no global randomness."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateSet:
    id: str
    candidates: tuple[str, ...]  # label ids in presentation order (letter A = index 0)
    gold_position: int


def example_rng(seed: int, example_id: str) -> random.Random:
    """RNG seeded from (seed, example id) via sha256: stable across runs and machines
    (Python's built-in hash() is salted per process and must not be used)."""
    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def nearest_labels(label_ids: Sequence[str], embeddings: Sequence[Sequence[float]]) -> dict[str, list[str]]:
    """For every label, all other labels ordered by decreasing cosine similarity (ties by id)."""
    neighbors = {}
    for i, lid in enumerate(label_ids):
        scored = [
            (-round(cosine(embeddings[i], embeddings[j]), 6), other)
            for j, other in enumerate(label_ids)
            if j != i
        ]
        neighbors[lid] = [other for _, other in sorted(scored)]
    return neighbors


def balanced_label_groups(label_ids: Sequence[str], embeddings: Sequence[Sequence[float]], target_size: int,
                          seed: int, restarts: int = 20, max_iter: int = 100) -> list[list[str]]:
    """Partition the labels ONCE into round(n / target_size) groups of near-equal size (sizes differ by at most
    1) by capacity-constrained k-means on cosine similarity (spec sec 6, v1.0.4). Every label of a group gets
    exactly the same candidate set (its group), so the option list carries no information about which member
    is gold. Deterministic: seeded k-means++ starts; the restart with the highest total within-group
    similarity wins; groups are sorted (members by id, groups by first member)."""
    import numpy as np

    x = np.asarray(embeddings, dtype=float)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n = len(label_ids)
    k = max(1, round(n / target_size))
    caps = [n // k + (1 if g < n % k else 0) for g in range(k)]
    order_ids = np.argsort(np.asarray(label_ids, dtype=object))          # deterministic tie-breaks by id
    rank = np.empty(n, dtype=int)
    rank[order_ids] = np.arange(n)
    best_assign, best_score = None, -np.inf
    for r in range(restarts):
        rng = np.random.default_rng([seed, r])
        centers = [x[rng.integers(n)]]                                     # k-means++ initialisation
        while len(centers) < k:
            d = 1 - np.max(np.stack([x @ c for c in centers]), axis=0)
            d = np.clip(d, 0, None) ** 2
            centers.append(x[rng.choice(n, p=d / d.sum())] if d.sum() > 0 else x[rng.integers(n)])
        centers = np.stack(centers)
        assign = None
        for _ in range(max_iter):
            sim = x @ centers.T                                            # (n, k)
            pairs = sorted(((-sim[i, g], rank[i], g, i) for i in range(n) for g in range(k)))
            new = np.full(n, -1)
            left = list(caps)
            for _, _, g, i in pairs:                                       # greedy capacity-constrained
                if new[i] < 0 and left[g] > 0:
                    new[i], left[g] = g, left[g] - 1
            if assign is not None and np.array_equal(new, assign):
                break
            assign = new
            centers = np.stack([x[assign == g].mean(axis=0) for g in range(k)])
            centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        score = float(sum((x[assign == g] @ x[assign == g].T).sum() for g in range(k)))
        if score > best_score + 1e-12:
            best_assign, best_score = assign, score
    groups = [sorted(label_ids[i] for i in range(n) if best_assign[i] == g) for g in range(k)]
    return sorted(groups, key=lambda grp: grp[0])


def balanced_gold_positions(example_ids: Sequence[str], k: int, seed: int) -> dict[str, int]:
    """Counterbalance the gold position within a split: every position 0..k-1 is used equally
    often (±1). Examples are ordered by sha256(seed:id), then position = rank mod k."""
    def key(eid: str) -> bytes:
        return hashlib.sha256(f"{seed}:position:{eid}".encode("utf-8")).digest()
    return {eid: rank % k for rank, eid in enumerate(sorted(example_ids, key=key))}


def build_candidate_set(
    example_id: str,
    gold: str,
    label_ids: Sequence[str],
    groups: Sequence[Sequence[str]] | None,
    max_candidates: int,
    seed: int,
    gold_position: int | None = None,
) -> CandidateSet:
    """<= max_candidates labels: all labels. Otherwise: the gold label's fixed group (balanced_label_groups).
    Order: seeded per-example shuffle; gold moved to its counterbalanced slot (the order of the others stays
    random). The set itself depends only on the gold's group, never on the example."""
    rng = example_rng(seed, example_id)
    if gold not in label_ids:
        raise ValueError(f"{example_id}: gold {gold!r} not in label set")

    if len(label_ids) <= max_candidates:
        chosen = list(label_ids)
    else:
        if groups is None:
            raise ValueError("label groups required when labels exceed max_candidates")
        matches = [g for g in groups if gold in g]
        if len(matches) != 1:
            raise ValueError(f"{example_id}: gold {gold!r} must be in exactly one group")
        chosen = sorted(matches[0])

    rng.shuffle(chosen)
    if gold_position is not None:
        chosen.remove(gold)
        chosen.insert(gold_position, gold)
    return CandidateSet(id=example_id, candidates=tuple(chosen), gold_position=chosen.index(gold))
