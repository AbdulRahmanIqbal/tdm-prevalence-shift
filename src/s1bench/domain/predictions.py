"""Prediction entity and pure helpers shared by every model adapter (spec sec 8)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

STATUS_OK = "ok"
STATUS_FAILED = "failed"    # transient/API error after retries (spec sec 13)
STATUS_BLOCKED = "blocked"  # refusal / safety block / content filter (spec sec 13)


@dataclass(frozen=True)
class Prediction:
    example_id: str
    model: str
    dataset: str
    split: str
    candidates: tuple[str, ...]           # label ids in presentation order
    probabilities: tuple[float, ...]      # aligned with candidates; empty if not ok
    predicted: str | None                 # label id (argmax); None if not ok
    gold: str
    status: str = STATUS_OK
    latency_ms: float | None = None
    input_tokens: int | None = None
    model_version: str | None = None
    error: str | None = None
    raw: Mapping[str, object] = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        return self.status == STATUS_OK and self.predicted == self.gold


def argmax_label(candidates: Sequence[str], probabilities: Sequence[float]) -> str:
    """Highest probability; ties broken by earliest candidate position (deterministic)."""
    best = max(range(len(candidates)), key=lambda i: (probabilities[i], -i))
    return candidates[best]


def seeded_subset(example_ids: Sequence[str], n: int, seed: int) -> list[str]:
    """Deterministic subset (e.g. pilot / smoke examples): first n ids ordered by sha256(seed:id)."""
    key = lambda eid: hashlib.sha256(f"{seed}:subset:{eid}".encode("utf-8")).digest()  # noqa: E731
    return sorted(example_ids, key=key)[:n]
