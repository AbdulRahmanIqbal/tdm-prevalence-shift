"""Use case: run one DecisionModel over a dataset split with the frozen candidate files (Exp 1 runner)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from s1bench.domain.predictions import STATUS_OK, Prediction, seeded_subset
from s1bench.domain.ports import CandidateReader, DecisionModel, LabelRepository, PredictionStore, ProcessedReader


@dataclass(frozen=True)
class RunSummary:
    model: str
    dataset: str
    split: str
    n: int
    n_ok: int
    accuracy_ok: float | None
    statuses: dict[str, int]
    model_versions: list[str]
    input_tokens: int
    mean_latency_ms: float | None


class RunPredictions:
    def __init__(self, model: DecisionModel, examples: ProcessedReader, candidates: CandidateReader,
                 labels: LabelRepository, store: PredictionStore,
                 progress: Callable[[int, int], None] | None = None, workers: int = 1) -> None:
        self._model = model
        self._examples = examples
        self._candidates = candidates
        self._labels = labels
        self._store = store
        self._progress = progress
        self._workers = workers

    def execute(self, dataset: str, split: str, limit: int | None, subset_seed: int) -> RunSummary:
        labels = self._labels.load(dataset)
        examples = {e.id: e for e in self._examples.read_split(dataset, split)}
        candidates = self._candidates.read_split(dataset, split)
        ids = list(examples)
        if limit is not None:
            ids = seeded_subset(ids, limit, subset_seed)

        def one(eid: str) -> Prediction:
            # Label-free view: the model never receives the gold label or gold position (no leakage path even
            # through a future adapter); gold is attached to the prediction afterwards.
            ex, cs = examples[eid], candidates[eid]
            blind = replace(ex, label="", label_index=-1)
            pred = self._model.predict(blind, replace(cs, gold_position=-1), labels)
            return replace(pred, gold=ex.label)

        if self._workers <= 1:
            predictions: list[Prediction] = []
            for i, eid in enumerate(ids, 1):
                predictions.append(one(eid))
                if self._progress:
                    self._progress(i, len(ids))
        else:  # API models only; results keep the input order
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                predictions = []
                for i, pred in enumerate(pool.map(one, ids), 1):
                    predictions.append(pred)
                    if self._progress:
                        self._progress(i, len(ids))
        self._store.write(self._model.name, dataset, split, predictions)

        ok = [p for p in predictions if p.status == STATUS_OK]
        statuses: dict[str, int] = {}
        for p in predictions:
            statuses[p.status] = statuses.get(p.status, 0) + 1
        latencies = [p.latency_ms for p in ok if p.latency_ms is not None]
        return RunSummary(
            model=self._model.name, dataset=dataset, split=split, n=len(predictions), n_ok=len(ok),
            accuracy_ok=(sum(p.correct for p in ok) / len(ok)) if ok else None,
            statuses=statuses,
            model_versions=sorted({p.model_version for p in ok if p.model_version}),
            input_tokens=sum(p.input_tokens or 0 for p in ok),
            mean_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
        )


def repeat_check(model: DecisionModel, examples: ProcessedReader, candidates: CandidateReader,
                 labels: LabelRepository, dataset: str, split: str, n: int, seed: int) -> dict[str, object]:
    """Spec sec 12: local decoder scoring must be deterministic. Score the same seeded subset twice (label-free
    view, as in RunPredictions) and compare the probability vectors exactly."""
    label_set = labels.load(dataset)
    exs = {e.id: e for e in examples.read_split(dataset, split)}
    cands = candidates.read_split(dataset, split)
    ids = seeded_subset(list(exs), n, seed)

    def score(eid: str) -> Prediction:
        return model.predict(replace(exs[eid], label="", label_index=-1), replace(cands[eid], gold_position=-1),
                             label_set)

    first = [score(eid) for eid in ids]
    second = [score(eid) for eid in ids]
    diff = [eid for eid, a, b in zip(ids, first, second) if a.status != b.status or a.probabilities != b.probabilities]
    max_abs = max((max((abs(x - y) for x, y in zip(a.probabilities, b.probabilities)), default=0.0)
                   for a, b in zip(first, second) if a.status == b.status == STATUS_OK), default=0.0)
    return {"model": model.name, "dataset": dataset, "split": split, "n": len(ids),
            "n_not_identical": len(diff), "max_abs_prob_difference": max_abs, "not_identical_ids": diff[:20],
            "statuses": sorted({p.status for p in first + second}), "deterministic": not diff}


def prefix_cache_timing(model, examples: ProcessedReader, candidates: CandidateReader, labels: LabelRepository,
                        dataset: str, split: str, n: int, seed: int, warmup: int = 10) -> dict[str, object]:
    """Spec 9 / audit C4 (v1.1): local-model latency with prefix caching on and off, on the seeded subset
    (label-free view, as in RunPredictions). Latency only; no predictions are written."""
    if not hasattr(model, "time_prefix_cache"):
        raise ValueError(f"{model.name}: prefix-cache timing is only defined for local decoder models")
    label_set = labels.load(dataset)
    exs = {e.id: e for e in examples.read_split(dataset, split)}
    cands = candidates.read_split(dataset, split)
    ids = seeded_subset(list(exs), n, seed)
    items = [(replace(exs[eid], label="", label_index=-1), replace(cands[eid], gold_position=-1), label_set)
             for eid in ids]
    return {"dataset": dataset, "split": split, "subset_seed": seed,
            **model.time_prefix_cache(items, warmup=warmup)}
