"""Use case: train a task-specific classifier on a label budget and predict dev/test (Exp 3; full-data
baseline for Exp 2). Predictions are restricted to each example's frozen candidate set, exactly like
the zero-shot models (spec sec 8: softmax renormalised over candidates)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from s1bench.domain.budgets import stratified_subset
from s1bench.domain.candidates import CandidateSet
from s1bench.domain.examples import Example, LabelSet
from s1bench.domain.metrics import Scored, macro_f1
from s1bench.domain.predictions import STATUS_OK, Prediction, argmax_label, seeded_subset
from s1bench.domain.ports import CandidateReader, LabelRepository, PredictionStore, ProcessedReader


def restrict(probs_row: np.ndarray, cands: CandidateSet, col: dict[str, int]) -> tuple[float, ...]:
    vals = np.array([probs_row[col[c]] for c in cands.candidates], dtype=float)
    total = vals.sum()
    if total <= 0:  # every candidate unseen in training: uniform (documented)
        return tuple([1.0 / len(vals)] * len(vals))
    return tuple((vals / total).tolist())


def to_predictions(model_name: str, examples: Sequence[Example], probs: np.ndarray,
                   cands: dict[str, CandidateSet], labels: LabelSet) -> list[Prediction]:
    col = {lid: i for i, lid in enumerate(labels.ids())}
    out = []
    for e, row in zip(examples, probs):
        cs = cands[e.id]
        p = restrict(row, cs, col)
        out.append(Prediction(example_id=e.id, model=model_name, dataset=e.dataset, split=e.split,
                              candidates=cs.candidates, probabilities=p,
                              predicted=argmax_label(cs.candidates, p), gold=e.label, status=STATUS_OK))
    return out


TIMING_N, TIMING_WARMUP, TIMING_SEED = 200, 10, 20261006


def time_model(model, dev: Sequence[Example]) -> dict[str, object] | None:
    """Spec 9 (v1.1): per-example latency of the selected classifier on a seeded dev subset (texts only, batch
    size 1). The first TIMING_WARMUP calls are dropped. Dev texts only: never the test split."""
    if not hasattr(model, "time_single"):
        return None
    by_id = {e.id: e for e in dev}
    ids = seeded_subset(list(by_id), min(len(by_id), TIMING_N + TIMING_WARMUP), TIMING_SEED)
    ms = np.array(model.time_single([by_id[i].text for i in ids])[TIMING_WARMUP:], dtype=float)
    if len(ms) == 0:
        return None
    return {"split": "dev", "n": int(len(ms)), "warmup_dropped": TIMING_WARMUP, "subset_seed": TIMING_SEED,
            "p50_ms": round(float(np.percentile(ms, 50)), 3), "p95_ms": round(float(np.percentile(ms, 95)), 3),
            "mean_ms": round(float(ms.mean()), 3),
            **(model.timing_info() if hasattr(model, "timing_info") else {})}


class TrainAndPredict:
    def __init__(self, factory: Callable[[object, int], object], c_grid: Sequence[object],
                 examples: ProcessedReader, candidates: CandidateReader, labels: LabelRepository,
                 store: PredictionStore, reserve_seeds: Sequence[int] = (1006, 1007, 1008),
                 dev_share: float | None = None, save_full_probs: bool = False) -> None:
        # Revision protocol (v1.3), both off by default: dev_share -> at a label budget, hyperparameters, checkpoints
        # and the failed-run rule use a development part split off the budget itself (budgets.split_budget) instead of
        # the full development set; save_full_probs -> also write the probabilities over ALL classes for the test
        # split (full_probs/<model>/<dataset>_test.json), so that other candidate sets can be scored later.
        self._dev_share = dev_share
        self._save_full_probs = save_full_probs
        # spec 10.3: a failed run restarts with the NEXT reserve seed; at most len(reserve_seeds) restarts per
        # (dataset, budget) in total, shared by all seeds and grid values; each reserve seed is used once.
        self._reserve_used: dict[tuple[str, object], int] = {}
        # factory(hparam, seed) -> object with fit(texts, labels, dev_texts, dev_labels, label_ids),
        # predict_proba(texts, label_ids) and a `failed` flag (spec 10.3 failed-run rule)
        self._factory = factory
        self._reserve = list(reserve_seeds)
        self._grid = list(c_grid)
        self._examples = examples
        self._candidates = candidates
        self._labels = labels
        self._store = store

    def execute(self, model_name: str, dataset: str, budget: int | None, seed: int,
                predict_test: bool = True) -> dict[str, object]:
        labels = self._labels.load(dataset)
        ids = labels.ids()
        full_train = self._examples.read_split(dataset, "train")
        if budget is not None and budget >= len(full_train):
            raise ValueError(f"{dataset}: budget {budget} >= train size {len(full_train)}; use 'full' (spec 10.3)")
        train = stratified_subset(full_train, budget, seed)
        dev = self._examples.read_split(dataset, "dev")
        test = self._examples.read_split(dataset, "test") if predict_test else []
        dev_c = self._candidates.read_split(dataset, "dev")
        test_c = self._candidates.read_split(dataset, "test") if predict_test else {}
        sel_dev, sel_dev_c = dev, dev_c        # examples used for selection, early stopping and the failed-run rule
        if self._dev_share is not None and budget is not None:
            from s1bench.domain.budgets import split_budget

            train, sel_dev = split_budget(train, self._dev_share, seed)
            sel_dev_c = self._candidates.read_split(dataset, "train")

        def fit(hparam, s):
            return self._factory(hparam, s).fit([e.text for e in train], [e.label for e in train],
                                                [e.text for e in sel_dev], [e.label for e in sel_dev], ids,
                                                dev_candidates=[list(sel_dev_c[e.id].candidates) for e in sel_dev])

        def release(m) -> None:
            if hasattr(m, "release"):
                m.release()

        key = (dataset, budget)
        best = None
        restarts: list[dict[str, object]] = []
        for c in self._grid:
            model = fit(c, seed)
            while getattr(model, "failed", False) and self._reserve_used.get(key, 0) < len(self._reserve):
                reserve = self._reserve[self._reserve_used.get(key, 0)]
                self._reserve_used[key] = self._reserve_used.get(key, 0) + 1
                flog = getattr(model, "log", {}) or {}
                restarts.append({"hparam": c, "failed_seed": getattr(model, "seed", None), "restart_seed": reserve,
                                 "failed_run": {k: flog.get(k) for k in ("best_dev_accuracy", "best_dev_macro_f1",
                                                                          "majority_dev_accuracy", "majority_kind")}})
                release(model)
                model = fit(c, reserve)
            dev_pred = to_predictions(model_name, sel_dev, model.predict_proba([e.text for e in sel_dev], ids),
                                      sel_dev_c, labels)
            score = macro_f1([Scored(p.candidates, p.probabilities, p.gold) for p in dev_pred])
            # v1.1.1: a model that is not failed always beats a failed one (then dev macro-F1 decides)
            if best is None or (not getattr(model, "failed", False), score) > (not getattr(best[2], "failed", False), best[0]):
                if best is not None:
                    release(best[2])
                best = (score, c, model, dev_pred)
            else:
                release(model)
        dev_score, c, model, dev_pred = best
        if sel_dev is not dev:   # written dev predictions are always on the full development split
            dev_pred = to_predictions(model_name, dev, model.predict_proba([e.text for e in dev], ids), dev_c, labels)
        still_failed = bool(getattr(model, "failed", False))
        if still_failed:
            # spec 10.3: a seed that still fails enters the curve with MAJORITY-CLASS performance: a constant
            # predictor with the train class frequencies, restricted to each example's candidates.
            freq = np.array([sum(e.label == lid for e in train) for lid in ids], dtype=float)
            const = np.tile(freq / freq.sum(), (1, 1))
            dev_pred = to_predictions(model_name, dev, np.repeat(const, len(dev), axis=0), dev_c, labels)
            test_pred = (to_predictions(model_name, test, np.repeat(const, len(test), axis=0), test_c, labels)
                         if predict_test else [])
            dev_score = macro_f1([Scored(p.candidates, p.probabilities, p.gold) for p in dev_pred])
        else:
            test_probs = model.predict_proba([e.text for e in test], ids) if predict_test else None
            test_pred = to_predictions(model_name, test, test_probs, test_c, labels) if predict_test else []
            if predict_test and self._save_full_probs:
                self._store.write_json(f"full_probs/{model_name}/{dataset}_test.json", {
                    "label_ids": list(ids), "example_ids": [e.id for e in test],
                    "probabilities": [[round(float(x), 7) for x in row] for row in test_probs]})
        latency = time_model(model, dev)
        release(model)
        self._store.write(model_name, dataset, "dev", dev_pred)
        test_score = None
        if predict_test:   # pilot / dev checks: --skip-test never touches the test split
            self._store.write(model_name, dataset, "test", test_pred)
            test_score = macro_f1([Scored(p.candidates, p.probabilities, p.gold) for p in test_pred])
        return {"model": model_name, "dataset": dataset, "budget": budget or "full", "seed": seed,
                "n_train": len(train), "classes_in_train": len({e.label for e in train}),
                **({"selection_dev": "split off the budget (v1.3)", "n_dev_selection": len(sel_dev)}
                   if sel_dev is not dev else {}),
                "selected_c": c, "dev_macro_f1": round(dev_score, 4), "test_macro_f1": None if test_score is None else round(test_score, 4),
                "restarts": restarts, "still_failed": still_failed, "latency": latency,
                "scored_as": "majority_class" if still_failed else "model",
                "train_log": getattr(model, "log", {})}
