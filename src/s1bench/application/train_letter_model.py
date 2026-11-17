"""Use case: fine-tune a local decoder (QLoRA) on a label budget and predict dev/test with the candidate-letter
readout (Exp 4 condition b; spec 10.3, 10.4; v1.2).

Same budget subsets (stratified_subset), seeds, reserve-seed queue and failed-run fallback as the trained classifiers
(TrainAndPredict, spec 10.3). The failed-run rule is applied to the kept, deployed model (the one whose predictions
are reported): its candidate-restricted dev accuracy must be above that of the train-frequency predictor. Every
item carries its frozen candidate set from data/candidates/; models receive a label-free view.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from s1bench.domain.budgets import stratified_subset
from s1bench.domain.candidates import CandidateSet
from s1bench.domain.examples import Example, LabelSet
from s1bench.domain.metrics import Scored, accuracy, macro_f1
from s1bench.domain.ports import CandidateReader, LabelRepository, PredictionStore, ProcessedReader
from s1bench.domain.predictions import Prediction, argmax_label


def majority_predictions(name: str, items: Sequence[tuple[Example, CandidateSet]], train_labels: Sequence[str],
                         labels: LabelSet) -> list[Prediction]:
    """Spec 10.3 fallback for a seed that still fails: the train class frequencies, restricted to each candidate set
    (uniform if no candidate occurs in training), as for the trained classifiers."""
    freq = {lid: sum(lab == lid for lab in train_labels) for lid in labels.ids()}
    out = []
    for ex, cs in items:
        vals = np.array([freq[c] for c in cs.candidates], dtype=float)
        p = vals / vals.sum() if vals.sum() > 0 else np.full(len(vals), 1.0 / len(vals))
        probs = tuple(float(x) for x in p)
        out.append(Prediction(example_id=ex.id, model=name, dataset=ex.dataset, split=ex.split,
                              candidates=cs.candidates, probabilities=probs,
                              predicted=argmax_label(cs.candidates, probs), gold=ex.label))
    return out


def _rows(preds: Sequence[Prediction]) -> list[Scored]:
    return [Scored(p.candidates, p.probabilities, p.gold) for p in preds]


def restarts_used(summaries: Sequence[Mapping[str, object]], model_prefix: str) -> dict[tuple[str, int], int]:
    """Reserve seeds already consumed per (dataset, budget) by earlier calls of the same run (spec 10.3: the pool of
    reserve seeds is shared by all seeds of a (dataset, budget), across calls)."""
    used: dict[tuple[str, int], int] = {}
    for s in summaries:
        if str(s.get("model", "")).startswith(model_prefix) and isinstance(s.get("budget"), int):
            key = (str(s["dataset"]), int(s["budget"]))
            used[key] = used.get(key, 0) + len(s.get("restarts") or [])
    return used


class TrainLetterModel:
    def __init__(self, factory: Callable[[int, str], object], examples: ProcessedReader, candidates: CandidateReader,
                 labels: LabelRepository, store: PredictionStore, reserve_seeds: Sequence[int] = (1006, 1007, 1008),
                 reserve_used: Mapping[tuple[str, int], int] | None = None, force_fail_first: bool = False,
                 dev_share: float | None = None) -> None:
        # dev_share (revision protocol v1.3, off by default): selection, early stopping and the failed-run rule use a
        # development part split off the budget (budgets.split_budget); then no full-dev predictions are written
        # (only test), since the full development set plays no role in that protocol.
        self._dev_share = dev_share
        # force_fail_first: dev pilot only (checks the restart path on the GPU); never in a paper run
        # factory(seed, model_name) -> object with fit(train_items, dev_items, labels) (trains and deploys),
        # predict(items, labels), release(), and `seed`, `log` attributes (QloraLetterClassifier)
        self._factory = factory
        self._examples = examples
        self._candidates = candidates
        self._labels = labels
        self._store = store
        self._reserve = list(reserve_seeds)
        self._reserve_used: dict[tuple[str, int], int] = dict(reserve_used or {})
        self._force_fail_first = force_fail_first

    def _attempt(self, seed: int, name: str, train_items, dev_items, labels):
        t0 = time.perf_counter()
        model = self._factory(seed, name).fit(train_items, dev_items, labels)
        t1 = time.perf_counter()
        dev_pred = model.predict(dev_items, labels)
        model.log["seconds"] = {"fit": round(t1 - t0, 1), "predict_dev": round(time.perf_counter() - t1, 1)}
        rows = _rows(dev_pred)
        acc, f1 = accuracy(rows), macro_f1(rows)
        majority = float(model.majority_dev_accuracy)      # unrounded (the log keeps a rounded copy)
        model.failed = acc <= majority
        model.log.update({"dev_accuracy_deployed": round(acc, 4), "dev_macro_f1_deployed": round(f1, 4),
                          "failed": model.failed, "failed_rule": "deployed-model dev accuracy <= majority"})
        return model, dev_pred

    def execute(self, model_name: str, dataset: str, budget: int, seed: int,
                predict_test: bool = True) -> dict[str, object]:
        labels = self._labels.load(dataset)
        full_train = self._examples.read_split(dataset, "train")
        if budget >= len(full_train):
            raise ValueError(f"{dataset}: budget {budget} >= train size {len(full_train)}")
        train = stratified_subset(full_train, budget, seed)
        train_c = self._candidates.read_split(dataset, "train")
        dev = self._examples.read_split(dataset, "dev")
        dev_c = self._candidates.read_split(dataset, "dev")
        if self._dev_share is not None:
            from s1bench.domain.budgets import split_budget

            train, budget_dev = split_budget(train, self._dev_share, seed)
            dev_items = [(e, train_c[e.id]) for e in budget_dev]
        else:
            dev_items = [(e, dev_c[e.id]) for e in dev]
        train_items = [(e, train_c[e.id]) for e in train]
        test_items = []
        if predict_test:
            test_c = self._candidates.read_split(dataset, "test")
            test_items = [(e, test_c[e.id]) for e in self._examples.read_split(dataset, "test")]

        key = (dataset, budget)
        restarts: list[dict[str, object]] = []
        model, dev_pred = self._attempt(seed, model_name, train_items, dev_items, labels)
        if self._force_fail_first:
            model.failed = True
            model.log["forced_fail"] = True
        while model.failed and self._reserve_used.get(key, 0) < len(self._reserve):
            reserve = self._reserve[self._reserve_used.get(key, 0)]
            self._reserve_used[key] = self._reserve_used.get(key, 0) + 1
            restarts.append({"failed_seed": model.seed, "restart_seed": reserve,
                             "failed_run": {k: model.log.get(k) for k in (
                                 "dev_accuracy_deployed", "dev_macro_f1_deployed", "best_dev_accuracy_4bit",
                                 "majority_dev_accuracy", "majority_kind")}})
            model.release()
            model, dev_pred = self._attempt(reserve, model_name, train_items, dev_items, labels)

        still_failed = bool(model.failed)
        if still_failed:   # spec 10.3: the seed enters the curve with majority-class performance
            train_labels = [e.label for e in train]
            dev_pred = majority_predictions(model_name, dev_items, train_labels, labels)
            test_pred = majority_predictions(model_name, test_items, train_labels, labels)
        else:
            t0 = time.perf_counter()
            test_pred = model.predict(test_items, labels) if predict_test else []
            model.log["seconds"]["predict_test"] = round(time.perf_counter() - t0, 1)
        log = dict(model.log)
        trained_seed = model.seed
        model.release()
        if self._dev_share is None:      # budget-internal dev examples come from the train split: not written
            self._store.write(model_name, dataset, "dev", dev_pred)
        test_score = None
        if predict_test:
            self._store.write(model_name, dataset, "test", test_pred)
            test_score = macro_f1(_rows(test_pred))
        return {"model": model_name, "dataset": dataset, "budget": budget, "seed": seed, "trained_seed": trained_seed,
                "n_train": len(train), "classes_in_train": len({e.label for e in train}),
                **({"selection_dev": "split off the budget (v1.3)", "n_dev_selection": len(dev_items)}
                   if self._dev_share is not None else {}),
                "dev_macro_f1": round(macro_f1(_rows(dev_pred)), 4),
                "test_macro_f1": None if test_score is None else round(test_score, 4), "restarts": restarts,
                "still_failed": still_failed, "scored_as": "majority_class" if still_failed else "model",
                "train_log": log}
