"""Use case: Exp 2 prevalence shift on saved predictions (spec sec 10.2). No new inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from s1bench.domain import label_shift as ls
from s1bench.domain.examples import LabelSet
from s1bench.domain.predictions import STATUS_OK, Prediction

ARMS = ("none", "temperature", "em", "bcts_em", "oracle")


@dataclass(frozen=True)
class Matrices:
    probs: np.ndarray   # (n, K) zeros outside candidates
    mask: np.ndarray    # (n, K) True on candidates
    gold: np.ndarray    # (n,) class index
    positions: np.ndarray  # (n, K) candidate position of each class (large outside candidates)
    n_excluded: int     # non-ok predictions dropped (reported)
    ids: tuple[str, ...] = ()  # example ids of the rows (same order)


def to_matrices(predictions: Sequence[Prediction], labels: LabelSet) -> Matrices:
    index = {lid: i for i, lid in enumerate(labels.ids())}
    ok = [p for p in predictions if p.status == STATUS_OK and p.probabilities]
    k = len(index)
    probs = np.zeros((len(ok), k))
    mask = np.zeros((len(ok), k), dtype=bool)
    gold = np.zeros(len(ok), dtype=int)
    positions = np.full((len(ok), k), 10**6, dtype=np.int64)
    for r, p in enumerate(ok):
        cols = [index[c] for c in p.candidates]
        positions[r, cols] = np.arange(len(cols))
        vals = np.asarray(p.probabilities, dtype=float)
        probs[r, cols] = vals / vals.sum()
        mask[r, cols] = True
        gold[r] = index[p.gold]
    return Matrices(probs, mask, gold, positions, len(predictions) - len(ok), tuple(p.example_id for p in ok))


class PrevalenceShift:
    def __init__(self, alphas: Sequence[float], draws: int, seed: int, coverage: float,
                 em_max_iter: int = 100, em_tol: float = 1e-6) -> None:
        self._alphas = list(alphas)
        self._draws = draws
        self._seed = seed
        self._coverage = coverage
        self._em = (em_max_iter, em_tol)

    def run(self, dev: Matrices, test: Matrices, train_prior: np.ndarray, source_prior_kind: str,
            full_test: tuple[Sequence[str], np.ndarray] | None = None) -> tuple[list[dict[str, object]], dict[str, object]]:
        """train_prior = the DATASET's train class prior: centre of the Dirichlet for every model, so all
        models are evaluated on identical shifted test sets (same seed); it is also the EM source prior
        for trained classifiers. Zero-shot models use their implicit dev prior for EM instead."""
        k = dev.probs.shape[1]
        test_prior = np.bincount(test.gold, minlength=k) / len(test.gold)
        # Calibration fitted once on dev (no shift).
        t = ls.fit_temperature(dev.probs, dev.mask, dev.gold)
        bcts = ls.fit_bcts(dev.probs, dev.mask, dev.gold)
        dev_by_arm = {"none": dev.probs, "temperature": ls.apply_temperature(dev.probs, dev.mask, t),
                      "bcts": ls.apply_bcts(dev.probs, dev.mask, bcts)}

        def source_prior(calibrated_dev: np.ndarray) -> np.ndarray:
            if source_prior_kind == "train":
                return train_prior
            return ls.implicit_prior(calibrated_dev)  # implicit prior of a zero-shot model (smoothed)

        pi_none = source_prior(dev_by_arm["none"])
        pi_bcts = source_prior(dev_by_arm["bcts"])
        thresholds = {
            "none": ls.coverage_threshold(dev_by_arm["none"].max(axis=1), self._coverage),
            "temperature": ls.coverage_threshold(dev_by_arm["temperature"].max(axis=1), self._coverage),
            # every arm's threshold comes from that arm applied to (unshifted) dev
            "em": ls.coverage_threshold(ls.em_prior_adjust(dev.probs, dev.mask, pi_none, *self._em)[0].max(axis=1),
                                        self._coverage),
            "bcts_em": ls.coverage_threshold(
                ls.em_prior_adjust(dev_by_arm["bcts"], dev.mask, pi_bcts, *self._em)[0].max(axis=1), self._coverage),
            "oracle": ls.coverage_threshold(ls.prior_adjust(
                dev.probs, dev.mask, pi_none, np.bincount(dev.gold, minlength=k) / len(dev.gold)).max(axis=1),
                self._coverage),
        }

        # every arm applied to unshifted dev (thresholds above; realized dev coverage recorded for all arms)
        dev_arm_probs = {
            "none": dev_by_arm["none"], "temperature": dev_by_arm["temperature"],
            "em": ls.em_prior_adjust(dev.probs, dev.mask, pi_none, *self._em)[0],
            "bcts_em": ls.em_prior_adjust(dev_by_arm["bcts"], dev.mask, pi_bcts, *self._em)[0],
            "oracle": ls.prior_adjust(dev.probs, dev.mask, pi_none, np.bincount(dev.gold, minlength=k) / len(dev.gold)),
        }

        def arms(p: np.ndarray, m: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
            em_p, _, _ = ls.em_prior_adjust(p, m, pi_none, *self._em)
            bc = ls.apply_bcts(p, m, bcts)
            bcem_p, _, _ = ls.em_prior_adjust(bc, m, pi_bcts, *self._em)
            return {"none": p, "temperature": ls.apply_temperature(p, m, t), "em": em_p,
                    "bcts_em": bcem_p, "oracle": ls.prior_adjust(p, m, pi_none, target)}

        # No-shift reference: original test as-is (alpha = inf).
        base = arms(test.probs, test.mask, test_prior)
        noshift_risk = {a: ls.selective_risk(base[a], test.gold, thresholds[a], test.positions)[0] for a in ARMS}

        rows: list[dict[str, object]] = []

        def record(alpha: str, draw: int, target: np.ndarray, idx: np.ndarray | None, sampling: str) -> None:
            sel = slice(None) if idx is None else idx
            p, mk, g, pos = test.probs[sel], test.mask[sel], test.gold[sel], test.positions[sel]
            for arm, probs in arms(p, mk, target).items():
                mm = ls.matrix_metrics(probs, g, positions=pos)
                risk, cov = ls.selective_risk(probs, g, thresholds[arm], pos)
                rows.append({
                    "alpha": alpha, "draw": draw, "sampling": sampling, "arm": arm, "n": int(len(g)),
                    "n_classes_present": int(len(np.unique(g))),
                    "unique_frac": 1.0 if idx is None else round(len(np.unique(idx)) / len(idx), 4),
                    "kl_to_test_prior": round(ls.kl(target, test_prior), 6),
                    **{f"{key}": round(val, 6) for key, val in asdict(mm).items()},
                    "threshold": round(thresholds[arm], 6), "risk_shifted": risk, "coverage_shifted": cov,
                    "risk_noshift_test": noshift_risk[arm],
                    "excess_risk": None if risk != risk else round(risk - noshift_risk[arm], 6),
                })

        record("inf", 0, test_prior, None, "original")
        # Common random numbers across models: shifted samples are drawn on the FULL processed test set
        # (independent of which rows a model lost), then mapped to this model's rows; rows the model lost
        # (failed/blocked) are dropped from the sample and counted.
        if full_test is None:
            full_ids, full_gold = test.ids or tuple(str(i) for i in range(len(test.gold))), test.gold
        else:
            full_ids, full_gold = full_test
        row_of = {eid: r for r, eid in enumerate(test.ids)} if test.ids else {str(i): i for i in range(len(test.gold))}
        to_row = np.array([row_of.get(eid, -1) for eid in full_ids])
        lost = {"with_replacement": [], "without_replacement": []}

        def mapped(idx_full: np.ndarray, sampling: str) -> np.ndarray:
            rows_ = to_row[idx_full]
            lost[sampling].append(float(np.mean(rows_ < 0)) if len(rows_) else 0.0)
            return rows_[rows_ >= 0]

        rng = np.random.default_rng(self._seed)
        for alpha in self._alphas:
            for draw in range(1, self._draws + 1):
                target = ls.dirichlet_target_prior(train_prior, alpha, rng)
                record(str(alpha), draw, target, mapped(
                    ls.resample_with_replacement(full_gold, target, len(full_gold), rng), "with_replacement"),
                    "with_replacement")
                record(str(alpha), draw, target, mapped(
                    ls.resample_without_replacement(full_gold, target, rng), "without_replacement"),
                    "without_replacement")
        # v1.1.1 (audit 2026-10-08): no-shift RESAMPLED control. Top-label ECE is biased upward on resampled
        # (duplicated) data, so shifted draws are compared with draws from the same with-replacement resampler at
        # the empirical test prior (q = test prior), not only with the original test set ("inf"). Drawn after the
        # shifted draws, so those keep their random numbers; same CRN mapping across models.
        full_prior = np.bincount(full_gold, minlength=k) / len(full_gold)
        for draw in range(1, self._draws + 1):
            record("noshift_resampled", draw, full_prior, mapped(
                ls.resample_with_replacement(full_gold, full_prior, len(full_gold), rng), "with_replacement"),
                "with_replacement")

        meta = {"temperature": t, "bcts_temperature": bcts.temperature, "source_prior_kind": source_prior_kind,
                "n_dev": int(len(dev.gold)), "n_test": int(len(test.gold)),
                "excluded_dev": dev.n_excluded, "excluded_test": test.n_excluded,
                "thresholds": thresholds, "coverage": self._coverage,
                "dev_coverage_at_threshold": {a: float(np.mean(dev_arm.max(axis=1) >= thresholds[a]))
                                              for a, dev_arm in dev_arm_probs.items()},
                "eps": ls.EPS, "implicit_prior_smoothing": ls.IMPLICIT_PRIOR_SMOOTHING,
                "mean_share_of_sample_lost": {k: float(np.mean(v)) if v else 0.0 for k, v in lost.items()}}
        return rows, meta
