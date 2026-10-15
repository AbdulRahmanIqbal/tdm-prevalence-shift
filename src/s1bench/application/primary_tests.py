"""Primary tests (spec sec 11.1): P1/P2 (decision training pairs) and PA1-PA3 (Pillar A), plus Holm.

P1/P2 (spec 11.2): paired bootstrap, B = 10,000, same test indices for both models; for each dataset two
tests, Macro-F1 and top-label ECE (15 bins); delta = metric(TDM) - metric(base). Macro-F1 > 0 favours the
TDM, ECE < 0 favours the TDM.

PA1-PA3: nested paired bootstrap (spec 11.2).

v1.0.3 design (after the first dev run showed the single-draw design is dominated by prior-draw
variance): the estimand averages over a FIXED set of target priors; each bootstrap replicate resamples
the test set and rebuilds all shifted samples from it. Both models always see identical samples.

PA1  d_r = [ECE_A(S_r) - ECE_A(N_r)] - [ECE_B(S_r) - ECE_B(N_r)]           (two-sided, RQ-A1)
     v1.1.1 (audit 2026-10-08): N_r = no-shift control built with the SAME with-replacement resampler at
     q = the base sample's own class prior (M draws, like S_r). Top-label ECE is biased upward on resampled
     (duplicated) data, so comparing S_r with the base sample T_r itself mixed resampling bias into the shift
     effect (~50% of the apparent ECE increase on CLINC150 at alpha = 1 in a dev check).
PA2  d_r = excess_A - excess_B, excess = risk(S_r; tau_dev) - risk(T_r; tau_dev), tau at 80% dev coverage
     (risk is a mean of per-example losses, unbiased under resampling: base sample T_r kept, spec 11.2)
PA3  d_r = ECE_A^EM(S_r) - ECE_A(S_r)   (EM with model A's source prior; H-A2 expects d < 0)
p = 2 * min(P(d <= 0), P(d >= 0)) (a CI-inversion p; smallest non-zero value 2/B); estimate = mean of d_r;
95% percentile CI. v1.1.1: B = 10,000 (exp2.yaml `pa.b`), so 2/B = 0.0002 is below Holm's first cutoff 0.05/35.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from s1bench.application.prevalence_shift import Matrices, to_matrices
from s1bench.domain import label_shift as ls
from s1bench.domain.examples import LabelSet
from s1bench.domain.predictions import STATUS_OK, Prediction
from s1bench.domain.stats import holm, paired_bootstrap

PRIMARY_FAMILY_SIZE = 35  # P1 10 + P2 10 + PA1-PA3 15 (spec 11.1)
FAILED_LIMIT = 0.01       # spec sec 13: failed > 1% on a dataset -> rerun, never analyse


def expected_family_keys(datasets: Sequence[str], pair_tests_: Sequence[str] = ("P1", "P2")) -> set[str]:
    """The preregistered 35 keys "<test>:<dataset>" (spec 11.1)."""
    tests = [f"{t}-{m}" for t in pair_tests_ for m in ("F1", "ECE")] + ["PA1", "PA2", "PA3"]
    return {f"{t}:{d}" for t in tests for d in datasets}


def input_check(predictions: Sequence[Prediction], expected_ids: Sequence[str]) -> dict[str, object]:
    """Coverage and status counts of one model's prediction file before a primary test (spec sec 13).
    problems != [] means the file must not enter a primary test (incomplete run, failed > 1%, unknown status).
    Blocked rows are allowed here; they are reported and excluded from the paired probability-based tests
    (spec 11.2/13: blocked = wrong + abstain applies to Accuracy/Macro-F1/AURC tables of Exp 1)."""
    from s1bench.domain.predictions import STATUS_BLOCKED, STATUS_FAILED

    ids = [p.example_id for p in predictions]
    status: dict[str, int] = {}
    for p in predictions:
        status[p.status] = status.get(p.status, 0) + 1
    expected = set(expected_ids)
    n = max(len(expected), 1)
    problems = []
    missing = expected - set(ids)
    if missing:
        problems.append(f"{len(missing)}/{len(expected)} examples missing (incomplete or subset run)")
    if set(ids) - expected:
        problems.append(f"{len(set(ids) - expected)} ids not in the processed split")
    if len(set(ids)) != len(ids):
        problems.append("duplicate example ids")
    unknown = set(status) - {STATUS_OK, STATUS_FAILED, STATUS_BLOCKED}
    if unknown:
        problems.append(f"unknown statuses {sorted(unknown)}")
    if status.get(STATUS_FAILED, 0) / n > FAILED_LIMIT:
        problems.append(f"failed {status[STATUS_FAILED]}/{n} > 1%: rerun the dataset (spec sec 13)")
    return {"n": len(ids), "status": status, "problems": problems}


def aligned_matrices(a: Sequence[Prediction], b: Sequence[Prediction], labels: LabelSet) -> tuple[Matrices, Matrices, int]:
    """Keep examples that are ok for BOTH models, in the same order (paired tests need this)."""
    ok_a = {p.example_id: p for p in a if p.status == STATUS_OK and p.probabilities}
    ok_b = {p.example_id: p for p in b if p.status == STATUS_OK and p.probabilities}
    ids = sorted(set(ok_a) & set(ok_b))
    dropped = len(set(ok_a) | set(ok_b)) - len(ids)
    return to_matrices([ok_a[i] for i in ids], labels), to_matrices([ok_b[i] for i in ids], labels), dropped


@dataclass(frozen=True)
class TestResult:
    test: str
    dataset: str
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    b: int


def _ece(m: Matrices, idx: np.ndarray, probs: np.ndarray | None = None) -> float:
    p = m.probs[idx] if probs is None else probs
    return ls.matrix_metrics(p, m.gold[idx], positions=m.positions[idx]).ece


def _risk(m: Matrices, idx: np.ndarray, tau: float) -> float:
    return ls.selective_risk(m.probs[idx], m.gold[idx], tau, m.positions[idx])[0]


def _summ(name: str, dataset: str, d: np.ndarray) -> TestResult:
    d = d[~np.isnan(d)]
    lo, hi = np.quantile(d, [0.025, 0.975])
    p = min(1.0, 2 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0))))   # spec 11.2 formula
    return TestResult(name, dataset, float(np.mean(d)), float(lo), float(hi), p, len(d))


def pa_tests(dataset: str, a_dev: Matrices, a_test: Matrices, b_dev: Matrices, b_test: Matrices,
             train_prior: np.ndarray, a_source_prior_kind: str, alpha: float = 1.0, coverage: float = 0.8,
             b: int = 10_000, seed: int = 20261006, n_priors: int = 50) -> list[TestResult]:
    """Estimand: the average over a FIXED set of `n_priors` target priors q_1..q_M ~ Dirichlet(alpha*K*pi)
    (the shift distribution is part of the design, not sampling error). Uncertainty comes from the finite
    test set: each bootstrap replicate resamples test examples (base sample), then builds every shifted
    sample from that base sample with the same q_m (common random numbers across models)."""
    if not np.array_equal(a_test.gold, b_test.gold):
        raise ValueError("test matrices must be aligned (same examples, same order)")
    n = len(a_test.gold)
    rng = np.random.default_rng(seed)
    priors = [ls.dirichlet_target_prior(train_prior, alpha, rng) for _ in range(n_priors)]
    tau_a = ls.coverage_threshold(a_dev.probs.max(axis=1), coverage)
    tau_b = ls.coverage_threshold(b_dev.probs.max(axis=1), coverage)
    pi_a = train_prior if a_source_prior_kind == "train" else ls.implicit_prior(a_dev.probs)

    pa1, pa2, pa3 = np.empty(b), np.empty(b), np.empty(b)
    for r in range(b):
        base = ls.stratified_bootstrap(a_test.gold, rng)    # class-stratified bootstrap of the test set
        risk_a0, risk_b0 = _risk(a_test, base, tau_a), _risk(b_test, base, tau_b)
        q0 = np.bincount(a_test.gold[base], minlength=len(train_prior)) / len(base)   # base's own prior
        d1, d2, d3 = [], [], []
        for q in priors:
            s_idx = base[ls.resample_with_replacement(a_test.gold[base], q, n, rng)]
            n_idx = base[ls.resample_with_replacement(a_test.gold[base], q0, n, rng)]   # no-shift control
            ea, eb = _ece(a_test, s_idx), _ece(b_test, s_idx)
            d1.append((ea - _ece(a_test, n_idx)) - (eb - _ece(b_test, n_idx)))
            d2.append((_risk(a_test, s_idx, tau_a) - risk_a0) - (_risk(b_test, s_idx, tau_b) - risk_b0))
            em_probs, _, _ = ls.em_prior_adjust(a_test.probs[s_idx], a_test.mask[s_idx], pi_a)
            d3.append(_ece(a_test, s_idx, em_probs) - ea)
        pa1[r], pa2[r], pa3[r] = np.nanmean(d1), np.nanmean(d2), np.nanmean(d3)
    return [_summ("PA1", dataset, pa1), _summ("PA2", dataset, pa2), _summ("PA3", dataset, pa3)]


def fast_macro_f1(pred: np.ndarray, gold: np.ndarray, k: int) -> float:
    """Macro-F1 over classes present in gold or predictions (same definition as label_shift.matrix_metrics)."""
    tp = np.bincount(gold[pred == gold], minlength=k)
    support = np.bincount(gold, minlength=k) + np.bincount(pred, minlength=k)  # = 2tp + fp + fn
    present = support > 0
    return float(np.mean(2 * tp[present] / support[present]))


def fast_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE, equal-width bins (same definition as label_shift.matrix_metrics)."""
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    gap = np.bincount(bins, weights=correct, minlength=n_bins) - np.bincount(bins, weights=conf, minlength=n_bins)
    return float(np.abs(gap).sum() / len(conf))


def pair_tests(test_name: str, dataset: str, a_test: Matrices, b_test: Matrices, b: int = 10_000,
               seed: int = 20261006) -> list[TestResult]:
    """P1/P2 on one dataset: A = decision-trained model, B = its base. Returns [Macro-F1, ECE]."""
    if not np.array_equal(a_test.gold, b_test.gold):
        raise ValueError("test matrices must be aligned (same examples, same order)")
    gold, k = a_test.gold, a_test.probs.shape[1]
    pa = ls.argmax_by_position(a_test.probs, a_test.positions)
    pb = ls.argmax_by_position(b_test.probs, b_test.positions)
    ca, cb = a_test.probs.max(axis=1), b_test.probs.max(axis=1)
    ka, kb = (pa == gold).astype(float), (pb == gold).astype(float)
    n = len(gold)
    f1 = paired_bootstrap(lambda i: fast_macro_f1(pa[i], gold[i], k), lambda i: fast_macro_f1(pb[i], gold[i], k),
                          n, b=b, seed=seed)
    ece = paired_bootstrap(lambda i: fast_ece(ca[i], ka[i]), lambda i: fast_ece(cb[i], kb[i]), n, b=b, seed=seed)
    return [TestResult(f"{test_name}-F1", dataset, f1.observed, f1.ci_low, f1.ci_high, f1.p_value, f1.b),
            TestResult(f"{test_name}-ECE", dataset, ece.observed, ece.ci_low, ece.ci_high, ece.p_value, ece.b)]


def holm_family(rows: Sequence[dict[str, object]], expected_keys: set[str]) -> tuple[list[dict[str, object]], dict]:
    """Holm over the WHOLE preregistered family, from saved per-file results (keys test, dataset, p_value,
    and the file's dev_check flag). m = size of the preregistered family even if tests are missing
    (conservative). Complete only if the keys are EXACTLY the expected set and no row is a dev check.
    Duplicate or non-preregistered keys are errors."""
    keys = [f"{r['test']}:{r['dataset']}" for r in rows]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"duplicate primary results: {dupes}")
    extra = sorted(set(keys) - expected_keys)
    if extra:
        raise ValueError(f"results outside the preregistered family: {extra}")
    adjusted = holm({k: float(r["p_value"]) for k, r in zip(keys, rows)}, m=len(expected_keys))
    dev = sorted({str(r.get("source_file")) for r in rows if r.get("dev_check")})
    missing = sorted(expected_keys - set(keys))
    complete = not missing and not dev
    status = {"complete": complete, "n_present": len(keys), "family_size": len(expected_keys),
              "missing": missing, "dev_check_files": dev}
    return [{**r, "p_holm": adjusted[k], "holm_family_complete": complete} for k, r in zip(keys, rows)], status


def with_holm(results: Sequence[TestResult], family_size: int = 35) -> list[dict[str, object]]:
    """Holm with m = the preregistered family size (35) even for a partial set (conservative: missing
    tests count as p = 1). Per-file convenience only; the final numbers come from holm_family."""
    adjusted = holm({f"{r.test}:{r.dataset}": r.p_value for r in results}, m=family_size)
    complete = len(results) == family_size
    return [{**r.__dict__, "p_holm": adjusted[f"{r.test}:{r.dataset}"], "holm_family_complete": complete}
            for r in results]
