"""Prevalence-shift math for Exp 2 (spec sec 10.2). numpy only; no I/O.

Shapes: every example has its own candidate set, so probabilities are stored as a dense
(n_examples, n_classes) matrix P with zeros outside the example's candidates, plus a boolean
mask M of the same shape (True on the candidates). Gold labels are class indices.

Literature: Dirichlet shift and with-replacement resampling (Lipton et al. 2018; Alexandari et al.
2020; Garg et al. 2020), EM prior adjustment (Saerens et al. 2002), temperature scaling
(Guo et al. 2017), BCTS (Alexandari et al. 2020), selective risk at a fixed coverage
(Geifman & El-Yaniv 2017).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6  # probability floor before taking logs (Jev returns exact zeros on a 0.01 grid)


# ---------------------------------------------------------------- target priors and resampling
def dirichlet_target_prior(source_prior: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Dirichlet(alpha * K * pi_s): centred on the source prior; equals Dirichlet(alpha * 1) for uniform pi_s."""
    k = len(source_prior)
    conc = np.maximum(alpha * k * source_prior, 1e-8)
    return rng.dirichlet(conc)


def resample_with_replacement(gold: np.ndarray, target_prior: np.ndarray, n: int,
                              rng: np.random.Generator) -> np.ndarray:
    """Class-stratified sample of size n: class counts ~ Multinomial(n, q), examples drawn with replacement."""
    counts = rng.multinomial(n, target_prior)
    idx = []
    for c, m in enumerate(counts):
        if m == 0:
            continue
        pool = np.flatnonzero(gold == c)
        if len(pool) == 0:
            raise ValueError(f"class {c} has target mass but no test examples")
        idx.append(rng.choice(pool, size=m, replace=True))
    return np.concatenate(idx) if idx else np.array([], dtype=int)


def stratified_bootstrap(gold: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap that keeps every class's test count n_c (resample with replacement within each class).
    Used as the PA base sample: an unstratified base can lose a small class (HWU64 has classes with 3 test
    examples: P(lost) ~ 5% per replicate), after which class-stratified shift resampling is undefined."""
    idx = [rng.choice(np.flatnonzero(gold == c), size=int(m), replace=True)
           for c, m in enumerate(np.bincount(gold)) if m > 0]
    return np.concatenate(idx)


def resample_without_replacement(gold: np.ndarray, target_prior: np.ndarray,
                                 rng: np.random.Generator) -> np.ndarray:
    """Sensitivity variant: largest n such that no class needs more examples than it has."""
    avail = np.bincount(gold, minlength=len(target_prior))
    positive = target_prior > 0
    n = int(min(len(gold), np.min(np.floor(avail[positive] / target_prior[positive]))))
    counts = np.floor(n * target_prior).astype(int)
    idx = [rng.choice(np.flatnonzero(gold == c), size=m, replace=False) for c, m in enumerate(counts) if m > 0]
    return np.concatenate(idx) if idx else np.array([], dtype=int)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12))))


# ---------------------------------------------------------------- recalibration
def _renorm(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scores = np.where(mask, scores, 0.0)
    return scores / scores.sum(axis=1, keepdims=True)


def _logits(probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, np.log(np.maximum(probs, EPS)), -np.inf)


def _softmax_masked(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask, z, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    e = np.where(mask, np.exp(z), 0.0)
    return e / e.sum(axis=1, keepdims=True)


def _nll(probs: np.ndarray, gold: np.ndarray) -> float:
    return float(-np.mean(np.log(np.maximum(probs[np.arange(len(gold)), gold], 1e-12))))


def fit_temperature(probs: np.ndarray, mask: np.ndarray, gold: np.ndarray) -> float:
    """Minimise dev NLL over T on a log grid, then refine (robust, no gradients needed)."""
    z = _logits(probs, mask)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 121))
    losses = [_nll(_softmax_masked(z / t, mask), gold) for t in grid]
    best = int(np.argmin(losses))
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    fine = np.exp(np.linspace(np.log(lo), np.log(hi), 41))
    return float(fine[int(np.argmin([_nll(_softmax_masked(z / t, mask), gold) for t in fine]))])


def apply_temperature(probs: np.ndarray, mask: np.ndarray, t: float) -> np.ndarray:
    return _softmax_masked(_logits(probs, mask) / t, mask)


@dataclass(frozen=True)
class BctsParams:
    temperature: float
    biases: np.ndarray  # one per class


def fit_bcts(probs: np.ndarray, mask: np.ndarray, gold: np.ndarray, iters: int = 500, lr: float = 0.05) -> BctsParams:
    """Bias-corrected temperature scaling (Alexandari et al. 2020): z/T + b_k, NLL on dev, Adam."""
    z = np.where(mask, np.log(np.maximum(probs, EPS)), 0.0)
    k = probs.shape[1]
    params = np.zeros(k + 1)  # [log T, b_1..b_K]
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(gold)), gold] = 1.0
    for step in range(1, iters + 1):
        t = np.exp(params[0])
        scores = z / t + params[1:]
        p = _softmax_masked(scores, mask)
        diff = (p - onehot) / len(gold)  # dNLL/dscores
        grad_b = diff.sum(axis=0)
        grad_logt = float(np.sum(diff * (-z / t)))
        grad = np.concatenate([[grad_logt], grad_b])
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        params -= lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    return BctsParams(float(np.exp(params[0])), params[1:].copy())


def apply_bcts(probs: np.ndarray, mask: np.ndarray, params: BctsParams) -> np.ndarray:
    z = np.where(mask, np.log(np.maximum(probs, EPS)), 0.0)
    return _softmax_masked(z / params.temperature + params.biases, mask)


def em_prior_adjust(probs: np.ndarray, mask: np.ndarray, source_prior: np.ndarray,
                    max_iter: int = 100, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, int]:
    """Saerens et al. (2002) EM, candidate-restricted (spec 10.2):
    E: p~(y|x) ∝ p(y|x) q(y)/pi_s(y) over the example's candidates; M: q = mean of p~ over examples."""
    pi_s = np.maximum(source_prior, 1e-12)
    q = pi_s.copy()
    adjusted = probs
    for it in range(1, max_iter + 1):
        adjusted = _renorm(probs * (q / pi_s), mask)
        q_new = adjusted.mean(axis=0)
        if np.max(np.abs(q_new - q)) < tol:
            q = q_new
            break
        q = q_new
    return adjusted, q, it


IMPLICIT_PRIOR_SMOOTHING = 1e-3


def implicit_prior(dev_probs: np.ndarray, smoothing: float = IMPLICIT_PRIOR_SMOOTHING) -> np.ndarray:
    """Zero-shot source prior = mean dev prediction (BBSE-style proxy), smoothed (+smoothing, renormalized)
    so a class the model never predicts on dev does not get pi ~ 0 and explode the EM ratio q/pi."""
    pi = dev_probs.mean(axis=0) + smoothing
    return pi / pi.sum()


def prior_adjust(probs: np.ndarray, mask: np.ndarray, source_prior: np.ndarray,
                 target_prior: np.ndarray) -> np.ndarray:
    """Oracle correction with the true target prior (upper bound arm)."""
    return _renorm(probs * (np.maximum(target_prior, 0) / np.maximum(source_prior, 1e-12)), mask)


# ---------------------------------------------------------------- metrics on matrices
@dataclass(frozen=True)
class MatrixMetrics:
    accuracy: float
    macro_f1: float
    ece: float
    brier: float
    nll: float
    aurc: float


def argmax_by_position(probs: np.ndarray, positions: np.ndarray | None) -> np.ndarray:
    """Argmax with ties broken by earliest candidate position (spec sec 8; same rule as
    domain.predictions.argmax_label). `positions` holds each class's position in the example's
    candidate order (large value outside the candidates); None -> lowest class index."""
    if positions is None:
        return probs.argmax(axis=1)
    best = probs.max(axis=1, keepdims=True)
    tied = np.isclose(probs, best, rtol=0.0, atol=1e-12) & (probs > 0)
    return np.where(tied, positions, np.iinfo(np.int64).max).argmin(axis=1)


def matrix_metrics(probs: np.ndarray, gold: np.ndarray, n_bins: int = 15,
                   positions: np.ndarray | None = None) -> MatrixMetrics:
    pred = argmax_by_position(probs, positions)
    conf = probs.max(axis=1)
    correct = (pred == gold).astype(float)
    n, k = probs.shape
    # ECE, top-label, equal-width bins
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece_v = sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * np.sum(bins == b) / n
                for b in range(n_bins) if np.any(bins == b))
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), gold] = 1.0
    brier_v = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    # macro-F1 over classes present in gold or predictions
    f1s = []
    for c in np.union1d(gold, pred):
        tp = np.sum((pred == c) & (gold == c))
        fp = np.sum((pred == c) & (gold != c))
        fn = np.sum((pred != c) & (gold == c))
        f1s.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    # AURC with tie-group averaging (same convention as domain.metrics.aurc)
    order = np.argsort(-conf, kind="stable")
    c_sorted, err_sorted = conf[order], 1.0 - correct[order]
    _, inv, counts = np.unique(-c_sorted, return_inverse=True, return_counts=True)
    group_err = np.bincount(inv, weights=err_sorted) / counts
    risks = np.cumsum(group_err[inv]) / np.arange(1, n + 1)
    return MatrixMetrics(float(correct.mean()), float(np.mean(f1s)), float(ece_v), brier_v,
                         _nll(probs, gold), float(risks.mean()))


def coverage_threshold(confidence: np.ndarray, coverage: float) -> float:
    """Smallest confidence among the top `coverage` fraction (accept if conf >= threshold)."""
    k = max(1, int(np.ceil(coverage * len(confidence))))
    return float(np.sort(confidence)[::-1][k - 1])


def selective_risk(probs: np.ndarray, gold: np.ndarray, threshold: float,
                   positions: np.ndarray | None = None) -> tuple[float, float]:
    """(risk, coverage) when accepting examples with confidence >= threshold."""
    conf = probs.max(axis=1)
    accepted = conf >= threshold
    if not np.any(accepted):
        return float("nan"), 0.0
    pred = argmax_by_position(probs, positions)
    risk = float(np.mean(pred[accepted] != gold[accepted]))
    return risk, float(np.mean(accepted))
