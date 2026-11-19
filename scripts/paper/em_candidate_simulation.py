"""Exploratory simulation (added after the results): does EM on candidate-restricted posteriors recover the target
prior? Uses the frozen EM code (s1bench.domain.label_shift.em_prior_adjust) on synthetic data whose posteriors are
exactly calibrated, so any error comes from the estimator and not from the model.

Generative model: L classes in fixed groups, x | y ~ N(mu_y, I_d), source prior pi_s. The model's output is the Bayes
posterior restricted to the candidate set, p(y | x, C) proportional to pi_s(y) N(x; mu_y, I) on C.
Shift as in Exp 2: target q ~ Dirichlet(alpha L pi_s), alpha = 1, n test examples drawn from q.
Candidate set: (a) the group of the gold label (our design; C is fixed by y); (b) an imperfect router that returns the
gold group except for a share eps of the examples, which get a random other group (the gold label is then missing from
C and the restricted posterior is misspecified). Reported: total-variation distance to q of the EM estimate, of the
source prior (no correction) and of the sample's class shares (labels known: the sampling floor), mean over draws.

Second table (posteriors distorted: over-/underconfident, label bias; EM with the implicit or the true source prior):
paper/tables/tab_em_miscalibration.tex.

Writes paper/tables/tab_em_simulation.tex. Run: .venv/Scripts/python scripts/paper/em_candidate_simulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from s1bench.domain import label_shift as ls  # noqa: E402

SEED = 20261104
DRAWS = 200
D, SEP = 16, 0.9          # feature dimension, scale of the class means (Bayes accuracy about 0.9 within a group)
SETTINGS = [("150 classes, 8 groups", 150, 8, 4500), ("64 classes, 3 groups", 64, 3, 1076),
            ("3 classes, all labels", 3, 1, 970)]
EPS = [0.0, 0.05, 0.1, 0.2]


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def simulate(n_classes: int, n_groups: int, n: int, rng: np.random.Generator) -> dict[str, list[float]]:
    group = np.concatenate([np.full(len(g), k) for k, g in enumerate(np.array_split(np.arange(n_classes), n_groups))])
    mu = rng.normal(0.0, SEP, size=(n_classes, D))
    pi_s = rng.dirichlet(np.full(n_classes, 5.0))
    out: dict[str, list[float]] = {"none": [], "sample": [], **{f"eps{e}": [] for e in EPS},
                                   **{f"miss{e}": [] for e in EPS}, "acc": []}
    for _ in range(DRAWS):
        q = ls.dirichlet_target_prior(pi_s, 1.0, rng)
        y = rng.choice(n_classes, size=n, p=q)
        x = mu[y] + rng.normal(size=(n, D))
        loglik = -0.5 * ((x[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
        joint = np.exp(loglik - loglik.max(axis=1, keepdims=True)) * pi_s        # source posterior, unnormalised
        out["none"].append(tv(pi_s, q))
        out["sample"].append(tv(np.bincount(y, minlength=n_classes) / n, q))   # labels known: sampling floor
        for e in EPS:
            chosen = group[y]                                                      # our design: C = G(y)
            if e > 0.0 and n_groups > 1:                                           # router wrong for a share eps
                other = (chosen + rng.integers(1, n_groups, size=n)) % n_groups
                chosen = np.where(rng.random(n) < e, other, chosen)
            mask = group[None, :] == chosen[:, None]
            probs = np.where(mask, joint, 0.0)
            probs /= probs.sum(axis=1, keepdims=True)
            _, q_hat, _ = ls.em_prior_adjust(probs, mask, pi_s, max_iter=1000, tol=1e-8)
            out[f"eps{e}"].append(tv(q_hat, q))
            out[f"miss{e}"].append(float(np.mean(chosen != group[y])))
            if e == 0.0:
                out["acc"].append(float(np.mean(probs.argmax(axis=1) == y)))
    return out


DISTORTIONS = [("calibrated", 1.0, 0.0), ("overconfident ($p^2$)", 2.0, 0.0), ("underconfident ($p^{1/2}$)", 0.5, 0.0),
               ("label bias", 1.0, 0.5)]


def simulate_miscalibrated(n_classes: int, n_groups: int, n: int, rng: np.random.Generator) -> dict[str, list[float]]:
    """Gold-group candidate sets; the model's posteriors are distorted (power and/or a fixed per-class bias), so they
    are no longer the source posteriors. EM uses either the implicit prior (mean distorted posterior on an unshifted
    source development sample of the same size, smoothed as in the pipeline) or the true source prior."""
    group = np.concatenate([np.full(len(g), k) for k, g in enumerate(np.array_split(np.arange(n_classes), n_groups))])
    mu = rng.normal(0.0, SEP, size=(n_classes, D))
    pi_s = rng.dirichlet(np.full(n_classes, 5.0))
    bias = np.exp(rng.normal(0.0, 1.0, size=n_classes))
    out: dict[str, list[float]] = {}

    def posteriors(y: np.ndarray, power: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
        x = mu[y] + rng.normal(size=(len(y), D))
        loglik = -0.5 * ((x[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
        joint = np.exp(loglik - loglik.max(axis=1, keepdims=True)) * pi_s
        mask = group[None, :] == group[y][:, None]
        p = np.where(mask, joint, 0.0)
        p /= p.sum(axis=1, keepdims=True)
        p = np.where(mask, p ** power * (bias ** sigma if sigma else 1.0), 0.0)
        return p / p.sum(axis=1, keepdims=True), mask

    for _ in range(DRAWS):
        q = ls.dirichlet_target_prior(pi_s, 1.0, rng)
        y_dev, y = rng.choice(n_classes, size=n, p=pi_s), rng.choice(n_classes, size=n, p=q)
        for name, power, sigma in DISTORTIONS:
            p_dev, _ = posteriors(y_dev, power, sigma)
            p, mask = posteriors(y, power, sigma)
            out.setdefault(f"{name}|ece", []).append(ls.matrix_metrics(p_dev, y_dev).ece)
            out.setdefault(f"{name}|none", []).append(tv(pi_s, q))
            for key, prior in (("implicit", ls.implicit_prior(p_dev)), ("true", pi_s)):
                _, q_hat, _ = ls.em_prior_adjust(p, mask, prior, max_iter=1000, tol=1e-8)
                out.setdefault(f"{name}|{key}", []).append(tv(q_hat, q))
    return out


def table_miscalibrated(rng: np.random.Generator) -> None:
    settings = [SETTINGS[0], SETTINGS[2]]
    lines = ["\\begin{tabular}{l" + "rrrr" * len(settings) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{name}}}" for name, *_ in settings) + " \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 4 * k}-{5 + 4 * k}}}" for k in range(len(settings))),
             "Posteriors" + " & ECE & No corr. & EM, implicit & EM, true $\\pi_s$" * len(settings) + " \\\\",
             "\\midrule"]
    res = [simulate_miscalibrated(nc, ng, n, rng) for _, nc, ng, n in settings]
    for name, *_ in DISTORTIONS:
        cells = []
        for r in res:
            cells += [f"{np.mean(r[f'{name}|{k}']):.3f}" for k in ("ece", "none", "implicit", "true")]
        print(f"{name:28s} " + " ".join(cells))
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper" / "tables" / "tab_em_miscalibration.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(SEED)
    lines = ["\\begin{tabular}{lrrr" + "r" * (len(EPS) - 1) + "}", "\\toprule",
             " & & & \\multicolumn{" + str(len(EPS)) + "}{c}{EM estimate, candidate set from} \\\\",
             "\\cmidrule(lr){4-" + str(3 + len(EPS)) + "}",
             "Setting & No correction & Labels known & gold group & "
             + " & ".join(f"router, $\\varepsilon={e:g}$" for e in EPS[1:]) + " \\\\", "\\midrule"]
    for name, n_classes, n_groups, n in SETTINGS:
        r = simulate(n_classes, n_groups, n, rng)
        cells = [f"{np.mean(r['none']):.3f}", f"{np.mean(r['sample']):.3f}", f"{np.mean(r['eps0.0']):.3f}"]
        cells += [f"{np.mean(r[f'eps{e}']):.3f} ({100 * np.mean(r[f'miss{e}']):.1f}\\%)" for e in EPS[1:]]
        print(f"{name:24s} n={n:5d} acc={np.mean(r['acc']):.3f} " + " | ".join(cells))
        if n_groups == 1:   # one group: the router always picks it, so every column equals the gold-group column
            cells[3:] = ["--"] * (len(EPS) - 1)
        lines.append(" & ".join([f"{name} ($n={n:,}$)".replace(",", "{,}")] + cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper" / "tables" / "tab_em_simulation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if "--miscalibration" in sys.argv:   # second table, tab_em_miscalibration (long run: about an hour on a CPU)
        table_miscalibrated(np.random.default_rng(SEED + 1))


if __name__ == "__main__":
    main()
