"""Exploratory (added after the results): PA1 and PA2 with the DeBERTa-v3-large training seed resampled inside the
bootstrap. Identical to the frozen pa_tests (s1bench.application.primary_tests) - same 50 fixed target priors, same
class-stratified base bootstrap, same shifted samples and no-shift controls, same thresholds - except that every
replicate first draws one of the five full-data DeBERTa seeds at random, so the interval covers seed variation as well
as the finite test set. PA3 involves Jev only and is unchanged. B = 1,000 (as the prior-set sensitivity).

Writes results/final_analysis_calibrob/pa_seed_bootstrap.json and paper/tables/tab_pa_seed_bootstrap.tex.
Run: .venv/Scripts/python scripts/paper/pa_seed_bootstrap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from s1bench.application.primary_tests import _ece, _risk, _summ, aligned_matrices  # noqa: E402
from s1bench.domain import label_shift as ls  # noqa: E402
from s1bench.infrastructure.storage.processed_store import YamlLabelRepository  # noqa: E402
from s1bench.infrastructure.storage.results_store import read_predictions  # noqa: E402

RES = ROOT / "results"
OUT = RES / "final_analysis_calibrob"
DATASETS = ["clinc150", "hwu64", "fpb", "hatexplain", "trec6"]
DS = {"clinc150": "CLINC150", "hwu64": "HWU64", "fpb": "FPB", "hatexplain": "HateXplain", "trec6": "TREC-6"}
SEEDS = ["deberta-v3-large"] + [f"deberta-v3-large@fulls{s}" for s in (1002, 1003, 1004, 1005)]
B = 1_000


def pa_seed_bootstrap(a_dev, a_test, b_devs, b_tests, train_prior, cfg) -> dict:
    n = len(a_test.gold)
    rng = np.random.default_rng(cfg["seed"])
    priors = [ls.dirichlet_target_prior(train_prior, 1.0, rng) for _ in range(int(cfg["pa"]["n_priors"]))]
    tau_a = ls.coverage_threshold(a_dev.probs.max(axis=1), cfg["coverage"])
    tau_b = [ls.coverage_threshold(d.probs.max(axis=1), cfg["coverage"]) for d in b_devs]
    pa1, pa2 = np.empty(B), np.empty(B)
    for r in range(B):
        s = int(rng.integers(len(b_tests)))                     # the only change: draw a DeBERTa seed
        b_test = b_tests[s]
        base = ls.stratified_bootstrap(a_test.gold, rng)
        risk_a0, risk_b0 = _risk(a_test, base, tau_a), _risk(b_test, base, tau_b[s])
        q0 = np.bincount(a_test.gold[base], minlength=len(train_prior)) / len(base)
        d1, d2 = [], []
        for q in priors:
            s_idx = base[ls.resample_with_replacement(a_test.gold[base], q, n, rng)]
            n_idx = base[ls.resample_with_replacement(a_test.gold[base], q0, n, rng)]
            ea, eb = _ece(a_test, s_idx), _ece(b_test, s_idx)
            d1.append((ea - _ece(a_test, n_idx)) - (eb - _ece(b_test, n_idx)))
            d2.append((_risk(a_test, s_idx, tau_a) - risk_a0) - (_risk(b_test, s_idx, tau_b[s]) - risk_b0))
        pa1[r], pa2[r] = np.nanmean(d1), np.nanmean(d2)
    return {t.test: {"estimate": t.estimate, "ci_low": t.ci_low, "ci_high": t.ci_high, "p": t.p_value}
            for t in (_summ("PA1", "", pa1), _summ("PA2", "", pa2))}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / "exp2.yaml").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pa_seed_bootstrap.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for ds in DATASETS:
        if ds in out:
            continue
        labels = YamlLabelRepository(ROOT / "data" / "labels").load(ds)
        jev = {s: read_predictions(RES / "final_jev", "jev", ds, s) for s in ("dev", "test")}
        a_dev = a_test = None
        b_devs, b_tests = [], []
        for m in SEEDS:
            ad, bd, _ = aligned_matrices(jev["dev"], read_predictions(RES / "final_train", m, ds, "dev"), labels)
            at, bt, _ = aligned_matrices(jev["test"], read_predictions(RES / "final_train", m, ds, "test"), labels)
            if a_test is not None and not np.array_equal(at.gold, a_test.gold):
                raise SystemExit(f"{ds}: seeds are not aligned")
            a_dev, a_test = ad, at
            b_devs.append(bd)
            b_tests.append(bt)
        report = json.loads((ROOT / "data" / "processed" / ds / "report.json").read_text(encoding="utf-8"))
        counts = report["label_counts"]["train"]
        train_prior = np.array([counts.get(lid, 0) for lid in labels.ids()], dtype=float)
        train_prior /= train_prior.sum()
        out[ds] = pa_seed_bootstrap(a_dev, a_test, b_devs, b_tests, train_prior, cfg)
        path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(ds, {k: f"{100 * v['estimate']:+.2f} [{100 * v['ci_low']:+.2f}, {100 * v['ci_high']:+.2f}]"
                   for k, v in out[ds].items()}, flush=True)

    holm = {(r["test"], r["dataset"]): r for r in json.loads(
        (RES / "final_analysis" / "primary" / "holm_family.json").read_text(encoding="utf-8"))["results"]}
    f = lambda x: f"{100 * x:+.1f}".replace("-", "$-$")
    lines = ["\\begin{tabular}{ll" + "r" * len(DATASETS) + "}", "\\toprule",
             "Test & Bootstrap & " + " & ".join(DS[d] for d in DATASETS) + " \\\\", "\\midrule"]
    for t in ("PA1", "PA2"):
        lines.append(f"{t} & seed 1001 (frozen) & " + " & ".join(
            f"{f(holm[(t, d)]['estimate'])} [{f(holm[(t, d)]['ci_low'])}, {f(holm[(t, d)]['ci_high'])}]" for d in DATASETS) + " \\\\")
        lines.append(f" & seeds resampled & " + " & ".join(
            f"{f(out[d][t]['estimate'])} [{f(out[d][t]['ci_low'])}, {f(out[d][t]['ci_high'])}]" for d in DATASETS) + " \\\\")
        lines.append("\\midrule" if t == "PA1" else "\\bottomrule")
    lines.append("\\end{tabular}")
    (ROOT / "paper" / "tables" / "tab_pa_seed_bootstrap.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
