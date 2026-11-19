"""Paper figures on calibration, thresholds and implicit priors from the FINAL Stage 1 runs (never pilot runs).
Writes paper/figures/fig_reliability.pdf, fig_riskcoverage.pdf, fig_implicit_prior.pdf (+ *_preview.png).

Inputs: test and dev predictions of final_jev, final_local, final_train (DeBERTa-v3-large seed 1001, the model of the
Exp 1 table and of the confirmatory tests); Exp 2 rows (results/final_*/exp2/) for the threshold operating points.
Every ECE and AURC drawn here is recomputed with the frozen metric code and checked against
results/final_analysis/exp1_test*.json; the script stops on any mismatch.

Colours follow the system in every paper figure (validated categorical slots, light): Jev blue, DeBERTa orange,
QLoRA aqua, TF-IDF yellow, Nimble-9B magenta, Qwen3.5-9B green, Tev1-4B violet, Qwen3.5-4B red. Series also differ
by marker shape.

Run: .venv/Scripts/python scripts/paper/fig_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from s1bench.application.exp1_metrics import _scored  # noqa: E402
from s1bench.application.prevalence_shift import to_matrices  # noqa: E402
from s1bench.domain import label_shift as ls  # noqa: E402
from s1bench.domain import metrics as mt  # noqa: E402
from s1bench.infrastructure.storage.processed_store import YamlLabelRepository  # noqa: E402
from s1bench.infrastructure.storage.results_store import read_predictions  # noqa: E402

OUT = ROOT / "paper" / "figures"
RES = ROOT / "results"
DATASETS = ["clinc150", "hwu64", "fpb", "hatexplain", "trec6"]
DS_LABEL = {"clinc150": "CLINC150", "hwu64": "HWU64", "fpb": "FPB", "hatexplain": "HateXplain", "trec6": "TREC-6"}
INK, INK2, GRID, TRUE = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
# system -> (run, label, colour, marker)
SYS = {"jev": ("final_jev", "Jev", "#2a78d6", "o"),
       "deberta-v3-large": ("final_train", "DeBERTa-v3-large", "#eb6834", "s"),
       "nimble-9b": ("final_local", "Nimble-9B", "#e87ba4", "v"),
       "qwen3.5-9b": ("final_local", "Qwen3.5-9B", "#008300", "D"),
       "tev1-4b": ("final_local", "Tev1-4B", "#6250d6", "^")}
N_BINS = 15

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def exp1_reference() -> dict[tuple[str, str], dict]:
    rows = []
    for f in ("exp1_test.json", "exp1_test_jev.json"):
        rows += json.loads((RES / "final_analysis" / f).read_text(encoding="utf-8"))
    return {(r["model"], r["dataset"]): r for r in rows if r["split"] == "test"}


EXP1 = exp1_reference()


def scored_test(model: str, dataset: str) -> list[mt.Scored]:
    preds = read_predictions(RES / SYS[model][0], model, dataset, "test")
    if any(p.status != "ok" for p in preds):
        raise SystemExit(f"{model}/{dataset}: non-ok rows; this figure assumes none (Exp 1 has none)")
    rows = [_scored(p) for p in preds]
    for key, val in (("ece", mt.ece(rows)), ("aurc", mt.aurc(rows))):
        if abs(val - EXP1[(model, dataset)][key]) > 1e-9:
            raise SystemExit(f"{model}/{dataset}: {key} {val} differs from Exp 1 {EXP1[(model, dataset)][key]}")
    return rows


def reliability_bins(rows: list[mt.Scored]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean confidence, accuracy and count of the non-empty bins (same binning as metrics.ece)."""
    conf = np.array([r.confidence for r in rows])
    corr = np.array([r.correct for r in rows], dtype=float)
    idx = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    keep = [b for b in range(N_BINS) if np.any(idx == b)]
    return (np.array([conf[idx == b].mean() for b in keep]), np.array([corr[idx == b].mean() for b in keep]),
            np.array([np.sum(idx == b) for b in keep]))


MIN_BIN = 10   # bins with fewer examples are not drawn (their share is printed for the caption)


def fig_reliability() -> None:
    """Reliability bars (Guo et al. 2017 style), one panel per system and dataset: bar = accuracy of a confidence bin
    (the 15 ECE bins), hatched = gap to the diagonal, faint bars = bins with fewer than MIN_BIN examples."""
    from matplotlib.patches import Patch
    rows_of_panels = [[("jev", "fpb"), ("deberta-v3-large", "fpb"), ("jev", "hatexplain"), ("deberta-v3-large", "hatexplain")],
                      [("nimble-9b", "fpb"), ("qwen3.5-9b", "fpb"), ("nimble-9b", "hwu64"), ("qwen3.5-9b", "hwu64")]]
    fig, grid = plt.subplots(2, 4, figsize=(6.6, 3.75), sharex=True, sharey=True)
    width = 1.0 / N_BINS
    edges = np.arange(N_BINS) * width
    for r, panel_row in enumerate(rows_of_panels):
        for c, (m, ds) in enumerate(panel_row):
            ax = grid[r, c]
            rows = scored_test(m, ds)
            conf = np.array([s.confidence for s in rows])
            corr = np.array([s.correct for s in rows], dtype=float)
            idx = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
            n = np.bincount(idx, minlength=N_BINS)
            acc = np.array([corr[idx == b].mean() if n[b] else 0.0 for b in range(N_BINS)])
            mid = np.array([conf[idx == b].mean() if n[b] else 0.0 for b in range(N_BINS)])
            _, lab, col, _ = SYS[m]
            for b in np.nonzero(n)[0]:
                faint = n[b] < MIN_BIN
                ax.bar(edges[b], acc[b], width, align="edge", color=col, alpha=0.3 if faint else 0.9,
                       edgecolor="white", lw=0.4, zorder=2)
                lo, hi = sorted((acc[b], mid[b]))
                ax.bar(edges[b], hi - lo, width, bottom=lo, align="edge", color="none", edgecolor=INK2,
                       hatch="////", lw=0.0, alpha=0.3 if faint else 0.65, zorder=3)
            ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=4)
            ax.text(0.04, 0.96, f"ECE {100 * mt.ece(rows):.1f}\n$n$ = {len(rows):,}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.2, color=INK, linespacing=1.3, zorder=5,
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.95})
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
            ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
            ax.set_title(f"{lab.replace('-v3-large', '')} · {DS_LABEL[ds]}", loc="left", fontsize=8)
            print(f"  reliability {m}/{ds}: {100 * n[n < MIN_BIN].sum() / len(rows):.1f}% of examples in bins < {MIN_BIN}")
    for ax in grid[:, 0]:
        ax.set_ylabel("Accuracy")
    for ax in grid[1]:
        ax.set_xlabel("Confidence")
    handles = [Patch(fc=INK2, alpha=0.5, label="accuracy of bin (system colour)"),
               Patch(fc="none", ec=INK2, hatch="////", lw=0.0, label="gap to perfect calibration"),
               plt.Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 2)), label="perfect calibration")]
    fig.tight_layout(w_pad=0.6, h_pad=0.9, rect=(0, 0.06, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.01), fontsize=7.5)
    save(fig, "fig_reliability")


def risk_coverage(rows: list[mt.Scored]) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk at coverage k/n, examples by decreasing confidence, tied groups averaged (as metrics.aurc)."""
    ordered = sorted(rows, key=lambda r: -r.confidence)
    errors, i = [], 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j].confidence == ordered[i].confidence:
            j += 1
        errors += [sum(not r.correct for r in ordered[i:j]) / (j - i)] * (j - i)
        i = j
    k = np.arange(1, len(errors) + 1)
    return k / len(errors), np.cumsum(errors) / k


def operating_points(model: str, dataset: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """(coverage, risk) of the dev threshold for 80% coverage: original test set, and mean over the 200 shifted
    test sets at alpha = 1 (with replacement), arm 'none' (Exp 2 rows)."""
    path = RES / SYS[model][0] / "exp2" / model / f"{dataset}.jsonl"
    orig, shifted = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["arm"] != "none":
            continue
        if r["alpha"] == "inf":
            orig = (r["coverage_shifted"], r["risk_shifted"])
        elif r["alpha"] == "1" and r["sampling"] == "with_replacement":
            shifted.append((r["coverage_shifted"], r["risk_shifted"]))
    if orig is None or len(shifted) != 200:
        raise SystemExit(f"{model}/{dataset}: Exp 2 rows incomplete")
    return orig, tuple(np.mean(np.array(shifted), axis=0))


def fig_riskcoverage() -> None:
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 5, figsize=(6.6, 2.0))
    for ax, ds in zip(axes, DATASETS):
        for m in ("jev", "deberta-v3-large"):
            rows = scored_test(m, ds)
            cov, risk = risk_coverage(rows)
            if abs(risk.mean() - EXP1[(m, ds)]["aurc"]) > 1e-9:
                raise SystemExit(f"{m}/{ds}: risk-coverage curve does not reproduce the AURC")
            _, lab, col, mk = SYS[m]
            ax.plot(cov, 100 * risk, color=col, lw=1.1, zorder=2, label=lab)
            (c0, r0), (c1, r1) = operating_points(m, ds)
            ax.annotate("", xy=(c1, 100 * r1), xytext=(c0, 100 * r0),
                        arrowprops={"arrowstyle": "-|>", "color": col, "lw": 0.7, "mutation_scale": 6}, zorder=3)
            ax.plot(c0, 100 * r0, marker=mk, ms=4.2, color=col, mfc="white", mew=1.0, ls="", zorder=4)
            ax.plot(c1, 100 * r1, marker=mk, ms=4.2, color=col, mec="white", mew=0.5, ls="", zorder=4)
        ax.axvline(0.8, color=INK2, lw=0.5, ls=(0, (1, 2)), zorder=1)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(DS_LABEL[ds], loc="left")
        ax.set_xlabel("Coverage")
    axes[0].set_ylabel("Selective risk (%)")
    series = [Line2D([], [], color=SYS[m][2], lw=1.1, marker=SYS[m][3], ms=4, mec="white", mew=0.5,
                     label=SYS[m][1]) for m in ("jev", "deberta-v3-large")]
    points = [Line2D([], [], color=INK2, marker="o", ms=4.2, mfc="white", mew=1.0, ls="",
                     label="threshold, no shift"),
              Line2D([], [], color=INK2, marker="o", ms=4.2, mec="white", mew=0.5, ls="",
                     label="same threshold, $\\alpha=1$ (mean)")]
    fig.tight_layout(w_pad=0.4, rect=(0, 0.1, 1, 1))
    fig.legend(handles=series + points, loc="lower center", ncol=4, frameon=False, handletextpad=0.3,
               columnspacing=1.4, bbox_to_anchor=(0.5, -0.02))
    save(fig, "fig_riskcoverage")


def fig_implicit_prior() -> None:
    models = ("jev", "tev1-4b", "nimble-9b")
    sets = ("fpb", "hatexplain", "trec6")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.1), gridspec_kw={"width_ratios": [3, 3, 6]}, sharey=True)
    for ax, ds in zip(axes, sets):
        labels = YamlLabelRepository(ROOT / "data" / "labels").load(ds)
        names = [lab.name for lab in labels.labels]
        width = 0.8 / (len(models) + 1)
        x = np.arange(len(names))
        true = None
        for j, m in enumerate(models):
            dev = to_matrices(read_predictions(RES / SYS[m][0], m, ds, "dev"), labels)
            if dev.n_excluded:
                raise SystemExit(f"{m}/{ds}: non-ok dev rows")
            true = np.bincount(dev.gold, minlength=len(names)) / len(dev.gold)
            ax.bar(x + (j + 1 - len(models) / 2) * width, 100 * ls.implicit_prior(dev.probs), width,
                   color=SYS[m][2], edgecolor="white", lw=0.6, label=SYS[m][1], zorder=2)
        ax.bar(x - len(models) / 2 * width, 100 * true, width, color=TRUE, edgecolor="white", lw=0.6,
               label="true (dev)", zorder=2)
        short = {"abbreviation": "abbr.", "description": "descr.", "numeric value": "number", "hate speech": "hate"}
        ax.set_xticks(x, [short.get(n, n) for n in names], fontsize=7.5)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(DS_LABEL[ds], loc="left")
    axes[0].set_ylabel("Class share (%)")
    handles, labs = axes[0].get_legend_handles_labels()
    order = [len(models)] + list(range(len(models)))   # true distribution first
    fig.tight_layout(w_pad=0.6, rect=(0, 0.1, 1, 1))
    fig.legend([handles[i] for i in order], [labs[i] for i in order], loc="lower center", ncol=4, frameon=False,
               handletextpad=0.3, columnspacing=1.4, bbox_to_anchor=(0.5, -0.02))
    save(fig, "fig_implicit_prior")


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}_preview.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_reliability()
    fig_riskcoverage()
    fig_implicit_prior()
    print("wrote fig_reliability.pdf, fig_riskcoverage.pdf, fig_implicit_prior.pdf")


if __name__ == "__main__":
    main()
