"""Paper figures from the FINAL Stage 1 results (never pilot runs). Writes paper/figures/*.pdf.

Inputs: results/final_analysis/primary/holm_family.json (P1, P2, PA1-PA3 with Holm-adjusted p) and
results/final_*/exp2/<model>/<dataset>.jsonl (Exp 2 rows). Colours follow the system
(Jev blue, DeBERTa orange, Tev1-4B violet, Nimble-9B magenta); significance is shown by marker fill (secondary encoding), never by colour alone.

Run: .venv/Scripts/python scripts/paper/make_figures.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
HOLM = ROOT / "results" / "final_analysis" / "primary" / "holm_family.json"
EXP2 = {"jev": ROOT / "results" / "final_jev" / "exp2" / "jev",
        "deberta-v3-large": ROOT / "results" / "final_train" / "exp2" / "deberta-v3-large"}

DATASETS = ["clinc150", "hwu64", "fpb", "hatexplain", "trec6"]
DS_LABEL = {"clinc150": "CLINC150", "hwu64": "HWU64", "fpb": "FPB", "hatexplain": "HateXplain", "trec6": "TREC-6"}
# colours follow the system in every paper figure (validated categorical slots, light; see fig_calibration.py)
BLUE, ORANGE, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#e87ba4", "#6250d6"   # Jev, DeBERTa, Nimble-9B, Tev1-4B
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ALPHA_SIG = 0.05

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def holm_rows() -> dict[tuple[str, str], dict]:
    data = json.loads(HOLM.read_text(encoding="utf-8"))
    if not data.get("complete"):
        raise SystemExit("Holm family is not complete: refusing to draw paper figures")
    return {(r["test"], r["dataset"]): r for r in data["results"]}


def forest(ax, rows, color, offset=0.0, label=None, scale=100.0, marker="o"):
    """One series of paired differences (in points when scale=100) with capped 95% CIs; filled = Holm-significant."""
    for i, ds in enumerate(DATASETS):
        r = rows[ds]
        y = i + offset
        lo, hi = r["ci_low"] * scale, r["ci_high"] * scale
        ax.plot([lo, hi], [y, y], color=color, lw=1.3, solid_capstyle="butt", zorder=2)
        ax.plot([lo, lo], [y - 0.09, y + 0.09], color=color, lw=1.1, zorder=2)
        ax.plot([hi, hi], [y - 0.09, y + 0.09], color=color, lw=1.1, zorder=2)
        sig = r["p_holm"] < ALPHA_SIG
        ax.plot(r["estimate"] * scale, y, marker, ms=5.8, mew=1.4, color=color,
                mfc=color if sig else "white", label=label if i == 0 else None, zorder=3)


def style_forest(ax, xlabel, title):
    for i in range(0, len(DATASETS), 2):    # light band on every other dataset row
        ax.axhspan(i - 0.5, i + 0.5, color="#f4f3ef", lw=0, zorder=0)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_yticks(range(len(DATASETS)), [DS_LABEL[d] for d in DATASETS])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(DATASETS) - 0.5, -0.5)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")


def sig_handles():
    from matplotlib.lines import Line2D
    return [Line2D([], [], marker="s", ls="", ms=5.5, mew=1.4, color=INK2, mfc=INK2, label="Holm-adjusted $p<0.05$"),
            Line2D([], [], marker="s", ls="", ms=5.5, mew=1.4, color=INK2, mfc="white", label="not significant")]


def bottom_legend(fig, handles, ncol):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, handletextpad=0.3,
               columnspacing=1.6, bbox_to_anchor=(0.5, -0.02))


def fig_teaser(h):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.0), sharey=True)
    forest(axes[0], {d: h[("P2-F1", d)] for d in DATASETS}, MAGENTA, marker="D")
    style_forest(axes[0], "$\\Delta$ Macro-F1 (points)", "(a) Nimble-9B $-$ base, F1 ($\\uparrow$)")
    forest(axes[1], {d: h[("P2-ECE", d)] for d in DATASETS}, MAGENTA, marker="D")
    style_forest(axes[1], "$\\Delta$ ECE (points)", "(b) Nimble-9B $-$ base, ECE ($\\downarrow$)")
    forest(axes[2], {d: h[("PA3", d)] for d in DATASETS}, BLUE)
    style_forest(axes[2], "$\\Delta$ ECE (points)", "(c) Jev, EM $-$ none ($\\downarrow$)")
    fig.tight_layout(w_pad=1.0, rect=(0, 0.1, 1, 1))
    bottom_legend(fig, sig_handles(), 2)
    save(fig, "fig1_teaser")
    plt.close(fig)


def fig_pairs(h):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), sharey=True)
    for ax, metric, xl, title in ((axes[0], "F1", "$\\Delta$ Macro-F1 (points)", "Macro-F1: TDM $-$ base ($\\uparrow$ favours TDM)"),
                                  (axes[1], "ECE", "$\\Delta$ ECE (points)", "ECE: TDM $-$ base ($\\downarrow$ favours TDM)")):
        forest(ax, {d: h[(f"P1-{metric}", d)] for d in DATASETS}, VIOLET, -0.17, "P1: Tev1-4B $-$ Qwen3.5-4B", marker="o")
        forest(ax, {d: h[(f"P2-{metric}", d)] for d in DATASETS}, MAGENTA, 0.17, "P2: Nimble-9B $-$ Qwen3.5-9B", marker="D")
        style_forest(ax, xl, title)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.12, 1, 1))
    series = axes[0].get_legend_handles_labels()[0]
    bottom_legend(fig, series + sig_handles(), 4)
    save(fig, "fig2_pairs")
    plt.close(fig)


def exp2_rows(model: str, dataset: str) -> dict[tuple[str, str], list[dict]]:
    out = defaultdict(list)
    with (EXP2[model] / f"{dataset}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["sampling"] == "with_replacement":
                out[(r["alpha"], r["arm"])].append(r)
    return out


def fig_shift():
    alphas = ["10", "1", "0.1"]
    fig, axes = plt.subplots(1, 5, figsize=(6.6, 2.05), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for model, color, marker, lab in (("jev", BLUE, "o", "Jev"), ("deberta-v3-large", ORANGE, "s", "DeBERTa-v3-large")):
            rows = exp2_rows(model, ds)
            base = np.mean([r["ece"] for r in rows[("noshift_resampled", "none")]])
            y = [100 * (np.mean([r["ece"] for r in rows[(a, "none")]]) - base) for a in alphas]
            ax.plot(range(3), y, color=color, lw=1.6, marker=marker, ms=4.5, label=lab)
        ax.axhline(0, color=INK2, lw=0.8, zorder=0)
        ax.set_xticks(range(3), ["10", "1", "0.1"])
        ax.set_title(DS_LABEL[ds], loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("$\\Delta$ECE (points)")
    fig.supxlabel("Dirichlet concentration $\\alpha$ (smaller = stronger shift)", fontsize=8, y=0.12)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.14, 1, 1))
    bottom_legend(fig, axes[0].get_legend_handles_labels()[0], 2)
    save(fig, "fig3_shift")
    plt.close(fig)


def fig_arms():
    arms = [("none", "none"), ("temperature", "temp."), ("em", "EM"), ("bcts_em", "BCTS+EM"), ("oracle", "oracle")]
    fig, axes = plt.subplots(1, 5, figsize=(6.6, 2.0), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for model, color, marker, off, lab in (("jev", BLUE, "o", -0.12, "Jev"),
                                               ("deberta-v3-large", ORANGE, "s", 0.12, "DeBERTa-v3-large")):
            rows = exp2_rows(model, ds)
            y = [100 * np.mean([r["ece"] for r in rows[("1", a)]]) for a, _ in arms]
            ax.plot(np.arange(len(arms)) + off, y, ls="", marker=marker, ms=4.5, color=color, label=lab)
        ax.set_xticks(range(len(arms)), [lab for _, lab in arms], rotation=60, ha="right")
        ax.set_title(DS_LABEL[ds], loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("ECE at $\\alpha=1$ (points)")
    fig.tight_layout(w_pad=0.6, rect=(0, 0.08, 1, 1))
    bottom_legend(fig, axes[0].get_legend_handles_labels()[0], 2)
    save(fig, "fig4_arms")
    plt.close(fig)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}_preview.png", dpi=200)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    h = holm_rows()
    fig_teaser(h)
    fig_pairs(h)
    fig_shift()
    fig_arms()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
