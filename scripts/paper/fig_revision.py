"""Exploratory (revision protocol v1.3; added after the results): figures for the revision analyses. Writes, each
with a *_preview.png for checking:

- fig_budget_realistic.pdf (G1, G2): system - Jev Macro-F1 along the label budget under the realistic protocol
  (selection inside the budget; solid, 95% bootstrap band, crossover ringed), with the frozen fixed-schedule curve of
  the same system as a thin dashed line at the same budgets. Inputs: results/final_realistic_analysis/curves and
  results/final_budget_analysis/curves.
- fig_router.pdf (G3): accuracy with the gold group vs the router's group (CLINC150, HWU64), and mean top-label
  confidence on the misrouted examples (gold label not offered). Input: results/final_analysis_calibrob/
  router_calibration.json.
- fig_prompt_precision.pdf (G5, G6): P1 and P2 on FPB in the TDM format (frozen), two generic formats and, for P2,
  bf16. Input: results/final_analysis_calibrob/prompt_precision.json.

Colours follow the system in every paper figure (see fig_calibration.py). Numbers are read from the analysis outputs
only; nothing is recomputed here.
Run: .venv/Scripts/python scripts/paper/fig_revision.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
RES = ROOT / "results"
REAL = RES / "final_realistic_analysis" / "curves"
FROZEN = RES / "final_budget_analysis" / "curves"
CALIB = RES / "final_analysis_calibrob"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#f4f3ef"

DATASETS = ["hatexplain", "fpb", "trec6", "hwu64", "clinc150"]   # by number of classes, as in fig5_budget
DS_LABEL = {"clinc150": "CLINC150 (150)", "hwu64": "HWU64 (64)", "fpb": "FPB (3)", "hatexplain": "HateXplain (3)",
            "trec6": "TREC-6 (6)"}
SERIES = [("deberta-v3-large", "DeBERTa-v3-large", ORANGE, "s"), ("tfidf-lr", "TF-IDF + LR", YELLOW, "D"),
          ("qwen3.5-4b-qlora", "Qwen3.5-4B + QLoRA", AQUA, "^")]
SYSTEMS = [("jev", "Jev", BLUE, "o"), ("tev1-4b", "Tev1-4B", VIOLET, "^"), ("nimble-9b", "Nimble-9B", MAGENTA, "v"),
           ("qwen3.5-4b", "Qwen3.5-4B", RED, "P"), ("qwen3.5-9b", "Qwen3.5-9B", GREEN, "D"),
           ("deberta-v3-large", "DeBERTa-v3-large", ORANGE, "s"), ("tfidf-lr", "TF-IDF + LR", YELLOW, "X")]

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}_preview.png", dpi=200)
    plt.close(fig)


def fig_budget_realistic() -> None:
    fig, grid = plt.subplots(2, 3, figsize=(6.6, 3.9))
    axes = grid.ravel()
    for ax, ds in zip(axes, DATASETS):
        for system, label, color, marker in SERIES:
            real = load(REAL / f"{system}_vs_jev@final_jev_{ds}.json")
            rs = real["curves"]["macro_f1"]["rows"]
            x = [float(real["sizes"][str(r["budget"])]) for r in rs]
            y = [100 * r["estimate"] for r in rs]
            ax.fill_between(x, [100 * r["ci_low"] for r in rs], [100 * r["ci_high"] for r in rs],
                            color=color, alpha=0.15, lw=0)
            ax.plot(x, y, color=color, lw=1.4, marker=marker, ms=4.2, mfc=color, mec="white", mew=0.6,
                    label=label, zorder=3)
            c = real["curves"]["macro_f1"]["crossover"]["crossover"]
            if c != "not reached":
                k = [r["budget"] for r in rs].index(c)
                ax.plot(x[k], y[k], marker="o", ms=9, mfc="none", mec=color, mew=1.1, zorder=4)
            fro = load(FROZEN / f"{system}_vs_jev@final_jev_{ds}.json")
            budgets = {r["budget"] for r in rs}
            fr = [r for r in fro["curves"]["macro_f1"]["rows"] if r["budget"] in budgets]
            ax.plot([float(fro["sizes"][str(r["budget"])]) for r in fr], [100 * r["estimate"] for r in fr],
                    color=color, lw=0.9, ls=(0, (2.5, 1.5)), alpha=0.9, zorder=2)
        ax.axhline(0, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.text(1.0, 0, " = Jev", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=7, color=INK2, clip_on=False)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([100, 1000]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["100", "1k"])
        ax.set_xlim(70, 7000)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_title(DS_LABEL[ds], loc="left")
    for ax in grid[:, 0]:
        ax.set_ylabel("$\\Delta$ Macro-F1 vs Jev (points)")
    for ax in axes[2:5]:
        ax.set_xlabel("Training labels (log scale)")
    leg = axes[5]
    leg.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color=INK2, lw=0.9, ls=(0, (2.5, 1.5))),
                Line2D([], [], ls="", marker="o", ms=9, mfc="none", mec=INK2, mew=1.1),
                Line2D([], [], color=INK2, lw=0.7, ls=(0, (3, 2)))]
    labels += ["frozen fixed schedule\n(same budgets)", "crossover (realistic):\nfirst budget with\ninterval above 0",
               "no difference (Jev)"]
    leg.legend(handles, labels, loc="center left", frameon=False, fontsize=7.6, handlelength=2.2,
               labelspacing=0.8, borderaxespad=0)
    fig.tight_layout(w_pad=1.6, h_pad=1.0)
    save(fig, "fig_budget_realistic")


def fig_router() -> None:
    d = load(CALIB / "router_calibration.json")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1]})
    ys = range(len(SYSTEMS))
    for ax, ds, title in ((axes[0], "clinc150", "(a) CLINC150 accuracy"), (axes[1], "hwu64", "(b) HWU64 accuracy")):
        for i, (key, _, color, marker) in enumerate(SYSTEMS):
            g, e = 100 * d[ds][key]["gold"]["accuracy"], 100 * d[ds][key]["e2e"]["accuracy"]
            ax.plot([e, g], [i, i], color=color, lw=1.3, zorder=2)
            ax.plot(g, i, marker=marker, ms=5.5, color=color, mfc="white", mew=1.3, zorder=3)
            ax.plot(e, i, marker=marker, ms=5.5, color=color, mfc=color, mew=1.3, zorder=3)
        miss, n = d[ds]["n_misrouted"], d[ds]["jev"]["gold"]["n"]
        ax.set_xlabel(f"Accuracy (%)\nrouter misses {miss:,} of {n:,}")
        ax.set_title(title, loc="left")
    ax = axes[2]
    for i, (key, _, color, marker) in enumerate(SYSTEMS):
        c1, c2 = 100 * d["clinc150"][key]["misrouted_mean_confidence"], 100 * d["hwu64"][key]["misrouted_mean_confidence"]
        ax.plot([min(c1, c2), max(c1, c2)], [i, i], color=color, lw=1.0, alpha=0.6, zorder=2)
        ax.plot(c1, i, marker=marker, ms=5.5, color=color, mfc=color, mew=1.3, zorder=3)
        ax.plot(c2, i, marker=marker, ms=5.5, color=color, mfc="white", mew=1.3, zorder=3)
    ax.set_xlim(30, 90)
    ax.set_xlabel("Mean top-label confidence (%)\non the misrouted examples")
    ax.set_title("(c) confidence, gold not offered", loc="left")
    for ax in axes:
        for i in range(0, len(SYSTEMS), 2):
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(ys), [s[1] for s in SYSTEMS])
    axes[0].set_ylim(len(SYSTEMS) - 0.5, -0.5)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.1, 1, 1))
    h = [Line2D([], [], marker="o", ls="", ms=5.5, mew=1.3, color=INK2, mfc="white", label="(a, b) gold group given"),
         Line2D([], [], marker="o", ls="", ms=5.5, mew=1.3, color=INK2, mfc=INK2, label="(a, b) router's group"),
         Line2D([], [], marker="o", ls="", ms=5.5, mew=1.3, color=INK2, mfc=INK2, label="(c) CLINC150"),
         Line2D([], [], marker="o", ls="", ms=5.5, mew=1.3, color=INK2, mfc="white", label="(c) HWU64")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, handletextpad=0.3, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.02))
    save(fig, "fig_router")


def fig_prompt_precision() -> None:
    d = load(CALIB / "prompt_precision.json")
    rows = [("TDM format (frozen)", "tdm"), ("plain, letters", "+plain"), ("plain, option likelihood", "+plain_names"),
            ("TDM format, bf16", "bf16")]
    pairs = [("P1", "P1: Tev1-4B $-$ Qwen3.5-4B", VIOLET, "o", -0.15),
             ("P2", "P2: Nimble-9B $-$ Qwen3.5-9B", MAGENTA, "D", 0.15)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1), sharey=True)
    for ax, metric, xl, title in ((axes[0], "F1", "$\\Delta$ Macro-F1 (points)", "Macro-F1: TDM $-$ base ($\\uparrow$)"),
                                  (axes[1], "ECE", "$\\Delta$ ECE (points)", "ECE: TDM $-$ base ($\\downarrow$)")):
        for pair, label, color, marker, off in pairs:
            for i, (_, key) in enumerate(rows):
                if key == "bf16":
                    if pair != "P2":
                        continue
                    r = d["g6"]["P2_bf16"][metric]
                else:
                    r = d["g5"][pair][key][metric]
                y = i + off
                lo, hi = 100 * r["ci_low"], 100 * r["ci_high"]
                ax.plot([lo, hi], [y, y], color=color, lw=1.3, solid_capstyle="butt", zorder=2)
                ax.plot([lo, lo], [y - 0.09, y + 0.09], color=color, lw=1.1, zorder=2)
                ax.plot([hi, hi], [y - 0.09, y + 0.09], color=color, lw=1.1, zorder=2)
                ax.plot(100 * r["estimate"], y, marker, ms=5.8, mew=1.4, color=color, mfc=color,
                        label=label if i == 0 else None, zorder=3)
        for i in range(0, len(rows), 2):
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(xl)
        ax.set_title(title, loc="left")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.1, 1, 1))
    fig.legend(handles=axes[0].get_legend_handles_labels()[0], loc="lower center", ncol=2, frameon=False,
               handletextpad=0.3, columnspacing=1.6, bbox_to_anchor=(0.5, -0.02))
    save(fig, "fig_prompt_precision")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_budget_realistic()
    fig_router()
    fig_prompt_precision()
    print("wrote fig_budget_realistic.pdf, fig_router.pdf, fig_prompt_precision.pdf")


if __name__ == "__main__":
    main()
