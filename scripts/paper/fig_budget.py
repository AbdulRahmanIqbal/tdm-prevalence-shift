"""Stage 2 paper figures along the label budget (Exp 3 + Exp 4 condition b). Writes paper/figures/fig5_budget.pdf
(Macro-F1 vs Jev), fig_budget_ece.pdf (ECE vs Jev) and fig_qlora_base.pdf (QLoRA vs Qwen3.5-4B and Tev1-4B), each
with a *_preview.png for checking.

Inputs: results/final_budget_analysis/curves/*.json only. Colours follow the system in every paper figure
(validated categorical slots on white); series also differ by marker shape (secondary encoding).
The rightmost DeBERTa / TF-IDF point is the full training set, placed at its size.

Run: .venv/Scripts/python scripts/paper/fig_budget.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
CURVES = ROOT / "results" / "final_budget_analysis" / "curves"

DATASETS = ["hatexplain", "fpb", "trec6", "hwu64", "clinc150"]   # by number of classes, as in tab_crossover
DS_LABEL = {"clinc150": "CLINC150 (150)", "hwu64": "HWU64 (64)", "fpb": "FPB (3)", "hatexplain": "HateXplain (3)",
            "trec6": "TREC-6 (6)"}
# colours follow the system in every paper figure (validated categorical slots, light; see fig_calibration.py)
ORANGE, AQUA, YELLOW, VIOLET, RED = "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("deberta-v3-large", "DeBERTa-v3-large", ORANGE, "s"), ("tfidf-lr", "TF-IDF + LR", YELLOW, "D"),
          ("qwen3.5-4b-qlora", "Qwen3.5-4B + QLoRA", AQUA, "^")]

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def rows(system: str, dataset: str, metric: str = "macro_f1", ref: str = "jev@final_jev") -> tuple[list[float], list[dict]]:
    d = json.loads((CURVES / f"{system}_vs_{ref}_{dataset}.json").read_text(encoding="utf-8"))
    rs = d["curves"][metric]["rows"]
    return [float(d["sizes"][str(r["budget"])]) for r in rs], rs


def budget_panels(series, metric: str, ylabel: str, name: str, zero_label: str, xmax: float = 25000,
                  crossover: bool = False) -> None:
    """2 x 3 grid: one panel per dataset (system - reference along the budget, 95% bootstrap band, dashed zero line
    labelled with the reference), legend in the sixth cell. crossover=True rings the crossover budget of each system
    (smallest budget whose interval lies above zero, read from the curve file)."""
    from matplotlib.lines import Line2D
    fig, grid = plt.subplots(2, 3, figsize=(6.6, 3.9))
    axes = grid.ravel()
    for ax, ds in zip(axes, DATASETS):
        for system, ref, label, color, marker in series:
            x, rs = rows(system, ds, metric, ref)
            y = [100 * r["estimate"] for r in rs]
            ax.fill_between(x, [100 * r["ci_low"] for r in rs], [100 * r["ci_high"] for r in rs],
                            color=color, alpha=0.15, lw=0)
            ax.plot(x, y, color=color, lw=1.4, marker=marker, ms=4.2, mfc=color, mec="white", mew=0.6,
                    label=label, zorder=3)
            if crossover:
                c = curve_meta(system, ds, ref)["curves"][metric]["crossover"]["crossover"]
                if c != "not reached":
                    k = [r["budget"] for r in rs].index(c)
                    ax.plot(x[k], y[k], marker="o", ms=9, mfc="none", mec=color, mew=1.1, zorder=4)
        ax.axhline(0, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.text(1.0, 0, f" {zero_label}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=7, color=INK2, clip_on=False)
        ax.set_xscale("log")
        ticks = [t for t in (100, 1000, 10000) if t <= xmax]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([{100: "100", 1000: "1k", 10000: "10k"}[t] for t in ticks])
        ax.set_xlim(70, xmax)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_title(DS_LABEL[ds], loc="left")
    for ax in grid[:, 0]:
        ax.set_ylabel(ylabel)
    for ax in axes[2:5]:
        ax.set_xlabel("Training labels (log scale)")
    leg = axes[5]
    leg.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if crossover:
        handles.append(Line2D([], [], ls="", marker="o", ms=9, mfc="none", mec=INK2, mew=1.1))
        labels.append("crossover: first budget\nwith interval above 0")
    handles.append(Line2D([], [], color=INK2, lw=0.7, ls=(0, (3, 2))))
    labels.append("no difference" + (" (Jev)" if "Jev" in zero_label else ""))
    leg.legend(handles, labels, loc="center left", frameon=False, fontsize=7.8, handlelength=2.2,
               labelspacing=0.9, borderaxespad=0)
    fig.tight_layout(w_pad=1.6, h_pad=1.0)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}_preview.png", dpi=200)
    plt.close(fig)


def curve_meta(system: str, dataset: str, ref: str) -> dict:
    return json.loads((CURVES / f"{system}_vs_{ref}_{dataset}.json").read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vs_jev = [(s, "jev@final_jev", lab, c, m) for s, lab, c, m in SERIES]
    budget_panels(vs_jev, "macro_f1", "$\\Delta$ Macro-F1 vs Jev (points)", "fig5_budget", "= Jev", crossover=True)
    budget_panels(vs_jev, "ece", "$\\Delta$ ECE vs Jev (points)", "fig_budget_ece", "= Jev")
    budget_panels([("qwen3.5-4b-qlora", "qwen3.5-4b@final_local", "QLoRA $-$ Qwen3.5-4B\n(frozen base)", RED, "o"),
                   ("qwen3.5-4b-qlora", "tev1-4b@final_local", "QLoRA $-$ Tev1-4B\n(vendor training)", VIOLET, "^")],
                  "macro_f1", "$\\Delta$ Macro-F1 (points)", "fig_qlora_base", "= 0", xmax=4000)
    print("wrote fig5_budget.pdf, fig_budget_ece.pdf, fig_qlora_base.pdf")


if __name__ == "__main__":
    main()
