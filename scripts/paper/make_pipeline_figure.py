"""Methodology overview figure (paper Fig. 2): evaluation pipeline in four stages.

Writes paper/figures/fig_pipeline.pdf (vector, for the paper) and fig_pipeline_preview.png (for checking).
Run: .venv/Scripts/python scripts/paper/make_pipeline_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SLATE, GREY = "#3d4a5c", "#6b6a66"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 6, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans", "text.color": INK})

W, H = 6.6, 3.55
PW, GAP = 1.50, 0.20
TOP, BOT, HEAD = 3.50, 0.50, 0.28
PANELS = [("Inputs", GREY), ("Systems", BLUE), ("Shift & recalibration", ORANGE), ("Evaluation", SLATE)]
F_SUB, F_TXT, F_CHIP, F_SMALL = 6.3, 5.9, 5.8, 5.4


def tint(hex_color: str, a: float) -> tuple:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def panel(ax, i: int, title: str, color: str) -> float:
    x = i * (PW + GAP)
    ax.add_patch(FancyBboxPatch((x, BOT), PW, TOP - BOT, boxstyle="round,pad=0,rounding_size=0.07",
                                fc=tint(color, 0.06), ec=tint(color, 0.55), lw=0.8))
    ax.add_patch(FancyBboxPatch((x, TOP - HEAD), PW, HEAD, boxstyle="round,pad=0,rounding_size=0.07",
                                fc=color, ec=color, lw=0.8))
    ax.add_patch(Rectangle((x, TOP - HEAD), PW, HEAD / 2, fc=color, ec=color, lw=0))
    ax.add_patch(Circle((x + 0.15, TOP - HEAD / 2), 0.085, fc="white", ec="none"))
    ax.text(x + 0.15, TOP - HEAD / 2, str(i + 1), ha="center", va="center", fontsize=6.8, weight="bold", color=color)
    ax.text(x + 0.28, TOP - HEAD / 2, title, ha="left", va="center", fontsize=6.7, weight="bold", color="white")
    return x


def sub(ax, x, y, text):
    ax.text(x + 0.09, y, text, ha="left", va="top", fontsize=F_SUB, weight="bold", color=INK)


def bullet(ax, x, y, text, color):
    ax.add_patch(Circle((x + 0.13, y - 0.04), 0.016, fc=color, ec="none"))
    ax.text(x + 0.19, y, text, ha="left", va="top", fontsize=F_TXT, color=INK2, linespacing=1.25)


def chip(ax, x, y, w, text, color, h=0.16, bold=False):
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.045",
                                fc=tint(color, 0.16), ec=tint(color, 0.7), lw=0.6))
    ax.text(x + w / 2, y - h / 2, text, ha="center", va="center", fontsize=F_CHIP, color=INK,
            weight="bold" if bold else "normal")


def main() -> None:
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    xs = [panel(ax, i, t, c) for i, (t, c) in enumerate(PANELS)]
    y0 = TOP - HEAD - 0.10

    # 1 Inputs
    x, y = xs[0], y0
    sub(ax, x, y, "Five public datasets")
    for j, (name, task) in enumerate([("CLINC150", "intent"), ("HWU64", "intent"), ("FPB", "fin. sentiment"),
                                      ("HateXplain", "moderation"), ("TREC-6", "question type")]):
        yy = y - 0.18 - j * 0.20
        chip(ax, x + 0.09, yy, 0.62, name, GREY)
        ax.text(x + 0.78, yy - 0.08, task, ha="left", va="center", fontsize=F_SMALL + 0.2, color=INK2)
    y -= 1.30
    sub(ax, x, y, "Frozen label texts")
    bullet(ax, x, y - 0.15, "name and one-sentence\ndescription per label", GREY)
    y -= 0.58
    sub(ax, x, y, "Candidate sets $C(x)$")
    bullet(ax, x, y - 0.15, "fixed label groups ($K$=18-22)\nor all labels; same items,\noptions and order for\nevery system", GREY)

    # 2 Systems
    x, y = xs[1], y0
    sub(ax, x, y, "Zero-shot TDM (hosted API)")
    chip(ax, x + 0.09, y - 0.16, PW - 0.18, "Jev", BLUE, bold=True)
    y -= 0.46
    sub(ax, x, y, "Decision-training pairs")
    ax.text(x + 0.09, y - 0.13, "same prompt, readout and precision", ha="left", va="top", fontsize=F_SMALL,
            color=INK2, style="italic")
    for k, (tdm, base) in enumerate([("Tev1-4B", "Qwen3.5-4B"), ("Nimble-9B", "Qwen3.5-9B")]):
        yy = y - 0.32 - k * 0.23
        chip(ax, x + 0.09, yy, 0.60, tdm, BLUE, bold=True)
        chip(ax, x + 0.81, yy, 0.60, base, GREY)
        ax.annotate("", xy=(x + 0.80, yy - 0.08), xytext=(x + 0.70, yy - 0.08),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
    y -= 0.86
    sub(ax, x, y, "Trained on task labels")
    chip(ax, x + 0.09, y - 0.16, 0.78, "DeBERTa-v3-large", GREY)
    chip(ax, x + 0.92, y - 0.16, 0.49, "TF-IDF+LR", GREY)
    ax.text(x + 0.09, y - 0.395, "full data; 100 to 5,000 labels, 5 seeds", ha="left", va="center",
            fontsize=F_SMALL, color=INK2, style="italic")
    chip(ax, x + 0.09, y - 0.50, PW - 0.18, "Qwen3.5-4B + QLoRA", GREY)
    ax.text(x + 0.09, y - 0.765, "100, 500, 2,500 labels, 3 seeds", ha="left", va="center",
            fontsize=F_SMALL, color=INK2, style="italic")
    y -= 0.88
    ax.add_patch(FancyBboxPatch((x + 0.09, y - 0.30), PW - 0.18, 0.30, boxstyle="round,pad=0,rounding_size=0.045",
                                fc="white", ec=tint(BLUE, 0.6), lw=0.6, ls=(0, (2, 1.5))))
    ax.text(x + PW / 2, y - 0.15, "output: $p(\\cdot\\mid x)$ over $C(x)$\non dev and test", ha="center",
            va="center", fontsize=F_CHIP, color=INK)

    # 3 Shift & recalibration
    x, y = xs[2], y0
    sub(ax, x, y, "Simulated prevalence shift")
    bx, by, bw = x + 0.22, y - 0.56, 0.11
    src, tgt = [0.30, 0.45, 0.25], [0.12, 0.23, 0.65]
    for k in range(3):
        ax.add_patch(Rectangle((bx + k * 0.38, by), bw, src[k] * 0.55, fc=tint(GREY, 0.5), ec="none"))
        ax.add_patch(Rectangle((bx + k * 0.38 + bw + 0.02, by), bw, tgt[k] * 0.55, fc=ORANGE, ec="none"))
    ax.plot([bx - 0.06, bx + 1.04], [by, by], color=INK2, lw=0.5)
    ax.text(bx - 0.06, by - 0.04, "source $\\pi_s$", ha="left", va="top", fontsize=F_SMALL, color=GREY)
    ax.text(bx + 1.04, by - 0.04, "target $q$", ha="right", va="top", fontsize=F_SMALL, color=ORANGE)
    y -= 0.78
    bullet(ax, x, y, "$q\\sim\\mathrm{Dirichlet}(\\alpha L\\pi_s)$,\n$\\alpha\\in\\{10,1,0.1\\}$", ORANGE)
    bullet(ax, x, y - 0.32, "resample test set class by\nclass; resampled no-shift\ncontrol", ORANGE)
    y -= 0.82
    sub(ax, x, y, "Recalibration (on dev)")
    arms = [("none", "raw"), ("temperature", "labelled"), ("EM with $\\hat\\pi_s$", "label-free"),
            ("BCTS + EM", "labelled"), ("oracle, true $q$", "reference")]
    for k, (arm, kind) in enumerate(arms):
        yy = y - 0.17 - k * 0.14
        ax.text(x + 0.13, yy, arm, ha="left", va="top", fontsize=F_TXT, color=INK,
                weight="bold" if "EM with" in arm else "normal")
        ax.text(x + PW - 0.09, yy, kind, ha="right", va="top", fontsize=F_SMALL - 0.1, color=GREY, style="italic")

    # 4 Evaluation
    x, y = xs[3], y0
    sub(ax, x, y, "Metrics")
    bullet(ax, x, y - 0.15, "Macro-F1, top-label ECE", SLATE)
    bullet(ax, x, y - 0.29, "excess risk, 80% coverage", SLATE)
    bullet(ax, x, y - 0.43, "Brier, NLL, AURC", SLATE)
    y -= 0.70
    sub(ax, x, y, "Confirmatory tests ($\\alpha$=1)")
    for k, (rq, test, what, col) in enumerate([("RQ1", "P1, P2", "TDM vs own base", BLUE),
                                               ("RQ2", "PA1, PA2", "Jev vs DeBERTa", ORANGE),
                                               ("RQ3", "PA3", "Jev: EM vs none", ORANGE)]):
        yy = y - 0.17 - k * 0.27
        chip(ax, x + 0.09, yy, 0.34, rq, col, h=0.2, bold=True)
        ax.text(x + 0.50, yy - 0.015, test, ha="left", va="top", fontsize=F_TXT, weight="bold", color=INK)
        ax.text(x + 0.50, yy - 0.12, what, ha="left", va="top", fontsize=F_SMALL, color=INK2)
    y -= 0.98
    bullet(ax, x, y, "35 tests, paired bootstrap\n($B$=10,000), Holm-adjusted", SLATE)
    y -= 0.42
    sub(ax, x, y, "Estimated, not tested")
    chip(ax, x + 0.09, y - 0.17, 0.34, "RQ4", AQUA, h=0.2, bold=True)
    ax.text(x + 0.50, y - 0.185, "crossover budget", ha="left", va="top", fontsize=F_TXT, weight="bold", color=INK)
    ax.text(x + 0.50, y - 0.295, "trained vs Jev, by labels", ha="left", va="top", fontsize=F_SMALL, color=INK2)

    # arrows between stages
    ymid = (TOP + BOT) / 2 + 0.25
    for i in range(3):
        x0, x1 = xs[i] + PW + 0.025, xs[i + 1] - 0.025
        ax.add_patch(FancyArrowPatch((x0, ymid), (x1, ymid),
                                     arrowstyle="simple,head_width=7,head_length=5,tail_width=2.4",
                                     fc=tint(INK2, 0.75), ec="none", mutation_scale=1))

    # bottom strip
    ax.add_patch(FancyBboxPatch((0, 0.05), W, 0.34, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="#f4f3ef", ec="#dcdad4", lw=0.6))
    ax.text(0.12, 0.22, "Fitted on the development set:", ha="left", va="center", fontsize=F_TXT, weight="bold")
    ax.text(1.62, 0.22, "thresholds $\\tau$; temperature, BCTS (labels); implicit prior $\\hat\\pi_s$ (no labels); checkpoints",
            ha="left", va="center", fontsize=F_TXT, color=INK2)
    ax.text(5.14, 0.22, "Test set:", ha="left", va="center", fontsize=F_TXT, weight="bold")
    ax.text(5.60, 0.22, "metrics, tests, curves", ha="left", va="center", fontsize=F_TXT, color=INK2)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_pipeline.pdf")
    fig.savefig(OUT / "fig_pipeline_preview.png", dpi=220)
    print("wrote", OUT / "fig_pipeline.pdf")


if __name__ == "__main__":
    main()
