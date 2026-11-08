#!/usr/bin/env python3
"""
Prediction-study figure in the format of the paper's FIGURE 3.

Layout copied from that figure:
  Panel A  "Predictions about teammates (round 8)" -- boxplots, x = letter
           (M, N, J), y = % of the 5 teammates predicted to choose it.
  Panel B  "Own intended switch (round 8)" -- bars with error bars,
           x = letter chosen, y = % of participants.
  Facets across columns, grey strip headers, blue/orange by teammate type,
  shared legend beneath.

The paper's facets are the three prediction studies (Human / GPT-5.5 /
Claude 4.8). This rerun is GPT-5.5 only, so the facets are the two game
lengths instead: 14-round (the published game) and 28-round.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mpl.rcdefaults()

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

BLUE, ORANGE = "#4C72B0", "#DD8452"
GRID, BORDER, STRIP = "#E3E3E3", "#B4B4B4", "#F0F0F0"
LETTERS = ["M", "N", "J"]
# facets = what is being predicted about; series = game length
FACETS = [("human", "Human Prediction\n(teammates described as humans)"),
          ("llm", "LLM Prediction\n(teammates described as LLMs)")]
HORIZON = [(14, "14-round game", BLUE, -0.19),
           (28, "28-round game", ORANGE, +0.19)]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
})


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return max(0.0, c-h), min(1.0, c+h)


def frame(ax, ylab=None, xlab=None):
    ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(BORDER); s.set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=6.5)
    ax.set_xticks(range(len(LETTERS))); ax.set_xticklabels(LETTERS, fontsize=7)
    ax.set_xlim(-0.6, len(LETTERS) - 0.4)
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    if ylab:
        ax.set_ylabel(ylab, fontsize=6.5)
    if xlab:
        ax.set_xlabel(xlab, fontsize=7)


def main():
    path = sorted(RESULTS.glob("prediction_study_28rounds_2*.csv"))[-1]
    df = pd.read_csv(path)
    colmap = {"M": "n_switch_M", "N": "n_switch_N", "J": "n_stay_J"}

    fig, axes = plt.subplots(2, len(FACETS), figsize=(6.8, 4.7))

    for col, (fkey, flab) in enumerate(FACETS):
        axA, axB = axes[0][col], axes[1][col]

        for h, lab, colr, off in HORIZON:
            s = df[(df.horizon == h) & (df.framing == fkey)]

            # --- A: distribution of predicted teammate counts, as proportion of 5
            data = [(s[colmap[L]] / 5).values for L in LETTERS]
            bp = axA.boxplot(data, positions=[i + off for i in range(len(LETTERS))],
                             widths=0.3, patch_artist=True, showmeans=True,
                             meanprops=dict(marker="D", markerfacecolor="white",
                                            markeredgecolor=colr, markersize=3),
                             medianprops=dict(color=colr, lw=0.9),
                             boxprops=dict(facecolor="white", edgecolor=colr, lw=0.9),
                             whiskerprops=dict(color=colr, lw=0.9),
                             capprops=dict(color=colr, lw=0.9),
                             flierprops=dict(marker="o", ms=2, mfc="none",
                                             mec=colr, alpha=0.6))
            for e in bp["boxes"]:
                e.set_zorder(3)

            # --- B: proportion of participants choosing each letter
            n = len(s)
            m = [(s.own_choice == L).sum() / n for L in LETTERS]
            lo, hi = zip(*[wilson(int((s.own_choice == L).sum()), n) for L in LETTERS])
            axB.bar([i + off for i in range(len(LETTERS))], m, 0.3,
                    color=colr, zorder=3, label=lab)
            axB.errorbar([i + off for i in range(len(LETTERS))], m,
                         yerr=[np.subtract(m, lo), np.subtract(hi, m)],
                         fmt="none", ecolor=colr, elinewidth=0.8, capsize=2, zorder=4)

        frame(axA, "% of teammates predicted to switch" if col == 0 else None)
        frame(axB, "% of participants" if col == 0 else None, "Letter chosen")
        axA.set_xticklabels(LETTERS, fontsize=7)

        for ax in (axA, axB):
            ax.text(0.5, 1.02, flab, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7,
                    bbox=dict(facecolor=STRIP, edgecolor=BORDER, lw=0.8,
                              boxstyle="square,pad=0.3"))

    fig.text(0.012, 0.965, "A  Predictions about teammates (round 8)",
             fontsize=7.5, weight="bold")
    fig.text(0.012, 0.475, "B  Own intended switch (round 8)",
             fontsize=7.5, weight="bold")

    fig.legend(handles=[
        Line2D([], [], color=BLUE, marker="s", mfc="white", ms=5, ls="none",
               label="14-round game"),
        Line2D([], [], color=ORANGE, marker="s", mfc="white", ms=5, ls="none",
               label="28-round game"),
        Patch(facecolor=BLUE, label="14-round game"),
        Patch(facecolor=ORANGE, label="28-round game")],
        loc="lower center", ncol=4, frameon=False, fontsize=6.5,
        bbox_to_anchor=(0.5, -0.012), title="Game length    (A: boxes    B: bars)",
        title_fontsize=6.5)

    fig.tight_layout(rect=[0.01, 0.055, 1, 0.945])
    fig.subplots_adjust(hspace=0.55)

    stamp = path.stem.replace("prediction_study_28rounds_", "")
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_prediction_facets_{stamp}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
