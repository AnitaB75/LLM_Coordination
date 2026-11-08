#!/usr/bin/env python3
"""
Prediction-study figure in the PAPER's aesthetic (matches Figure 2 of the
NeurIPS workshop draft), not the house academic style.

Replicated elements: faceted panels with grey strip labels, thin grey panel
border, light horizontal gridlines, percentage y-axis, stacked bars with in-bar
percentage labels, centred plot title, horizontal legend beneath the panels,
green = switched to M or N, blue = stayed on J.

Rows   : predicted teammates' choices | the agent's own round-8 choice
Columns: 14-round game (control) | 28-round game (manipulation)
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

mpl.rcdefaults()  # deliberately NOT the academic style

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

GREEN, BLUE = "#34A853", "#4285F4"
GRID, BORDER, STRIP = "#E5E5E5", "#BDBDBD", "#F0F0F0"
FRAMINGS = [("human", "Human\nteammates"), ("llm", "LLM\nteammates")]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
})


def panel(ax, vals, title_left=None):
    """vals: list of (xlabel, pct_mn) — stacked bar, M/N on top of J."""
    xs = range(len(vals))
    for x, (lab, mn) in zip(xs, vals):
        j = 1 - mn
        ax.bar(x, j, 0.46, color=BLUE, zorder=3)
        ax.bar(x, mn, 0.46, bottom=j, color=GREEN, zorder=3)
        if j > 0.06:
            ax.text(x, j / 2, f"{j:.0%}", ha="center", va="center",
                    color="white", fontsize=7, zorder=4)
        if mn > 0.06:
            ax.text(x, j + mn / 2, f"{mn:.0%}", ha="center", va="center",
                    color="white", fontsize=7, zorder=4)
        elif mn > 0:
            ax.text(x, 1.015, f"{mn:.0%}", ha="center", va="bottom",
                    color=GREEN, fontsize=6.5, zorder=4)
    ax.set_xlim(-0.62, len(vals) - 0.38)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([v[0] for v in vals], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=6.5)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(BORDER); s.set_linewidth(0.8)
    ax.tick_params(length=0)
    if title_left:
        ax.set_ylabel(title_left, fontsize=7)


def main():
    path = sorted(RESULTS.glob("prediction_study_28rounds_2*.csv"))[-1]
    df = pd.read_csv(path)

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharey=True)

    for col, h in enumerate([14, 28]):
        pred = [(lab, df[(df.horizon == h) & (df.framing == f)].n_switch_MN.mean() / 5)
                for f, lab in FRAMINGS]
        own = [(lab, df[(df.horizon == h) & (df.framing == f)].own_switch.mean())
               for f, lab in FRAMINGS]
        panel(axes[0][col], pred,
              "% of 5 teammates predicted\nto switch" if col == 0 else None)
        panel(axes[1][col], own,
              "% choosing M or N\nthemselves" if col == 0 else None)

        # facet strip label across the top
        ax = axes[0][col]
        ax.text(0.5, 1.06, f"{h}-round game" + ("  (control)" if h == 14 else ""),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
                bbox=dict(facecolor=STRIP, edgecolor=BORDER, lw=0.8,
                          boxstyle="square,pad=0.35"))

    axes[0][0].text(-0.34, 0.5, "A", transform=axes[0][0].transAxes,
                    fontsize=10, va="center")
    axes[1][0].text(-0.34, 0.5, "B", transform=axes[1][0].transAxes,
                    fontsize=10, va="center")

    fig.suptitle("Round-8 predictions and choices in the prediction study (GPT-5.5)",
                 fontsize=8.5, y=0.985)
    fig.legend(handles=[Patch(facecolor=GREEN, label="Switched to M or N"),
                        Patch(facecolor=BLUE, label="Stayed on J")],
               loc="lower center", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0.02, 0.05, 1, 0.95])
    fig.subplots_adjust(hspace=0.42)

    stamp = path.stem.replace("prediction_study_28rounds_", "")
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_prediction_paperstyle_{stamp}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
