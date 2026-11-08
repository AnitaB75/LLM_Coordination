#!/usr/bin/env python3
"""
Prediction-study figure in the aesthetic of the paper's FIGURE 3
(the "Bayesian team-plan models: observed vs simulated trajectories" panel).

Replicated elements: two side-by-side facets with plain centred facet titles,
a small centred supertitle, light grey gridlines on both axes, thin grey panel
borders, coloured marker+line series, and a small framed legend inside the
right-hand panel.

Left facet : predicted number of the 5 teammates switching, as a proportion
Right facet: the agent's own round-8 choice
x-axis     : game length (14-round control vs 28-round manipulation)
series     : how the teammates were described (human vs LLM)

Error bars (95% CI) are included even though Figure 3 has none: the horizon
effect here is small and not statistically reliable, and showing the point
estimates bare would overstate it.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcdefaults()

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

BLUE, ORANGE = "#1f77b4", "#ff7f0e"
GRID, BORDER = "#D9D9D9", "#B0B0B0"
SERIES = [("human", "Human teammates", BLUE, "o", "-"),
          ("llm", "LLM teammates", ORANGE, "s", "--")]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
})


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p, d = k / n, 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return max(0, c-h), min(1, c+h)


def style(ax, title):
    ax.set_title(title, fontsize=8, pad=6)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(BORDER); s.set_linewidth(0.8)
    ax.tick_params(length=2.5, labelsize=6.5, color=BORDER)
    ax.set_xlabel("Game length (rounds)", fontsize=7)
    ax.set_xticks([14, 28]); ax.set_xlim(11, 31)
    ax.set_ylim(-0.02, 1.0)
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])


def main():
    path = sorted(RESULTS.glob("prediction_study_28rounds_2*.csv"))[-1]
    df = pd.read_csv(path)
    horizons = [14, 28]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)

    for key, lab, col, mk, ls in SERIES:
        m, lo, hi = [], [], []
        for h in horizons:
            s = df[(df.horizon == h) & (df.framing == key)].n_switch_MN / 5
            mu = s.mean()
            se = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0
            m.append(mu); lo.append(max(0, mu - 1.96*se)); hi.append(mu + 1.96*se)
        axes[0].errorbar(horizons, m,
                         yerr=[np.subtract(m, lo), np.subtract(hi, m)],
                         color=col, marker=mk, ls=ls, lw=1.2, ms=4,
                         capsize=2.5, elinewidth=0.8, label=lab, zorder=3)

        m, lo, hi = [], [], []
        for h in horizons:
            s = df[(df.horizon == h) & (df.framing == key)]
            k, n = int(s.own_switch.sum()), len(s)
            a, b = wilson(k, n)
            m.append(k/n); lo.append(a); hi.append(b)
        axes[1].errorbar(horizons, m,
                         yerr=[np.subtract(m, lo), np.subtract(hi, m)],
                         color=col, marker=mk, ls=ls, lw=1.2, ms=4,
                         capsize=2.5, elinewidth=0.8, label=lab, zorder=3)

    style(axes[0], "Predicted teammates")
    style(axes[1], "Own choice")
    axes[0].set_ylabel("Proportion switching to M or N", fontsize=7)
    axes[1].legend(loc="upper right", fontsize=6.5, framealpha=1,
                   edgecolor=BORDER, fancybox=False, borderpad=0.5)

    fig.suptitle("Prediction study: expected and own switching at Round 8 (GPT-5.5)",
                 fontsize=8.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    stamp = path.stem.replace("prediction_study_28rounds_", "")
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_prediction_fig3style_{stamp}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
