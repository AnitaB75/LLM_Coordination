#!/usr/bin/env python3
"""
Prediction-study figure matched to the aesthetic of the 28-round game figure
that follows the "Figure 3. XYZ" caption in the workshop draft.

Replicated elements: centred small plot title, percentage y-axis (0-100%),
blue "Human framing" / red "Other-LLM framing" series with circle/square
markers, light grey gridlines, thin grey panel border, small framed legend in
the upper right, grey dashed reference line with an italic grey label.

Left  : predicted % of the 5 teammates who will switch at round 8
Right : the agent's own round-8 choice (% choosing M or N)
x-axis: game length, 14-round (the published game) vs 28-round.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcdefaults()

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

BLUE, RED = "#1f77b4", "#d62728"
GRID, BORDER, GREY = "#DCDCDC", "#AFAFAF", "#808080"
SERIES = [("human", "Human framing", BLUE, "o"),
          ("llm", "Other-LLM framing", RED, "s")]

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


def style(ax, ylabel):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(BORDER); s.set_linewidth(0.8)
    ax.tick_params(length=2.5, labelsize=6.5, color=BORDER)
    ax.set_xlabel("Game length (rounds)", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xticks([14, 28]); ax.set_xlim(11.5, 30.5)
    ax.set_ylim(-0.02, 1.0)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def main():
    path = sorted(RESULTS.glob("prediction_study_28rounds_2*.csv"))[-1]
    df = pd.read_csv(path)
    hs = [14, 28]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))

    for key, lab, col, mk in SERIES:
        pm, plo, phi = [], [], []
        om, olo, ohi = [], [], []
        for h in hs:
            s = df[(df.horizon == h) & (df.framing == key)]
            v = s.n_switch_MN / 5
            mu = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
            pm.append(mu); plo.append(max(0, mu-1.96*se)); phi.append(mu+1.96*se)
            k, n = int(s.own_switch.sum()), len(s)
            a, b = wilson(k, n)
            om.append(k/n); olo.append(a); ohi.append(b)
        axes[0].errorbar(hs, pm, yerr=[np.subtract(pm, plo), np.subtract(phi, pm)],
                         color=col, marker=mk, ms=4, lw=1.2, capsize=2.5,
                         elinewidth=0.8, label=lab, zorder=3)
        axes[1].errorbar(hs, om, yerr=[np.subtract(om, olo), np.subtract(ohi, om)],
                         color=col, marker=mk, ms=4, lw=1.2, capsize=2.5,
                         elinewidth=0.8, label=lab, zorder=3)

    style(axes[0], "% of teammates predicted to switch")
    style(axes[1], "% choosing M or N")

    for ax in axes:
        ax.axvline(14, color=GREY, lw=0.8, ls="--", zorder=1)
        ax.text(14.35, 0.965, "Published\ngame length", color=GREY, fontsize=6,
                style="italic", va="top", ha="left")
    axes[0].set_title("Predictions about teammates (round 8)", fontsize=7.5, pad=5)
    axes[1].set_title("Own intended switch (round 8)", fontsize=7.5, pad=5)
    axes[1].legend(loc="upper right", fontsize=6.5, framealpha=1,
                   edgecolor=BORDER, fancybox=False, borderpad=0.5)

    fig.suptitle("Rate of predicted and intended switching to M or N (GPT-5.5)",
                 fontsize=8.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    stamp = path.stem.replace("prediction_study_28rounds_", "")
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_prediction_matched_{stamp}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
