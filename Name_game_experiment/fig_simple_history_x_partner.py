#!/usr/bin/env python3
"""
Simple interaction plot for the 2 (history) x 4 (partner framing) name game.

One panel per model config; x is the history condition, one line per partner
framing. A flat line means history did not matter for that framing; a steep one
means it did. The self-vs-human spread at each x is the primary contrast, so
the whole prediction is readable off the slopes.

Writes PNG + PDF to ~/Downloads.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("academic")

HERE = Path(__file__).resolve().parent
SRC = HERE / "history_x_partner_output" / "primary_first_available_round.csv"
OUT = Path.home() / "Downloads"

HISTORIES = ["history", "no_history"]
XLAB = ["J-only\nrounds 1-7", "No J-only\nphase"]
PARTNERS = ["self", "human", "other_llm", "neutral"]
STYLE = {
    "self":      dict(label="Self",      marker="o", mfc="black", ls="-",   color="black"),
    "human":     dict(label="Human",     marker="s", mfc="white", ls="--",  color="black"),
    "other_llm": dict(label="Other LLM", marker="^", mfc="0.55",  ls="-.",  color="0.35"),
    "neutral":   dict(label="Neutral",   marker="D", mfc="white", ls=":",   color="0.55"),
}

tab = pd.read_csv(SRC)
configs = sorted(tab["config"].unique())

fig, axes = plt.subplots(1, len(configs), sharey=True,
                         figsize=(5.5, 2.5), squeeze=False)
axes = axes[0]
x = np.arange(len(HISTORIES))

for ax, config in zip(axes, configs):
    sub = tab[tab["config"] == config]
    for partner in PARTNERS:
        s = (sub[sub["partner_condition"] == partner]
             .set_index("history_condition").reindex(HISTORIES))
        st = STYLE[partner]
        lo = np.clip(s["p_mn"] - s["ci_lo"], 0, None)
        hi = np.clip(s["ci_hi"] - s["p_mn"], 0, None)
        ax.errorbar(x, s["p_mn"], yerr=[lo, hi],
                    marker=st["marker"], mfc=st["mfc"], mec=st["color"],
                    color=st["color"], ls=st["ls"], ms=4, lw=0.9,
                    capsize=2, capthick=0.7, elinewidth=0.7, label=st["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(XLAB, fontsize=7.5)
    ax.set_xlim(-0.35, len(HISTORIES) - 0.65)
    ax.set_ylim(-0.04, 1.04)
    ax.text(0.5, 1.03, config, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8)

axes[0].set_ylabel("P(chose M or N)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=7.5, ncol=4,
           loc="lower center", bbox_to_anchor=(0.5, -0.10))
fig.tight_layout()

OUT.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    f = OUT / f"name_game_history_x_partner_simple.{ext}"
    fig.savefig(f, dpi=300, bbox_inches="tight")
    print(f"-> {f}")
