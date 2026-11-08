#!/usr/bin/env python3
"""
Per-agent letter choices by round, as color blocks, one panel per group.

Rows are the 6 agents in a group, columns are rounds 1-14, cell colour is the
letter chosen. The letter is also printed in each cell, so the figure carries
its meaning in greyscale and does not depend on the palette (house style).

Usage:
    python3 plot_raster_opus48.py                      # static_adaptive
    python3 plot_raster_opus48.py --condition reveal8_adaptive
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

plt.style.use("academic")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# order defines the integer codes used by the colormap
LETTERS = ["J", "M", "N", "Other"]
COLORS = ["#DDDDDD", "#4477AA", "#EE6677", "#FFFFFF"]
TITLES = {
    "static_adaptive":  "full schedule from round 1, adaptive thinking",
    "reveal8_adaptive": "M/N revealed at round 8, adaptive thinking",
    "static_nothink":   "full schedule from round 1, thinking disabled",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="static_adaptive", choices=list(TITLES))
    args = ap.parse_args()

    path = sorted(RESULTS.glob("name_game_opus48_llmonly_2*.csv"))[-1]
    df = pd.read_csv(path)
    df = df[df.condition == args.condition]
    if df.empty:
        raise SystemExit(f"no rows for condition {args.condition}")

    code = {l: i for i, l in enumerate(LETTERS)}
    df["code"] = df.letter.where(df.letter.isin(LETTERS), "Other").map(code)

    groups = sorted(df.group_key.unique(), key=lambda g: int(g.rsplit("G", 1)[1]))
    cmap = ListedColormap(COLORS)

    fig, axes = plt.subplots(
        len(groups), 1, figsize=(5.5, 5.2), sharex=True,
        gridspec_kw=dict(hspace=0.35),
    )

    for ax, g in zip(np.atleast_1d(axes), groups):
        sub = df[df.group_key == g]
        mat = (sub.pivot_table(index="actor_slot", columns="round", values="code")
                  .reindex(index=range(1, 7), columns=range(1, 15)))
        ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=len(LETTERS) - 1,
                  aspect="auto", interpolation="nearest")

        for (i, j), v in np.ndenumerate(mat.values):
            if not np.isnan(v):
                ax.text(j, i, LETTERS[int(v)], ha="center", va="center",
                        fontsize=4.6, color="black")

        # grid between cells
        ax.set_xticks(np.arange(-0.5, 14, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.6)
        ax.tick_params(which="minor", length=0)

        ax.axvline(6.5, color="black", lw=1.0)          # payoff change, R7|R8
        ax.set_yticks(range(6))
        ax.set_yticklabels(range(1, 7), fontsize=5)
        ax.set_ylabel(f"G{g.rsplit('G', 1)[1]}", rotation=0, labelpad=12, va="center")
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

    bottom = np.atleast_1d(axes)[-1]
    bottom.set_xticks(range(14))
    bottom.set_xticklabels(range(1, 15))
    bottom.set_xlabel("Round")

    present = [l for l in LETTERS if l in set(df.letter.where(df.letter.isin(LETTERS), "Other"))]
    np.atleast_1d(axes)[0].legend(
        handles=[Patch(facecolor=COLORS[code[l]], edgecolor="black", lw=0.4, label=l)
                 for l in present],
        loc="lower left", bbox_to_anchor=(0, 1.15), ncol=len(present),
        frameon=False, fontsize=7, handlelength=1.1, handleheight=1.1,
    )
    fig.supylabel("Agent within group", fontsize=8, x=0.005)
    fig.tight_layout()

    stamp = path.stem.replace("name_game_opus48_llmonly_", "")
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_opus48_raster_{args.condition}_{stamp}.{ext}"
        fig.savefig(out, dpi=300)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
