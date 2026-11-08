"""
Generate Name Game stimulus image sets at multiple M/N reward levels (15/20/50).

Reuses the drawing primitives from generate_stimuli.py; only the reward-dependent
pages are redefined to take a `reward` argument:
  - the M/N announcement       (04_new_letters.png)        -> headline + chip values
  - the round-8 result boards   (05-08_round8_result_*M)    -> payoff legend + label

Each reward level gets a full, self-contained image folder so the Claude pilot can
point at exactly one folder per cell:
  stimuli/reward_15/ , stimuli/reward_20/ , stimuli/reward_50/

Run:  python3 generate_stimuli_rewards.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import generate_stimuli as g   # reuse primitives, constants, reward-independent pages

REWARDS = [15, 20, 50]
BASE = os.path.dirname(os.path.abspath(__file__))


def draw_payoff_legend(ax, y, reward):
    """Bottom legend with M/N at the given reward (mirrors g._draw_payoff_legend)."""
    vals = {ch: 0 for ch in g.LETTERS_DISPLAY}
    vals.update({"J": 10, "M": reward, "N": reward})
    ax.text(7, y + 1.18, "What each letter is worth (points):", ha="center",
            va="center", fontsize=13, fontweight="bold", color=g.SUBINK)
    cw, ch_, gap = 1.18, 0.9, 0.12
    x0 = (14 - (10 * cw + 9 * gap)) / 2
    for i, ch in enumerate(g.LETTERS_DISPLAY):
        x = x0 + i * (cw + gap)
        color = g.lcolor(ch); v = vals[ch]
        ax.add_patch(FancyBboxPatch(
            (x, y), cw, ch_, boxstyle="round,pad=0.01,rounding_size=0.06",
            linewidth=1.6, edgecolor=color, facecolor="white"))
        ax.text(x + cw/2, y + ch_ * 0.62, ch, ha="center", va="center",
                fontsize=16, fontweight="bold", color=color)
        ax.text(x + cw/2, y + ch_ * 0.24, str(v), ha="center", va="center",
                fontsize=11, fontweight="bold" if v > 0 else "normal",
                color=g.INK if v > 0 else g.SUBINK)


def draw_announcement(reward, outpath):
    values = {ch: 0 for ch in g.LETTERS_DISPLAY}
    values.update({"J": 10, "M": reward, "N": reward})
    fig, ax = plt.subplots(figsize=(14, 8.2), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.2); ax.axis("off")
    ax.text(7, 7.6, "New for Round 8", ha="center", va="center",
            fontsize=33, fontweight="bold", color=g.R8_EDGE)
    ax.text(7, 7.05, f"M and N are now worth {reward} points each.",
            ha="center", va="center", fontsize=17, color=g.INK)
    g._draw_letter_board(ax, top_y=5.0, show_values=True, values=values,
                         highlight_new=("M", "N"))
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", outpath)


def draw_result(n_M, reward, outpath):
    mat = [["J"] * 8 + ["?"] for _ in range(g.N_PLAYERS)]
    for p in range(g.N_PLAYERS - n_M, g.N_PLAYERS):
        mat[p][7] = "M"
    fig, ax = plt.subplots(figsize=(14, 8.3), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.3); ax.axis("off")
    ax.text(7, 7.75, "Rounds 1–8 Results — What Will Happen in Round 9?",
            ha="center", va="center", fontsize=25, fontweight="bold", color=g.INK)
    g._draw_grid(ax, mat, top=6.45, highlight_col=7, predict_col=8,
                 divider_after_col=6, divider_label=f"M & N now worth {reward}")
    draw_payoff_legend(ax, y=0.55, reward=reward)
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", outpath)


def main():
    for reward in REWARDS:
        out = os.path.join(BASE, "stimuli", f"reward_{reward}")
        os.makedirs(out, exist_ok=True)
        # reward-independent shared pages -> regenerate into each folder (self-contained)
        g.draw_how_to_play(os.path.join(out, "00_how_to_play.png"))
        g.draw_how_points(os.path.join(out, "01_how_points.png"))
        g.draw_history(os.path.join(out, "03_history_rounds_1-7.png"))
        # reward-dependent pages
        draw_announcement(reward, os.path.join(out, "04_new_letters.png"))
        for i, n_M in enumerate((0, 1, 2, 3), start=5):
            draw_result(n_M, reward, os.path.join(out, f"0{i}_round8_result_{n_M}M.png"))


if __name__ == "__main__":
    main()
