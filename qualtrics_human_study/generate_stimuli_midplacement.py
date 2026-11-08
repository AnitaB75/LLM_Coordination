"""
Salience-control variants of the two intro slides for the Name Game study.

Motivation: in the original stimuli, LETTERS_DISPLAY leads with J M N, so M and N
sit at the front of the array (leftmost / top-left) and M always precedes N. That
ordering is a candidate explanation for the M >> N choice asymmetry (M is chosen
over N despite equal value). These variants move M and N into the MIDDLE of the
letter array and break the M-before-N reading order, to de-confound salience.

Two slides only (mirrors 00_how_to_play.png / 01_how_points.png):

  00_how_to_play.png  single row of 10 letters -> M and N pushed to the center,
                      N left of M with a filler letter between them (no adjacent
                      M/N pair, M to the right of N).

  01_how_points.png   2 x 5 board -> N in the bottom row dead-center; M in the top
                      row one column up-and-to-the-right of N (a gentle diagonal,
                      not a salient vertical stack). J stays highlighted at 10.

  04_new_letters.png  round-8 announcement, same de-confounded board -> M and N are
                      highlighted (tint + NEW badge) IN their central/offset spots
                      rather than at the front of the array. Reward defaults to 15
                      (matches the base set); pass a reward for the 20/50 arms.

Everything else (fonts, colors, sizes, captions) is copied from generate_stimuli
so ONLY letter position changes.

Run:  python3 generate_stimuli_midplacement.py
Output -> stimuli/midplacement/{00_how_to_play,01_how_points,04_new_letters}.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

import generate_stimuli as g   # reuse constants, colors, styling

BASE = os.path.dirname(os.path.abspath(__file__))

# --- de-confounded letter positions --------------------------------------
# single-row order for how_to_play: M and N central, N before M, F between them
HOWTO_ROW = ["J", "P", "Q", "R", "N", "F", "M", "C", "X", "Y"]

# 2 x 5 board for how_points (row 0 = top, row 1 = bottom):
#   N bottom-center (col 2); M top row col 3 -> up-and-right of N.
POINTS_TOP = ["J", "P", "Q", "M", "R"]     # M at col 3
POINTS_BOTTOM = ["F", "C", "N", "X", "Y"]  # N at col 2


def draw_how_to_play(outpath):
    """Intro slide 1 with M/N moved to the middle of the single letter row."""
    fig, ax = plt.subplots(figsize=(14, 8.3), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.3); ax.axis("off")
    ax.text(7, 7.80, "How the Game Works", ha="center", va="center",
            fontsize=32, fontweight="bold", color=g.INK)
    ax.text(7, 7.34, "Step 1 of 2", ha="center", va="center", fontsize=13, color=g.SUBINK)

    rules = [
        "Six people play together — and they cannot talk to each other.",
        "Each round, all six players pick one letter at the same time.",
        "The players score points only when all six pick the SAME letter.",
        "After each round, the players see what everyone chose.",
    ]
    y = 6.45
    for i, line in enumerate(rules, 1):
        ax.add_patch(Circle((1.55, y), 0.27, facecolor=g.LETTER_COLORS["J"], zorder=2))
        ax.text(1.55, y, str(i), ha="center", va="center", fontsize=15,
                fontweight="bold", color="white", zorder=3)
        ax.text(2.15, y, line, ha="left", va="center", fontsize=16.5, color=g.INK)
        y -= 0.82

    ax.text(7, 3.20, "The letters players can choose:", ha="center", va="center",
            fontsize=15, fontweight="bold", color=g.SUBINK)
    cw, ch_, gap = 1.12, 1.15, 0.16
    x0 = (14 - (10 * cw + 9 * gap)) / 2
    yc = 1.55
    for i, letter in enumerate(HOWTO_ROW):
        x = x0 + i * (cw + gap)
        ax.add_patch(FancyBboxPatch((x, yc), cw, ch_, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.8, edgecolor=g.GRAY, facecolor="white"))
        ax.text(x + cw/2, yc + ch_/2, letter, ha="center", va="center",
                fontsize=22, fontweight="bold", color=g.GRAY)

    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", os.path.basename(outpath))


def _draw_board_grid(ax, grid, top_y, values, highlight=("J",)):
    """2-row letter board with explicit per-cell placement (mirrors
    g._draw_letter_board mono+show_values styling). None cell = skip."""
    chip_w, chip_h, gx, gy = 2.2, 1.75, 0.35, 0.55
    ncols = max(len(r) for r in grid)
    start_x = (14 - (ncols * chip_w + (ncols - 1) * gx)) / 2
    for r, rowletters in enumerate(grid):
        for c, ch in enumerate(rowletters):
            if ch is None:
                continue
            x = start_x + c * (chip_w + gx)
            y = top_y - r * (chip_h + gy)
            is_hi = ch in highlight
            color = g.lcolor(ch) if is_hi else g.GRAY
            ax.add_patch(FancyBboxPatch(
                (x, y), chip_w, chip_h, boxstyle="round,pad=0.02,rounding_size=0.1",
                linewidth=1.8, edgecolor=color, facecolor="white", zorder=1))
            ax.text(x + chip_w/2, y + chip_h * 0.60, ch, ha="center", va="center",
                    fontsize=40, fontweight="bold", color=color, zorder=3)
            v = values.get(ch, 0)
            vcol = g.INK if v > 0 else g.SUBINK
            ax.text(x + chip_w/2, y + chip_h * 0.22, f"{v} points", ha="center",
                    va="center", fontsize=14, fontweight="bold" if v > 0 else "normal",
                    color=vcol, zorder=3)


def draw_how_points(outpath):
    """Intro slide 2 with M/N moved into the middle of the 2-row board
    (N bottom-center, M diagonally up-right of it)."""
    values = {ch: 0 for ch in g.LETTERS_DISPLAY}
    values["J"] = 10
    fig, ax = plt.subplots(figsize=(14, 8.2), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.2); ax.axis("off")
    ax.text(7, 7.60, "How Points Are Scored", ha="center", va="center",
            fontsize=32, fontweight="bold", color=g.INK)
    ax.text(7, 7.16, "Step 2 of 2", ha="center", va="center", fontsize=13, color=g.SUBINK)
    ax.text(7, 6.66, "Players score only when all six choose the same letter.",
            ha="center", va="center", fontsize=15.5, color=g.INK)
    _draw_board_grid(ax, [POINTS_TOP, POINTS_BOTTOM], top_y=4.7, values=values,
                     highlight=("J",))
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", os.path.basename(outpath))


def _draw_announce_board(ax, grid, top_y, values, highlight_new):
    """De-confounded highlight board for the announcement (mirrors
    g._draw_letter_board with mono=False, show_values=True, show_badge=True):
    letters colored by lcolor, highlighted M/N get tint + crisp border + NEW badge."""
    chip_w, chip_h, gx, gy = 2.2, 1.75, 0.35, 0.55
    ncols = max(len(r) for r in grid)
    start_x = (14 - (ncols * chip_w + (ncols - 1) * gx)) / 2
    for r, rowletters in enumerate(grid):
        for c, ch in enumerate(rowletters):
            if ch is None:
                continue
            x = start_x + c * (chip_w + gx)
            y = top_y - r * (chip_h + gy)
            is_new = ch in highlight_new
            color = g.lcolor(ch)
            ax.add_patch(FancyBboxPatch(
                (x, y), chip_w, chip_h, boxstyle="round,pad=0.02,rounding_size=0.1",
                linewidth=3 if is_new else 1.8, edgecolor=color,
                facecolor=(color if is_new else "white"),
                alpha=0.16 if is_new else 1.0, zorder=1))
            if is_new:  # crisp colored border over the tinted fill
                ax.add_patch(FancyBboxPatch(
                    (x, y), chip_w, chip_h, boxstyle="round,pad=0.02,rounding_size=0.1",
                    linewidth=3, edgecolor=color, facecolor="none", zorder=2))
            ax.text(x + chip_w/2, y + chip_h * 0.60, ch, ha="center", va="center",
                    fontsize=40, fontweight="bold", color=color, zorder=3)
            v = values.get(ch, 0)
            vcol = g.INK if v > 0 else g.SUBINK
            ax.text(x + chip_w/2, y + chip_h * 0.22, f"{v} points", ha="center",
                    va="center", fontsize=14, fontweight="bold" if v > 0 else "normal",
                    color=vcol, zorder=3)
            if is_new:
                bw, bh = 0.95, 0.34
                bx, byy = x + chip_w/2 - bw/2, y + chip_h - bh/2
                ax.add_patch(FancyBboxPatch((bx, byy), bw, bh,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             linewidth=0, facecolor=color, zorder=4))
                ax.text(bx + bw/2, byy + bh/2, "NEW", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white", zorder=5)


def draw_announcement(outpath, reward=15):
    """Round-8 announcement with M/N highlighted in their de-confounded spots."""
    values = {ch: 0 for ch in g.LETTERS_DISPLAY}
    values.update({"J": 10, "M": reward, "N": reward})
    fig, ax = plt.subplots(figsize=(14, 8.2), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.2); ax.axis("off")
    ax.text(7, 7.6, "New for Round 8", ha="center", va="center",
            fontsize=33, fontweight="bold", color=g.R8_EDGE)
    ax.text(7, 7.05, f"M and N are now worth {reward} points each.",
            ha="center", va="center", fontsize=17, color=g.INK)
    _draw_announce_board(ax, [POINTS_TOP, POINTS_BOTTOM], top_y=5.0, values=values,
                         highlight_new=("M", "N"))
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", os.path.basename(outpath))


def main():
    out = os.path.join(BASE, "stimuli", "midplacement")
    os.makedirs(out, exist_ok=True)
    draw_how_to_play(os.path.join(out, "00_how_to_play.png"))
    draw_how_points(os.path.join(out, "01_how_points.png"))
    draw_announcement(os.path.join(out, "04_new_letters.png"))


if __name__ == "__main__":
    main()
