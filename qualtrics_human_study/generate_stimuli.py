"""
Qualtrics stimulus images for the human Name Game PREDICTION study.

Per-arm flow:
  00_how_the_game_works.png   intro: rules + all available letters (no player choices)
  01_history_rounds_1-7.png   rounds 1-7 history (all J) -- no match/points shown
  02_new_letters.png          round-8 announcement: all letters w/ values; M & N -> 15
  03..06_round8_result_*M.png round-8 results (0M/1M/2M/3M) -- no match/points shown
                              -> participants then predict round 9

Letters in the game: J P Q R N M F C X Y.  J scores 10 (rounds 1-7);
from round 8, M and N each score 15; all other letters score 0.

Per the design, the board images deliberately do NOT announce whether players
matched or how many points were earned -- only the letters chosen.

Edit CONFIG and re-run:  python3 generate_stimuli.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

# ----------------------------- CONFIG -------------------------------------
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stimuli")

N_PLAYERS = 6
LETTERS_DISPLAY = ["J", "M", "N", "P", "Q", "R", "F", "C", "X", "Y"]  # J/M/N first
LETTER_COLORS = {"J": "#2E6FDB", "M": "#E8820C", "N": "#8E44AD"}      # blue/orange/purple
GRAY = "#8A93A3"
CELL_FACE, CELL_EDGE = "#F4F6FB", "#D3DAE8"
R8_FACE, R8_EDGE = "#FFF6E6", "#E8820C"
INK, SUBINK = "#1A2233", "#5A6678"

PLAYER_LABELS = [f"Player {i+1}" for i in range(N_PLAYERS)]
GRID_LEFT, GRID_RIGHT = 1.75, 13.55
ROW_H, PAD = 0.74, 0.07
# --------------------------------------------------------------------------


def lcolor(ch):
    return LETTER_COLORS.get(ch, GRAY)


# ---------- player x round board (history / round-8 result) ----------------
PREDICT_FACE, PREDICT_EDGE = "#EEF1F6", "#5A6678"   # round-9 "?" column styling


def _draw_grid(ax, mat, top, highlight_col=None, predict_col=None,
               divider_after_col=None, divider_label=None):
    """Letter grid only (no points row). highlight_col = round-8 result (orange);
    predict_col = round-9 prediction target ("?" cells);
    divider_after_col = draw a dotted phase-shift line after that column.
    Returns bottom y."""
    n_rounds = len(mat[0])
    col_w = (GRID_RIGHT - GRID_LEFT) / n_rounds
    col_x = lambda c: GRID_LEFT + c * col_w
    row_y = lambda r: top - r * ROW_H
    bottom = row_y(N_PLAYERS - 1) - ROW_H

    # column background bands
    for col, face, edge in ((highlight_col, R8_FACE, R8_EDGE),
                            (predict_col, PREDICT_FACE, PREDICT_EDGE)):
        if col is not None:
            ax.add_patch(Rectangle(
                (col_x(col) + PAD/2, row_y(N_PLAYERS - 1) - ROW_H + PAD/2),
                col_w - PAD, ROW_H * N_PLAYERS, facecolor=face,
                edgecolor=edge, linewidth=2.0, zorder=0))

    for c in range(n_rounds):
        is_h, is_p = (c == highlight_col), (c == predict_col)
        hdr_color = R8_EDGE if is_h else (PREDICT_EDGE if is_p else SUBINK)
        ax.text(col_x(c) + col_w/2, top + 0.32, f"Round {c+1}", ha="center", va="center",
                fontsize=13, fontweight="bold" if (is_h or is_p) else "normal",
                color=hdr_color)

    for r in range(N_PLAYERS):
        ax.text(GRID_LEFT - 0.18, row_y(r) - ROW_H/2, PLAYER_LABELS[r],
                ha="right", va="center", fontsize=13, color=INK)
        for c in range(n_rounds):
            ch = mat[r][c]
            is_h, is_p = (c == highlight_col), (c == predict_col)
            x0, y0 = col_x(c) + PAD/2, row_y(r) - ROW_H + PAD/2
            if is_p:
                edge, face, ls = PREDICT_EDGE, "white", (0, (3, 2))
            elif is_h:
                edge, face, ls = R8_EDGE, "white", "solid"
            else:
                edge, face, ls = CELL_EDGE, CELL_FACE, "solid"
            ax.add_patch(FancyBboxPatch(
                (x0, y0), col_w - PAD, ROW_H - PAD,
                boxstyle="round,pad=0.005,rounding_size=0.06", linewidth=1.4,
                edgecolor=edge, facecolor=face, linestyle=ls, zorder=2))
            ax.text(x0 + (col_w - PAD)/2, y0 + (ROW_H - PAD)/2, ch, ha="center",
                    va="center", fontsize=22, fontweight="bold",
                    color=(PREDICT_EDGE if is_p else lcolor(ch)), zorder=3)

    # dotted phase-shift divider (after divider_after_col), with label above
    if divider_after_col is not None:
        xd = col_x(divider_after_col + 1)
        ax.plot([xd, xd], [bottom + PAD/2, top + 0.42], color=R8_EDGE,
                linewidth=2.4, linestyle=(0, (2, 3)), zorder=5)
        if divider_label:
            ax.text(xd, top + 0.60, divider_label, ha="center", va="bottom",
                    fontsize=12.5, fontweight="bold", color=R8_EDGE)
    return bottom


def _draw_payoff_legend(ax, y):
    """Bottom legend: all letters and their point values (J=10, M/N=15, rest 0)."""
    vals = {ch: 0 for ch in LETTERS_DISPLAY}
    vals.update({"J": 10, "M": 15, "N": 15})
    ax.text(7, y + 1.18, "What each letter is worth (points):", ha="center",
            va="center", fontsize=13, fontweight="bold", color=SUBINK)
    cw, ch_, gap = 1.18, 0.9, 0.12
    x0 = (14 - (10 * cw + 9 * gap)) / 2
    for i, ch in enumerate(LETTERS_DISPLAY):
        x = x0 + i * (cw + gap)
        color = lcolor(ch); v = vals[ch]
        ax.add_patch(FancyBboxPatch(
            (x, y), cw, ch_, boxstyle="round,pad=0.01,rounding_size=0.06",
            linewidth=1.6, edgecolor=color, facecolor="white"))
        ax.text(x + cw/2, y + ch_ * 0.62, ch, ha="center", va="center",
                fontsize=16, fontweight="bold", color=color)
        ax.text(x + cw/2, y + ch_ * 0.24, str(v), ha="center", va="center",
                fontsize=11, fontweight="bold" if v > 0 else "normal",
                color=INK if v > 0 else SUBINK)


# ---------- letter board (intro / announcement) ----------------------------
def _draw_letter_board(ax, top_y, show_values=False, values=None, highlight_new=(),
                       show_badge=True, mono=False):
    """2 rows x 5 letter chips. Optional point values; optional highlight + NEW badge.
    mono=True renders every non-highlighted letter in grey."""
    chip_w, chip_h, gx, gy = 2.2, 1.75, 0.35, 0.55
    start_x = (14 - (5 * chip_w + 4 * gx)) / 2
    for idx, ch in enumerate(LETTERS_DISPLAY):
        r, c = idx // 5, idx % 5
        x = start_x + c * (chip_w + gx)
        y = top_y - r * (chip_h + gy)
        is_new = ch in highlight_new
        color = lcolor(ch) if (is_new or not mono) else GRAY
        ax.add_patch(FancyBboxPatch(
            (x, y), chip_w, chip_h, boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=3 if is_new else 1.8, edgecolor=color,
            facecolor=(color if is_new else "white"),
            alpha=0.16 if is_new else 1.0, zorder=1))
        if is_new:  # re-draw crisp colored border over the tinted fill
            ax.add_patch(FancyBboxPatch(
                (x, y), chip_w, chip_h, boxstyle="round,pad=0.02,rounding_size=0.1",
                linewidth=3, edgecolor=color, facecolor="none", zorder=2))
        ly = y + chip_h * (0.60 if show_values else 0.5)
        ax.text(x + chip_w/2, ly, ch, ha="center", va="center",
                fontsize=40, fontweight="bold", color=color, zorder=3)
        if show_values:
            v = values.get(ch, 0)
            vcol = INK if v > 0 else SUBINK
            ax.text(x + chip_w/2, y + chip_h * 0.22,
                    f"{v} points", ha="center", va="center",
                    fontsize=14, fontweight="bold" if v > 0 else "normal",
                    color=vcol, zorder=3)
        if is_new and show_badge:
            bw, bh = 0.95, 0.34
            bx, byy = x + chip_w/2 - bw/2, y + chip_h - bh/2
            ax.add_patch(FancyBboxPatch((bx, byy), bw, bh,
                                        boxstyle="round,pad=0.01,rounding_size=0.08",
                                        linewidth=0, facecolor=color, zorder=4))
            ax.text(bx + bw/2, byy + bh/2, "NEW", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white", zorder=5)


# ------------------------------- pages -------------------------------------
def _payoff_chip(ax, x, y, w, h, letter, value, is_new):
    """Horizontal chip: colored letter + '{value} pts', optional NEW badge."""
    color = lcolor(letter)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                linewidth=2.4, edgecolor=color, facecolor="white"))
    ax.text(x + w * 0.32, y + h/2, letter, ha="center", va="center",
            fontsize=30, fontweight="bold", color=color)
    ax.text(x + w * 0.72, y + h/2, f"{value} pts", ha="center", va="center",
            fontsize=15, fontweight="bold", color=INK)
    if is_new:
        bw, bh = 0.82, 0.32
        ax.add_patch(FancyBboxPatch((x + w/2 - bw/2, y + h - bh/2), bw, bh,
                                    boxstyle="round,pad=0.01,rounding_size=0.08",
                                    linewidth=0, facecolor=color))
        ax.text(x + w/2, y + h - bh/2 + bh/2, "NEW", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white")


def draw_how_to_play(outpath):
    """Intro slide 1: the rules + the letters you can choose."""
    fig, ax = plt.subplots(figsize=(14, 8.3), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.3); ax.axis("off")
    ax.text(7, 7.80, "How the Game Works", ha="center", va="center",
            fontsize=32, fontweight="bold", color=INK)
    ax.text(7, 7.34, "Step 1 of 2", ha="center", va="center", fontsize=13, color=SUBINK)

    rules = [
        "Six people play together — and they cannot talk to each other.",
        "Each round, all six players pick one letter at the same time.",
        "The players score points only when all six pick the SAME letter.",
        "After each round, the players see what everyone chose.",
    ]
    y = 6.45
    for i, line in enumerate(rules, 1):
        ax.add_patch(Circle((1.55, y), 0.27, facecolor=LETTER_COLORS["J"], zorder=2))
        ax.text(1.55, y, str(i), ha="center", va="center", fontsize=15,
                fontweight="bold", color="white", zorder=3)
        ax.text(2.15, y, line, ha="left", va="center", fontsize=16.5, color=INK)
        y -= 0.82

    ax.text(7, 3.20, "The letters players can choose:", ha="center", va="center",
            fontsize=15, fontweight="bold", color=SUBINK)
    cw, ch_, gap = 1.12, 1.15, 0.16
    x0 = (14 - (10 * cw + 9 * gap)) / 2
    yc = 1.55
    for i, letter in enumerate(LETTERS_DISPLAY):
        x = x0 + i * (cw + gap)
        ax.add_patch(FancyBboxPatch((x, yc), cw, ch_, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.8, edgecolor=GRAY, facecolor="white"))
        ax.text(x + cw/2, yc + ch_/2, letter, ha="center", va="center",
                fontsize=22, fontweight="bold", color=GRAY)

    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", os.path.basename(outpath))


def draw_how_points(outpath):
    """Intro slide 2: all letters with values — J = 10, everything else = 0."""
    values = {ch: 0 for ch in LETTERS_DISPLAY}
    values["J"] = 10
    fig, ax = plt.subplots(figsize=(14, 8.2), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.2); ax.axis("off")
    ax.text(7, 7.60, "How Points Are Scored", ha="center", va="center",
            fontsize=32, fontweight="bold", color=INK)
    ax.text(7, 7.16, "Step 2 of 2", ha="center", va="center", fontsize=13, color=SUBINK)
    ax.text(7, 6.66, "Players score only when all six choose the same letter.",
            ha="center", va="center", fontsize=15.5, color=INK)
    _draw_letter_board(ax, top_y=4.7, show_values=True, values=values,
                       highlight_new=("J",), show_badge=False, mono=True)
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", os.path.basename(outpath))


def draw_payoff_rounds17(outpath):
    """Rounds 1-7 payoff slide: J highlighted at 10 points (styled like the M/N reveal)."""
    values = {ch: 0 for ch in LETTERS_DISPLAY}
    values["J"] = 10
    fig, ax = plt.subplots(figsize=(14, 8.2), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.2); ax.axis("off")
    ax.text(7, 7.6, "Rounds 1 to 7", ha="center", va="center",
            fontsize=33, fontweight="bold", color=LETTER_COLORS["J"])
    ax.text(7, 7.05, "Only J scores — 10 points if all six players choose it.",
            ha="center", va="center", fontsize=17, color=INK)
    _draw_letter_board(ax, top_y=5.0, show_values=True, values=values,
                       highlight_new=("J",), show_badge=False)
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", os.path.basename(outpath))


def draw_announcement(outpath):
    values = {ch: 0 for ch in LETTERS_DISPLAY}
    values.update({"J": 10, "M": 15, "N": 15})
    fig, ax = plt.subplots(figsize=(14, 8.2), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.2); ax.axis("off")
    ax.text(7, 7.6, "New for Round 8", ha="center", va="center",
            fontsize=33, fontweight="bold", color=R8_EDGE)
    ax.text(7, 7.05, "M and N are now worth 15 points each.",
            ha="center", va="center", fontsize=17, color=INK)
    _draw_letter_board(ax, top_y=5.0, show_values=True, values=values,
                       highlight_new=("M", "N"))
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", os.path.basename(outpath))


def draw_history(outpath):
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.5); ax.axis("off")
    ax.text(7, 6.05, "The Game So Far — Rounds 1 to 7", ha="center", va="center",
            fontsize=29, fontweight="bold", color=INK)
    ax.text(7, 5.60, "What each player chose in every round.",
            ha="center", va="center", fontsize=16, color=SUBINK)
    mat = [["J"] * 7 for _ in range(N_PLAYERS)]
    _draw_grid(ax, mat, top=4.85)
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", os.path.basename(outpath))


def draw_result(n_M, outpath):
    n_J = N_PLAYERS - n_M
    # rounds 1-8 of choices, plus a round-9 column of "?" (the prediction target)
    mat = [["J"] * 8 + ["?"] for _ in range(N_PLAYERS)]
    for p in range(N_PLAYERS - n_M, N_PLAYERS):
        mat[p][7] = "M"
    fig, ax = plt.subplots(figsize=(14, 8.3), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.3); ax.axis("off")
    ax.text(7, 7.75, "Rounds 1–8 Results — What Will Happen in Round 9?",
            ha="center", va="center", fontsize=25, fontweight="bold", color=INK)
    _draw_grid(ax, mat, top=6.45, highlight_col=7, predict_col=8,
               divider_after_col=6, divider_label="M & N now worth 15")
    _draw_payoff_legend(ax, y=0.55)
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", os.path.basename(outpath))


def main():
    os.makedirs(OUT, exist_ok=True)
    draw_how_to_play(os.path.join(OUT, "00_how_to_play.png"))
    draw_how_points(os.path.join(OUT, "01_how_points.png"))
    draw_payoff_rounds17(os.path.join(OUT, "02_payoff_rounds_1-7.png"))
    draw_history(os.path.join(OUT, "03_history_rounds_1-7.png"))
    draw_announcement(os.path.join(OUT, "04_new_letters.png"))
    for i, n_M in enumerate((0, 1, 2, 3), start=5):
        draw_result(n_M, os.path.join(OUT, f"0{i}_round8_result_{n_M}M.png"))


if __name__ == "__main__":
    main()
