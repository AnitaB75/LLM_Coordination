"""
AI-framed variants of the two intro slides for the Name Game study.

The base stimuli (generate_stimuli.py) tell participants their six co-players are
PEOPLE. These variants recreate the same two slides with the co-players framed as
AI agents, so the human/AI framing can be manipulated while everything else --
letters, payoffs, layout, ordering -- is held fixed:

  "Six people play together"  -> "Six AI agents play together"
  "players"                   -> "AI players"

Six robot personas are drawn above the rules to carry the framing visually.

Two slides only (mirrors 00_how_to_play.png / 01_how_points.png):
  stimuli/ai_framing/00_how_to_play.png   rules + robot personas + available letters
  stimuli/ai_framing/01_how_points.png    robot personas + all letters w/ values (J=10)

The robots are deliberately IDENTICAL in colour. They are drawn in a neutral slate
that is distinct from the J/M/N letter colours (blue/orange/purple), so no robot can
cue a letter and no agent is made more salient than another -- the same
de-confounding logic as generate_stimuli_midplacement.py. Vary ROBOT_INK per index
only if individuating the agents is wanted.

Run:  python3 generate_stimuli_ai.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

import generate_stimuli as g   # reuse primitives, constants, letter board

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stimuli", "ai_framing")

ROBOT_INK = "#5A6678"     # slate body -- deliberately not a J/M/N letter colour
ROBOT_FACE = "#EEF1F6"    # panel fill
ROBOT_EYE = "#1A2233"

AI_RULES = [
    "Six AI agents play together — and they cannot talk to each other.",
    "Each round, all six AI players pick one letter at the same time.",
    "The AI players score points only when all six pick the SAME letter.",
    "After each round, the AI players see what everyone chose.",
]


# ------------------------------ robot persona ------------------------------
def _draw_robot(ax, cx, cy, s=1.0, label=None):
    """Little robot centred at (cx, cy); `s` scales the whole body.

    Geometry is expressed relative to cy so a row of robots lines up on their
    mid-body. Full extent is roughly cy-0.60*s (feet) to cy+0.84*s (antenna tip).
    """
    def box(x_c, y_c, w, h, **kw):
        kw.setdefault("boxstyle", "round,pad=0.005,rounding_size=%.3f" % (0.07 * s))
        kw.setdefault("linewidth", 1.8)
        kw.setdefault("edgecolor", ROBOT_INK)
        kw.setdefault("facecolor", ROBOT_FACE)
        ax.add_patch(FancyBboxPatch((x_c - w / 2, y_c - h / 2), w, h, **kw))

    # antenna
    ax.plot([cx, cx], [cy + 0.56 * s, cy + 0.76 * s],
            color=ROBOT_INK, linewidth=1.8 * s, solid_capstyle="round", zorder=2)
    ax.add_patch(Circle((cx, cy + 0.80 * s), 0.065 * s,
                        facecolor=ROBOT_INK, edgecolor="none", zorder=3))
    # head + face
    box(cx, cy + 0.30 * s, 0.80 * s, 0.56 * s, zorder=2)
    for dx in (-0.17 * s, 0.17 * s):
        ax.add_patch(Circle((cx + dx, cy + 0.36 * s), 0.072 * s,
                            facecolor=ROBOT_EYE, edgecolor="none", zorder=4))
    box(cx, cy + 0.13 * s, 0.30 * s, 0.06 * s,
        linewidth=0, facecolor=ROBOT_INK, zorder=4)      # mouth
    # arms (behind body), body, feet
    for dx in (-0.60 * s, 0.60 * s):
        box(cx + dx, cy - 0.26 * s, 0.15 * s, 0.36 * s, zorder=1)
    box(cx, cy - 0.26 * s, 0.94 * s, 0.62 * s, zorder=2)
    ax.add_patch(Circle((cx, cy - 0.26 * s), 0.10 * s,
                        facecolor=ROBOT_INK, edgecolor="none", zorder=3))
    for dx in (-0.24 * s, 0.24 * s):
        box(cx + dx, cy - 0.63 * s, 0.22 * s, 0.14 * s, zorder=1)

    if label:
        ax.text(cx, cy - 0.90 * s, label, ha="center", va="center",
                fontsize=12, color=g.INK)


def _draw_robot_row(ax, cy, s=1.0, labels=True):
    """The six AI players, evenly spaced and centred on the 14-unit canvas."""
    xs = [2.0 + i * 2.0 for i in range(g.N_PLAYERS)]        # 2.0 .. 12.0, centre 7.0
    for i, cx in enumerate(xs):
        _draw_robot(ax, cx, cy, s=s,
                    label=(f"AI Player {i+1}" if labels else None))


# --------------------------------- pages -----------------------------------
def draw_how_to_play(outpath):
    """Intro slide 1 (AI-framed): rules + the six AI players + available letters."""
    fig, ax = plt.subplots(figsize=(14, 9.9), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 9.9); ax.axis("off")
    ax.text(7, 9.42, "How the Game Works", ha="center", va="center",
            fontsize=32, fontweight="bold", color=g.INK)
    ax.text(7, 8.96, "Step 1 of 2", ha="center", va="center", fontsize=13, color=g.SUBINK)

    _draw_robot_row(ax, cy=7.85, s=0.92)

    y = 6.10
    for i, line in enumerate(AI_RULES, 1):
        ax.add_patch(Circle((1.55, y), 0.27, facecolor=g.LETTER_COLORS["J"], zorder=2))
        ax.text(1.55, y, str(i), ha="center", va="center", fontsize=15,
                fontweight="bold", color="white", zorder=3)
        ax.text(2.15, y, line, ha="left", va="center", fontsize=16.5, color=g.INK)
        y -= 0.82

    ax.text(7, 2.85, "The letters AI players can choose:", ha="center", va="center",
            fontsize=15, fontweight="bold", color=g.SUBINK)
    cw, ch_, gap = 1.12, 1.15, 0.16
    x0 = (14 - (10 * cw + 9 * gap)) / 2
    yc = 1.20
    for i, letter in enumerate(g.LETTERS_DISPLAY):
        x = x0 + i * (cw + gap)
        ax.add_patch(FancyBboxPatch((x, yc), cw, ch_,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.8, edgecolor=g.GRAY, facecolor="white"))
        ax.text(x + cw/2, yc + ch_/2, letter, ha="center", va="center",
                fontsize=22, fontweight="bold", color=g.GRAY)

    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", os.path.basename(outpath))


def draw_how_points(outpath):
    """Intro slide 2 (AI-framed): the six AI players + all letters, J = 10."""
    values = {ch: 0 for ch in g.LETTERS_DISPLAY}
    values["J"] = 10
    fig, ax = plt.subplots(figsize=(14, 9.4), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 9.4); ax.axis("off")
    ax.text(7, 8.90, "How Points Are Scored", ha="center", va="center",
            fontsize=32, fontweight="bold", color=g.INK)
    ax.text(7, 8.46, "Step 2 of 2", ha="center", va="center", fontsize=13, color=g.SUBINK)
    ax.text(7, 7.96, "AI players score only when all six choose the same letter.",
            ha="center", va="center", fontsize=15.5, color=g.INK)

    _draw_robot_row(ax, cy=6.95, s=0.66, labels=False)

    g._draw_letter_board(ax, top_y=4.30, show_values=True, values=values,
                         highlight_new=("J",), show_badge=False, mono=True)
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", os.path.basename(outpath))


def main():
    os.makedirs(OUT, exist_ok=True)
    draw_how_to_play(os.path.join(OUT, "00_how_to_play.png"))
    draw_how_points(os.path.join(OUT, "01_how_points.png"))


if __name__ == "__main__":
    main()
