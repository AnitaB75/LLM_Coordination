"""
Trial-by-trial choice grids for the 3-condition Name Game runs -- ALL ON ONE PAGE.

One figure, 2 rows (models) x 3 cols (conditions). Each panel stacks its 10
trials; every trial is a 6-player (rows) x 14-round (cols) board, each cell
SOLID-filled with the chosen letter's colour (J=blue, M=orange, N=purple, other
grey), letter drawn on top. Same palette as the Qualtrics stimuli: an all-J game
reads blue, all-M orange, split rounds show as multi-colour columns. Dashed line
after round 7 = payoff shift.

Saved to ~/Downloads/choices_all.png.  Run:  python3 plot_trial_choices.py
"""
import os, csv, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

LETTER_COLORS = {"J": "#2E6FDB", "M": "#E8820C", "N": "#8E44AD"}
GREY = "#8A93A3"
INK, SUBINK, R8_EDGE = "#1A2233", "#5A6678", "#E8820C"
N_PLAYERS, N_ROUNDS = 6, 14
OUT = os.path.expanduser("~/Downloads")

# each model: (label, 3-condition CSV glob, mixed_ai CSV glob)
MODELS = [("gpt-5.4", "name_game_3cond_gpt_*.csv", "name_game_mixedAI_gpt_*.csv"),
          ("claude-opus-4-8", "name_game_3cond_claude_*.csv", "name_game_mixedAI_claude_*.csv")]
CONDS = ["self", "mixed", "mixed_ai", "human"]
COND_SUB = {"self": "all copies of me", "mixed": "self + human",
            "mixed_ai": "generic AI + human", "human": "all human"}


def lcolor(ch):
    return LETTER_COLORS.get(ch, GREY)


def load(pat):
    return list(csv.DictReader(open(sorted(glob.glob(pat))[-1])))


def games(rows, cond):
    out = {}
    for r in rows:
        if r["partner_condition"] != cond:
            continue
        t, rd = int(r["trial"]), int(r["round"])
        mat = out.setdefault(t, [[""] * N_ROUNDS for _ in range(N_PLAYERS)])
        for i in range(N_PLAYERS):
            mat[i][rd - 1] = r[f"p{i+1}"]
    return dict(sorted(out.items()))


def draw_panel(ax, model, cond, gmap):
    trials = list(gmap)
    gap = 1  # blank rows between trials
    unit = N_PLAYERS + gap
    total_rows = len(trials) * unit
    ax.set_xlim(-2.4, N_ROUNDS + 0.3)
    ax.set_ylim(total_rows, -1.6)   # inverted: trial 1 at top
    ax.axis("off")
    ax.set_title(f"{model}  —  {cond}\n({COND_SUB.get(cond, '')})",
                 fontsize=12, fontweight="bold", color=INK, pad=10)

    # round-number header
    for c in range(N_ROUNDS):
        ax.text(c + 0.5, -0.7, str(c + 1), ha="center", va="center",
                fontsize=6.5, color=SUBINK)

    for ti, t in enumerate(trials):
        mat = gmap[t]
        y0 = ti * unit
        ax.text(-0.35, y0 + N_PLAYERS / 2, f"t{t}", ha="right", va="center",
                fontsize=7.5, color=INK)
        for i in range(N_PLAYERS):
            for c in range(N_ROUNDS):
                ch_ = mat[i][c]
                ax.add_patch(Rectangle((c + 0.03, y0 + i + 0.03), 0.94, 0.94,
                                       facecolor=lcolor(ch_), edgecolor="white",
                                       linewidth=0.4))
                ax.text(c + 0.5, y0 + i + 0.5, ch_, ha="center", va="center",
                        fontsize=5.5, fontweight="bold", color="white")
        ax.plot([7, 7], [y0, y0 + N_PLAYERS], color=R8_EDGE, linewidth=1.3,
                linestyle=(0, (2, 2)), zorder=5)


def main():
    fig = plt.figure(figsize=(29, 26), dpi=145)
    gs = GridSpec(2, 4, figure=fig, hspace=0.10, wspace=0.13)
    for ri, (model, three_pat, mai_pat) in enumerate(MODELS):
        rows = load(three_pat) + load(mai_pat)   # combine 3-cond + mixed_ai
        for ci, cond in enumerate(CONDS):
            ax = fig.add_subplot(gs[ri, ci])
            draw_panel(ax, model, cond, games(rows, cond))

    fig.suptitle("Name Game — per-player letter choice by trial  "
                 "(rows = 6 players, cols = 14 rounds)",
                 fontsize=17, fontweight="bold", color=INK, y=0.995)
    # shared legend
    handles = [Rectangle((0, 0), 1, 1, facecolor=LETTER_COLORS["J"]),
               Rectangle((0, 0), 1, 1, facecolor=LETTER_COLORS["M"]),
               Rectangle((0, 0), 1, 1, facecolor=LETTER_COLORS["N"]),
               Rectangle((0, 0), 1, 1, facecolor=GREY),
               plt.Line2D([0], [0], color=R8_EDGE, lw=1.6, ls=(0, (2, 2)))]
    labels = ["J = 10", "M = 15", "N = 15", "other = 0",
              "payoff shift (after round 7)"]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 0.006))

    out = os.path.join(OUT, "choices_all.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
