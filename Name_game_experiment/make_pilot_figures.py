"""
Clean, minimal scientific figures for the Name Game pilots.
Saves each PNG to ./figures/ and ~/Downloads/.

Fig 1  experimental design schematic
Fig 2  partner-belief manipulation (convention abandonment), n=10/cell
Fig 3  the focal-point hack and the un-hackable roundabout design
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

CLAUDE, GPT = "#2E6FDB", "#E8820C"           # validated CVD-safe pair
INK, SUB, MUT = "#1A2233", "#5A6678", "#9AA3B2"
GRID = "#E7EAF0"
OUTDIRS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"),
           os.path.expanduser("~/Downloads")]
for dpath in OUTDIRS:
    os.makedirs(dpath, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": MUT, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB,
    "text.color": INK, "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    for dpath in OUTDIRS:
        fig.savefig(os.path.join(dpath, name), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ─────────────────────────────────────────────────────────────────────────
# FIG 1 — experimental design
# ─────────────────────────────────────────────────────────────────────────
def fig_design():
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0, 9.5, "The Name Game — experimental design", fontsize=15, fontweight="bold", color=INK)
    ax.text(0, 8.85, "6 model instances, 14 rounds, simultaneous choice, no communication. "
            "Everyone scores only if all 6 match.", fontsize=10.5, color=SUB)

    # round timeline
    y0, h = 6.2, 1.05
    for r in range(14):
        phase1 = r < 7
        col = "#EEF2FB" if phase1 else "#FFF4E6"
        edge = CLAUDE if phase1 else GPT
        ax.add_patch(FancyBboxPatch((r + 0.06, y0), 0.88, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                                    facecolor=col, edgecolor=edge, linewidth=1.1))
        ax.text(r + 0.5, y0 + h / 2, str(r + 1), ha="center", va="center", fontsize=9, color=SUB)
    ax.plot([7, 7], [y0 - 0.35, y0 + h + 0.55], color=GPT, lw=2, ls=(0, (2, 2)))
    ax.text(7, y0 + h + 0.75, "payoff shift", ha="center", fontsize=9.5, color=GPT, fontweight="bold")
    ax.text(3.5, y0 - 0.75, "Rounds 1–7:  only J scores (10 pts)\n→ group converges on J (a convention)",
            ha="center", fontsize=10, color=INK)
    ax.text(10.5, y0 - 0.75, "Rounds 8–14:  J = 10,  M = 15,  N = 15\n→ switch to a higher option, or hold J?",
            ha="center", fontsize=10, color=INK)

    # conditions
    ax.text(0, 3.3, "Partner-belief manipulation (what each instance is told about the other 5):",
            fontsize=11, fontweight="bold", color=INK)
    conds = [("self", "all 5 are\ncopies of you"),
             ("mixed", "copies-of-you\n+ humans"),
             ("mixed_ai", "other AI models\n+ humans"),
             ("human", "all 5 are\nhuman participants")]
    for i, (name, desc) in enumerate(conds):
        x = i * 3.5
        ax.add_patch(FancyBboxPatch((x, 1.35), 3.2, 1.55, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="#F7F8FB", edgecolor=MUT, linewidth=1))
        ax.text(x + 0.2, 2.55, name, fontsize=11, fontweight="bold", color=CLAUDE)
        ax.text(x + 0.2, 2.05, desc, fontsize=8.6, color=SUB, va="top", linespacing=1.35)
    ax.text(0, 0.7, "Belief-only: all 6 seats are the same model on the same prompt — nobody is actually human. "
            "Run on gpt-5.4 and claude-opus-4-8, n = 10 games per cell.", fontsize=9, color=MUT)
    save(fig, "fig1_design.png")


# ─────────────────────────────────────────────────────────────────────────
# FIG 2 — belief manipulation result
# ─────────────────────────────────────────────────────────────────────────
def fig_belief():
    conds = ["self", "mixed\n(self+human)", "mixed_ai\n(AI+human)", "human"]
    claude = [10, 3, 0, 0]
    gpt = [10, 9, 4, 4]
    x = range(len(conds)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    b1 = ax.bar([i - w/2 for i in x], claude, w, color=CLAUDE, label="Claude (opus-4-8)", zorder=3)
    b2 = ax.bar([i + w/2 for i in x], gpt, w, color=GPT, label="GPT (5.4)", zorder=3)
    for bars, vals in ((b1, claude), (b2, gpt)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, str(v), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(0, 11)
    ax.set_yticks(range(0, 11, 2))
    ax.set_ylabel("games that abandoned J for a higher option  (of 10)")
    ax.set_xticks(list(x)); ax.set_xticklabels(conds, fontsize=10)
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
    ax.set_title("Believing partners are copies of you drives convention-breaking", pad=12, fontsize=12)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.text(0.5, -0.09, "Round-14 outcome, n = 10 games / cell. Both models abandon J only when they believe "
             "partners share their reasoning;\ngeneric-AI partners (mixed_ai) are treated like humans.",
             ha="center", fontsize=9, color=SUB)
    save(fig, "fig2_belief_manipulation.png")


# ─────────────────────────────────────────────────────────────────────────
# FIG 3 — the hack and the solution
# ─────────────────────────────────────────────────────────────────────────
def fig_hack_solution():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 4.9), dpi=200,
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) a-priori self-play convergence on a higher option, by stimulus label
    labs = ["letters\n(M / N)", "random\nstrings", "car\nmakes", "roundabout\n(private frames)"]
    rate = [100, 100, 100, 0]
    cols = [MUT, MUT, MUT, CLAUDE]
    bars = axl.barh(range(len(labs))[::-1], rate, color=cols, height=0.62, zorder=3)
    for i, (v, lab) in enumerate(zip(rate, labs)):
        yy = list(range(len(labs)))[::-1][i]
        axl.text(v + 2 if v < 90 else v - 3, yy, f"{v}%", va="center",
                 ha="left" if v < 90 else "right",
                 fontsize=10, fontweight="bold", color=INK if v < 90 else "white")
    axl.set_yticks(range(len(labs))[::-1]); axl.set_yticklabels(labs, fontsize=9.5)
    axl.set_xlim(0, 108); axl.set_xticks(range(0, 101, 25))
    axl.set_xlabel("games coordinating on a higher option a priori  (%)")
    axl.xaxis.grid(True, color=GRID, lw=1, zorder=0); axl.set_axisbelow(True)
    axl.set_title("Self-play defeats every label —\nuntil the frame is private", fontsize=11.5, pad=12)

    # (b) roundabout: points earned vs achievable, standard vs devalued convention
    grp = ["standard\n(J = 10)", "devalued\n(J = 3)"]
    earned = [70, 21]; achievable = [70, 105]
    xx = range(len(grp)); w = 0.36
    axr.bar([i - w/2 for i in xx], achievable, w, color=GRID, edgecolor=MUT, lw=1,
            label="achievable (coordinate on a 15)", zorder=2)
    axr.bar([i + w/2 for i in xx], earned, w, color=CLAUDE, label="earned (held J)", zorder=3)
    for i, (a, e) in enumerate(zip(achievable, earned)):
        axr.text(i - w/2, a + 1.5, str(a), ha="center", fontsize=9.5, color=SUB)
        axr.text(i + w/2, e + 1.5, str(e), ha="center", fontsize=9.5, fontweight="bold", color=INK)
    axr.set_ylim(0, 118); axr.set_xticks(list(xx)); axr.set_xticklabels(grp, fontsize=9.5)
    axr.set_ylabel("points, rounds 8–14")
    axr.yaxis.grid(True, color=GRID, lw=1, zorder=0); axr.set_axisbelow(True)
    axr.set_title("0 / 20 leaks — and they won't\nexplore even at a 5× incentive", fontsize=11.5, pad=12)
    axr.legend(frameon=False, fontsize=8.6, loc="upper left")

    fig.text(0.5, -0.03, "Self-play, both models. Left: any labelled option set is coordinated a priori via a "
             "manufactured focal point (alphabet / position / salience); private per-seat frames drive it to 0. "
             "Right: roundabout, n = 5 games × 2 models — the group holds the convention even when it is worth 5× "
             "less than an achievable switch.", ha="center", fontsize=8.6, color=SUB, wrap=True)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig3_hack_and_solution.png")


fig_design()
fig_belief()
fig_hack_solution()
print("\nsaved to:", *OUTDIRS, sep="\n  ")
