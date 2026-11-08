"""
Summary figure for the Step 1b screen.

Three panels:
  A  GPT vs Claude dyadic match, all 30 pairs. Distance from the diagonal is the
     dissociation; the cluster at (1,1) is the battery's ceiling problem.
  B  Delta (GPT - Claude) with 95% bootstrap CI for the selected pairs — the
     crossover, both directions.
  C  The conditional-position diagnostic: rate of choosing the modal identity
     when it sits on the left vs on the right. On-diagonal = choosing by FORM.
     Off-diagonal = choosing by SIDE. This is what separates a real preference
     from a side default, and it is the panel that carries the mechanism.

Writes to step1_output/ and ~/Downloads/.

Run:  python3 step1b_screen_figure.py [<ts>]
"""
import os, sys, csv, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("academic")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "step1_output")
DL = os.path.expanduser("~/Downloads")

MS = {"gpt": dict(marker="o", mfc="black", mec="black"),
      "claude": dict(marker="s", mfc="white", mec="black")}
NICE = {"gpt": "GPT-5.4", "claude": "Claude Opus 4.8"}


def main():
    ts = sys.argv[1] if len(sys.argv) > 1 else sorted(
        glob.glob(os.path.join(OUTDIR, "step1b_screen_summary_*.csv")))[-1].split("summary_")[1][:-4]
    S = list(csv.DictReader(open(os.path.join(OUTDIR, f"step1b_screen_summary_{ts}.csv"))))
    L = list(csv.DictReader(open(os.path.join(OUTDIR, f"step1b_screen_selected_{ts}.csv"))))
    idx = {(r["model"], r["stimulus_pair"]): r for r in S}
    f = float

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(8.0, 3.0))

    # ---------------- A: GPT vs Claude ----------------
    n_ceiling = 0
    for r in L:
        g, c = f(r["gpt_match"]), f(r["claude_match"])
        if g == 1.0 and c == 1.0:
            n_ceiling += 1
        sel = r["selected"] == "True"
        axA.plot(c, g, marker="o", ms=5.0 if sel else 3.0,
                 mfc="black" if sel else "white", mec="black", mew=0.7, ls="none")
    for r in L:
        if r["selected"] != "True":
            continue
        g, c = f(r["gpt_match"]), f(r["claude_match"])
        if g > 0.97:                       # ceiling row — label below, else labels collide
            dx, dy, ha = 0, -10, "center"
        elif g > c:
            dx, dy, ha = 6, -1, "left"
        else:
            dx, dy, ha = -6, 3, "right"
        axA.annotate(r["stimulus_pair"], (c, g), textcoords="offset points",
                     xytext=(dx, dy), fontsize=6.2, ha=ha)
    axA.plot([0.42, 1.04], [0.42, 1.04], ls=(0, (4, 3)), lw=0.7, color="black", zorder=0)
    axA.annotate(f"{n_ceiling} pairs at\nceiling (1, 1)", xy=(1.0, 1.0), xytext=(-2, -30),
                 textcoords="offset points", fontsize=6.2, ha="right", linespacing=1.3)
    axA.set_xlim(0.42, 1.06); axA.set_ylim(0.42, 1.06)
    axA.set_xlabel("Claude Opus 4.8"); axA.set_ylabel("GPT-5.4")
    axA.set_title("A   Dyadic match probability", fontsize=8.5, loc="left", pad=6)

    # ---------------- B: delta forest ----------------
    sel = [r for r in L if r["selected"] == "True"]
    sel.sort(key=lambda r: f(r["delta_gpt_minus_claude"]))
    ys = range(len(sel))
    for y, r in zip(ys, sel):
        d = f(r["delta_gpt_minus_claude"])
        lo, hi = f(r["delta_ci_lo"]), f(r["delta_ci_hi"])
        axB.plot([lo, hi], [y, y], lw=0.8, color="black", solid_capstyle="butt")
        axB.plot([lo, lo], [y - .13, y + .13], lw=0.8, color="black")
        axB.plot([hi, hi], [y - .13, y + .13], lw=0.8, color="black")
        axB.plot(d, y, marker="o", ms=4.5, mfc="black", mec="black", ls="none")
    axB.axvline(0, ls=(0, (4, 3)), lw=0.7, color="black", zorder=0)
    axB.set_yticks(list(ys))
    axB.set_yticklabels([r["stimulus_pair"] for r in sel], fontsize=7)
    axB.set_ylim(-1.2, len(sel) - 0.4)
    axB.set_xlim(-0.75, 0.75)
    axB.set_xticks([-0.5, 0, 0.5])
    axB.set_xlabel("GPT − Claude")
    axB.set_title("B   Dissociation (95% CI)", fontsize=8.5, loc="left", pad=6)
    # direction cues below the lowest row, clear of the data
    axB.text(-0.70, -1.05, "Claude higher", fontsize=6.2, va="bottom")
    axB.text(0.70, -1.05, "GPT higher", fontsize=6.2, va="bottom", ha="right")

    # ---------------- C: conditional-position diagnostic ----------------
    for model in ("gpt", "claude"):
        xs = [f(idx[(model, r["stimulus_pair"])]["p_modal_given_AB"]) for r in L]
        ys2 = [f(idx[(model, r["stimulus_pair"])]["p_modal_given_BA"]) for r in L]
        axC.plot(xs, ys2, ms=3.6, mew=0.7, ls="none", label=NICE[model], **MS[model])
    axC.plot([0, 1.04], [0, 1.04], ls=(0, (4, 3)), lw=0.7, color="black", zorder=0)
    # label only the s_arrow exemplars — Claude on the diagonal, GPT far off it
    for model, dx, dy, ha in (("gpt", 6, -2, "left"), ("claude", -6, -9, "right")):
        r = idx[(model, "s_arrow")]
        axC.annotate(f"s_arrow\n({NICE[model].split()[0]})",
                     (f(r["p_modal_given_AB"]), f(r["p_modal_given_BA"])),
                     textcoords="offset points", xytext=(dx, dy), fontsize=6.2,
                     ha=ha, linespacing=1.25)
    axC.set_xlim(-0.06, 1.10); axC.set_ylim(-0.06, 1.14)
    axC.set_xticks([0, 0.5, 1.0]); axC.set_yticks([0, 0.5, 1.0])
    axC.set_xlabel("P(modal | modal on left)")
    axC.set_ylabel("P(modal | modal on right)")
    axC.set_title("C   Form vs. side", fontsize=8.5, loc="left", pad=6)
    axC.legend(frameon=False, loc="center left", fontsize=6.4, handletextpad=0.3,
               labelspacing=0.25, borderaxespad=0.2)

    for ax in (axA, axB, axC):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(w_pad=1.6)

    os.makedirs(DL, exist_ok=True)
    for d in (OUTDIR, DL):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(d, f"step1b_screen_findings.{ext}"))
    plt.close(fig)
    print("wrote step1b_screen_findings.png/.pdf to:")
    print(" ", OUTDIR)
    print(" ", DL)


if __name__ == "__main__":
    main()
