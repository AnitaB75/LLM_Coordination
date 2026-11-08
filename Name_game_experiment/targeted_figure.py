"""
STEP 1c figure - four panels.

  A  GPT vs Claude dyadic match probability, all screened candidates
  B  disparity effect sizes with bootstrap 95% CIs
  C  the form-consistency diagnostic - which disparities are real focality
     differences and which are one model falling back on position
  D  confirmation results for the selected battery, fresh sample

Academic style throughout: Arial ~9pt, no titles inside the axes, black 0.8pt
spines with top/right removed, outward ticks, thin capped error bars. Series are
separated by MARKER and FILL as well as tone, so the figure survives greyscale.
PDF is written alongside the PNG so the text stays vector.

Run:  python3 targeted_figure.py
"""
import os, csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("academic")
except OSError:
    pass
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.edgecolor": "black", "text.color": "black",
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "step1c_output")


def read(name):
    p = os.path.join(OUT, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def f(r, k, default=float("nan")):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return default


def bare(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    comp = read("targeted_screen_comparison.csv")
    sel = read("targeted_screen_selected.csv")
    conf = read("targeted_confirmation_comparison.csv")
    if not comp:
        sys.exit("no targeted_screen_comparison.csv - run targeted_analysis.py first")

    status = {r["stimulus_id"]: r["screen_status"] for r in sel}
    picked = {r["stimulus_id"] for r in sel if r["selection_status"] == "SELECT"}

    # Shape carries the meaning, not tone, so the figure survives greyscale.
    STY = {  # status -> (marker, facecolor, size)
        "PRIMARY":          ("o", "black", 6.0),
        "SECONDARY":        ("s", "white", 5.5),
        "POSITION_DEFAULT": ("v", "white", 5.8),
        "REJECT":           ("o", "0.75", 3.4),
    }
    LEG = [("clean disparity (PRIMARY)", "PRIMARY"),
           ("partial position fallback", "SECONDARY"),
           ("weak model chooses a side", "POSITION_DEFAULT"),
           ("no disparity", "REJECT")]

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34,
                          left=0.09, right=0.98, top=0.95, bottom=0.13)

    # ---- A. GPT vs Claude match -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for r in comp:
        st = status.get(r["stimulus_pair"], "REJECT")
        m, fc, ms = STY.get(st, STY["REJECT"])
        ax.plot(f(r, "claude_match"), f(r, "gpt_match"), marker=m, ms=ms,
                mfc=fc, mec="black", mew=0.7, ls="none")
        # only the clean disparity is named here; panel B lists every candidate
        # on its axis, so repeating the names would just collide
        if st == "PRIMARY":
            ax.annotate(r["stimulus_pair"], (f(r, "claude_match"), f(r, "gpt_match")),
                        textcoords="offset points", xytext=(-8, 8), fontsize=6.5,
                        ha="right")
    lim = (0.42, 1.04)
    ax.plot(lim, lim, ls=(0, (4, 3)), lw=0.7, color="black", zorder=0)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xlabel("Claude Opus 4.8 — dyadic match")
    ax.set_ylabel("GPT-5.4 — dyadic match")
    ax.text(-0.20, 1.04, "A", transform=ax.transAxes, fontsize=11, va="top")
    bare(ax)

    # one legend for the whole figure, below the panels, so it never sits on data
    handles = [plt.Line2D([], [], marker=STY[k][0], mfc=STY[k][1], mec="black",
                          mew=0.7, ls="none", ms=STY[k][2], label=lab)
               for lab, k in LEG]

    # ---- B. effect sizes with CIs -----------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    top = sorted(comp, key=lambda r: f(r, "delta_gpt_minus_claude"))[:0] or comp
    top = sorted(comp, key=lambda r: -abs(f(r, "delta_gpt_minus_claude")))[:12]
    top = sorted(top, key=lambda r: f(r, "delta_gpt_minus_claude"))
    y = np.arange(len(top))
    for i, r in enumerate(top):
        d = f(r, "delta_gpt_minus_claude")
        lo, hi = f(r, "delta_ci_lo"), f(r, "delta_ci_hi")
        st = status.get(r["stimulus_pair"], "REJECT")
        m, fc, ms = STY.get(st, STY["REJECT"])
        ax.plot([lo, hi], [i, i], lw=0.8, color="black", solid_capstyle="butt")
        for x in (lo, hi):
            ax.plot([x, x], [i - 0.16, i + 0.16], lw=0.8, color="black")
        ax.plot(d, i, marker=m, ms=ms, mfc=fc, mec="black", mew=0.7, ls="none")
    ax.axvline(0, lw=0.7, color="black", ls=(0, (4, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels([r["stimulus_pair"] for r in top], fontsize=6.5)
    ax.set_xlabel("GPT − Claude dyadic match (95% CI)")
    ax.text(-0.42, 1.04, "B", transform=ax.transAxes, fontsize=11, va="top")
    bare(ax)

    # ---- C. form-consistency diagnostic -----------------------------------
    ax = fig.add_subplot(gs[1, 0])
    # stagger label offsets in a repeating cycle so near-coincident points, of
    # which this panel has several, stay individually readable
    OFFSETS = [(6, 6), (6, -11), (-7, 7), (-7, -12)]
    k = 0
    for r in comp:
        st = status.get(r["stimulus_pair"], "REJECT")
        m, fc, ms = STY.get(st, STY["REJECT"])
        x, y = f(r, "weak_form_consistency"), abs(f(r, "delta_gpt_minus_claude"))
        ax.plot(x, y, marker=m, ms=ms, mfc=fc, mec="black", mew=0.7, ls="none")
        if st in ("PRIMARY", "SECONDARY", "POSITION_DEFAULT"):
            dx, dy = OFFSETS[k % len(OFFSETS)]
            k += 1
            if x < 0.15:            # keep labels on points near the axis inboard
                dx = abs(dx)
            ax.annotate(r["stimulus_pair"], (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=6.2,
                        ha="left" if dx > 0 else "right")
    ax.axvline(0.80, lw=0.7, color="black", ls=(0, (2, 2)))
    ax.axhline(0.20, lw=0.7, color="black", ls=(0, (2, 2)))
    ax.set_xlim(-0.02, 1.04)
    ax.set_xlabel("weaker model — form consistency")
    ax.set_ylabel("|GPT − Claude| dyadic match")
    ax.text(0.03, 0.96, "clean disparities\nlie right of 0.80",
            transform=ax.transAxes, fontsize=6.5, va="top")
    ax.text(-0.20, 1.04, "C", transform=ax.transAxes, fontsize=11, va="top")
    bare(ax)

    # ---- D. confirmation --------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    if conf:
        # markers here denote MODEL, not screen status, so neither reuses the
        # triangle/square shapes the status legend has already claimed
        conf = sorted(conf, key=lambda r: -f(r, "delta_gpt_minus_claude"))
        y = np.arange(len(conf))
        for i, r in enumerate(conf):
            for key, mk, fc, off in (("gpt", "o", "black", -0.16),
                                     ("claude", "o", "white", 0.16)):
                v = f(r, f"{key}_match")
                lo, hi = f(r, f"{key}_ci_lo"), f(r, f"{key}_ci_hi")
                ax.plot([lo, hi], [i + off] * 2, lw=0.8, color="black",
                        solid_capstyle="butt")
                for x in (lo, hi):
                    ax.plot([x, x], [i + off - 0.1, i + off + 0.1], lw=0.8, color="black")
                ax.plot(v, i + off, marker=mk, ms=4.6, mfc=fc, mec="black",
                        mew=0.7, ls="none")
        ax.set_yticks(y)
        ax.set_yticklabels([r["stimulus_pair"] for r in conf], fontsize=6.5)
        ax.set_xlabel("dyadic match, confirmation sample (95% CI)")
        ax.set_xlim(0.40, 1.04)
        ax.axvline(0.5, lw=0.7, color="black", ls=(0, (2, 2)))
        ax.legend(handles=[
            plt.Line2D([], [], marker="o", mfc="black", mec="black", ls="none",
                       ms=4.6, label="GPT-5.4"),
            plt.Line2D([], [], marker="o", mfc="white", mec="black", ls="none",
                       ms=4.6, label="Claude Opus 4.8")],
            loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2, frameon=False,
            handletextpad=0.4, columnspacing=1.2, borderpad=0.2, fontsize=7)
    else:
        ax.text(0.5, 0.5, "confirmation not yet run", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    ax.text(-0.42, 1.04, "D", transform=ax.transAxes, fontsize=11, va="top")
    bare(ax)

    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.005), handletextpad=0.4, columnspacing=1.6)

    png = os.path.join(OUT, "targeted_stimulus_screen.png")
    fig.savefig(png, facecolor="white")
    fig.savefig(png.replace(".png", ".pdf"), facecolor="white")
    plt.close(fig)
    print(f"wrote {png}\nwrote {png.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
