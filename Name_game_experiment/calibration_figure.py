"""
STEP 2 figure - predicted vs empirical coordination probability.

One panel per focal model. x = empirical dyadic match from the fresh ground-truth
trials; y = predicted probability. Self-estimates and auditor-estimates are drawn
as separate series and joined by a thin line for each stimulus, so the self-vs-
auditor gap on a given stimulus reads directly as the length of that segment.
The dashed diagonal is perfect calibration.

The shaded band below 0.5 marks the region no rater should enter: with two
options, two independent players match at 0.50 by chance, so the empirical
quantity is bounded below at ~0.50. Predictions under it are miscalibrated by
construction.

Academic style: Arial ~9pt, no titles inside the axes, 0.8pt black spines with
top/right removed, outward ticks. Series differ by MARKER and FILL as well as
tone so the figure survives greyscale. PDF written alongside the PNG.

Run:  python3 calibration_figure.py
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
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.edgecolor": "black", "text.color": "black",
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "calibration_output")
DISPLAY = {"gpt": "GPT-5.4", "claude": "Claude Opus 4.8"}


def main():
    p = os.path.join(OUT, "calibration_summary.csv")
    if not os.path.exists(p):
        sys.exit("no calibration_summary.csv - run calibration_analysis.py first")
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k in ("empirical", "self_mean", "auditor_mean", "self_advantage"):
            r[k] = float(r[k])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.30, wspace=0.10)

    for ax, focal in zip(axes, ("gpt", "claude")):
        rs = [r for r in rows if r["focal_model"] == focal]
        ax.axhspan(0.0, 0.5, color="0.92", zorder=0, lw=0)
        ax.plot([0.4, 1.02], [0.4, 1.02], ls=(0, (4, 3)), lw=0.7, color="black", zorder=1)

        for r in rs:
            ax.plot([r["empirical"]] * 2, [r["self_mean"], r["auditor_mean"]],
                    lw=0.6, color="0.55", zorder=2)
        ax.plot([r["empirical"] for r in rs], [r["self_mean"] for r in rs],
                marker="o", ms=5.2, mfc="black", mec="black", mew=0.7, ls="none",
                zorder=3, label="self-estimate")
        ax.plot([r["empirical"] for r in rs], [r["auditor_mean"] for r in rs],
                marker="s", ms=5.0, mfc="white", mec="black", mew=0.8, ls="none",
                zorder=3, label="auditor-estimate")

        # name the stimuli with the largest self-vs-auditor gap
        for r in sorted(rs, key=lambda r: -abs(r["self_mean"] - r["auditor_mean"]))[:4]:
            ymid = (r["self_mean"] + r["auditor_mean"]) / 2
            ax.annotate(r["stimulus"], (r["empirical"], ymid),
                        textcoords="offset points", xytext=(6, -2), fontsize=6.2)

        ax.set_xlim(0.42, 1.03)
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel("empirical dyadic match")
        ax.text(0.02, 0.97, f"focal: {DISPLAY[focal]}", transform=ax.transAxes,
                fontsize=8.5, va="top")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("predicted coordination probability")
    axes[0].text(0.44, 0.44, "chance floor", fontsize=6.4, color="0.35", va="center")

    fig.legend(handles=[
        plt.Line2D([], [], marker="o", mfc="black", mec="black", ls="none", ms=5.2,
                   label="self-estimate (focal model rates itself)"),
        plt.Line2D([], [], marker="s", mfc="white", mec="black", ls="none", ms=5.0,
                   label="auditor-estimate (other model rates the focal model)")],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.10),
        handletextpad=0.4, columnspacing=2.0)
    fig.text(0.5, 0.035,
             "Dashed line: perfect calibration. Shaded band: below the 0.50 chance "
             "floor for a two-option match.\nVertical segments join the two estimates "
             "for one stimulus; their length is the self-vs-auditor gap.",
             ha="center", fontsize=7, color="0.3", linespacing=1.5)

    png = os.path.join(OUT, "calibration_self_vs_auditor.png")
    fig.savefig(png, facecolor="white")
    fig.savefig(png.replace(".png", ".pdf"), facecolor="white")
    plt.close(fig)
    print(f"wrote {png}\nwrote {png.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
