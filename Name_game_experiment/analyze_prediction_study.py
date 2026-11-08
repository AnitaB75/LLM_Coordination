#!/usr/bin/env python3
"""
Analysis + figure for the 28-round rerun of Experiment 2 (prediction study).

Panel A: predicted number of the 5 teammates switching to M/N.
Panel B: the agent's own round-8 choice (proportion choosing M or N).
Both by framing (human / LLM) x horizon (14 control / 28 manipulation).
Error bars are 95% CI (normal approx for A, Wilson for B).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("academic")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FRAMINGS = ["human", "llm"]
FLABEL = {"human": "Teammates described as humans", "llm": "Teammates described as LLMs"}
STYLE = {"human": dict(marker="o", mfc="black", ls="-"),
         "llm":   dict(marker="s", mfc="white", ls="--")}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return (max(0, c-h), min(1, c+h))


def main():
    path = sorted(RESULTS.glob("prediction_study_28rounds_2*.csv"))[-1]
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {len(df)} agents\n")

    print("PREDICTIONS ABOUT THE 5 TEAMMATES (mean count)")
    print(f"{'horizon':>8s} {'framing':>8s} {'->M':>6s} {'->N':>6s} {'stay J':>7s} "
          f"{'->M/N':>7s} {'own switch':>11s} {'n':>4s}")
    for h in sorted(df.horizon.unique()):
        for fr in FRAMINGS:
            s = df[(df.horizon == h) & (df.framing == fr)]
            if not len(s):
                continue
            print(f"{h:>8d} {fr:>8s} {s.n_switch_M.mean():6.2f} {s.n_switch_N.mean():6.2f} "
                  f"{s.n_stay_J.mean():7.2f} {s.n_switch_MN.mean():7.2f} "
                  f"{s.own_switch.mean():10.0%} {len(s):5d}")

    print("\nOWN ROUND-8 CHOICE (counts)")
    for h in sorted(df.horizon.unique()):
        for fr in FRAMINGS:
            s = df[(df.horizon == h) & (df.framing == fr)]
            print(f"{h:>8d} {fr:>8s}  {dict(s.own_choice.value_counts())}")

    print("\nAGENTS PREDICTING >=1 TEAMMATE SWITCHES")
    for h in sorted(df.horizon.unique()):
        for fr in FRAMINGS:
            s = df[(df.horizon == h) & (df.framing == fr)]
            print(f"{h:>8d} {fr:>8s}  {(s.n_switch_MN >= 1).sum()}/{len(s)}")

    # ---- figure -----------------------------------------------------------
    horizons = sorted(df.horizon.unique())
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.9))

    ax = axes[0]
    for fr in FRAMINGS:
        m, lo, hi = [], [], []
        for h in horizons:
            s = df[(df.horizon == h) & (df.framing == fr)].n_switch_MN
            mu, se = s.mean(), s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0
            m.append(mu); lo.append(mu - 1.96*se); hi.append(mu + 1.96*se)
        ax.errorbar(horizons, m, yerr=[np.array(m)-np.array(lo), np.array(hi)-np.array(m)],
                    color="black", lw=0.9, ms=3.6, capsize=2.5, elinewidth=0.7,
                    label=FLABEL[fr], **STYLE[fr])
    ax.set_ylabel("Predicted no. of teammates\nswitching to M or N")
    ax.set_ylim(-0.1, 5)
    ax.set_yticks(range(6))

    ax = axes[1]
    for fr in FRAMINGS:
        m, lo, hi = [], [], []
        for h in horizons:
            s = df[(df.horizon == h) & (df.framing == fr)]
            k, n = int(s.own_switch.sum()), len(s)
            a, b = wilson(k, n)
            m.append(k/n); lo.append(a); hi.append(b)
        ax.errorbar(horizons, m, yerr=[np.array(m)-np.array(lo), np.array(hi)-np.array(m)],
                    color="black", lw=0.9, ms=3.6, capsize=2.5, elinewidth=0.7,
                    label=FLABEL[fr], **STYLE[fr])
    ax.set_ylabel("Own round-8 choice:\nproportion choosing M or N")
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    for ax, lab in zip(axes, ("A", "B")):
        ax.set_xlabel("Game length (rounds)")
        ax.set_xticks(horizons); ax.set_xlim(11, 31)
        ax.set_title(lab, loc="left", fontsize=9)
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()

    stamp = path.stem.replace("prediction_study_28rounds_", "")
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_prediction_study_{stamp}.{ext}"
        fig.savefig(out, dpi=300)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
