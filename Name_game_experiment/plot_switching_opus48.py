#!/usr/bin/env python3
"""
Switching rate over trials for the Opus 4.8 LLM-only experiment.

SWITCHING is defined per actor as choosing a DIFFERENT letter than in the
previous round, so it is defined for rounds 2-14 and requires stable actor
identity across rounds (actor_id_stable). This is the transition rate, and is a
distinct quantity from the M/N adoption level plotted in
analyze_opus48_llmonly.py -- switching counts any change of letter, including
M->N churn and returns to J.

Benchmarks from the combined human/LLM dataset are restricted to
actor_id_stable == True rows; the 728 unstable mixed-group LLM rows are dropped
because a slot that does not track the same agent across rounds would
manufacture spurious switches.

Academic style: series separated by marker AND linetype for greyscale.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("academic")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
COMBINED = Path.home() / "Downloads" / "name_game_combined_human_llm_mixed.csv"

ORDER = ["static_adaptive", "reveal8_adaptive", "static_nothink"]
LABELS = {
    "static_adaptive":  "Opus 4.8 — full schedule, adaptive",
    "reveal8_adaptive": "Opus 4.8 — revealed at R8, adaptive",
    "static_nothink":   "Opus 4.8 — full schedule, no thinking",
}
STYLE = {
    "static_adaptive":  dict(marker="o", mfc="black", ls="-",  ms=3.6),
    # reveal8 and static_nothink are both exactly 0.000 at every round, so they
    # overplot. Different marker sizes keep both visible without displacing
    # either series off its true value.
    "reveal8_adaptive": dict(marker="s", mfc="white", ls="--", ms=5.4),
    "static_nothink":   dict(marker="^", mfc="0.6",   ls="-.", ms=2.8),
}


def add_switch(df, keys):
    """Per-actor round-to-round switch indicator (NaN at round 1)."""
    df = df.sort_values(keys + ["round"]).copy()
    df["prev"] = df.groupby(keys).letter.shift()
    df["switched"] = (df.letter != df["prev"]).where(df["prev"].notna())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions-only", action="store_true",
                    help="plot only the three Opus 4.8 conditions (no human/GPT benchmarks)")
    args = ap.parse_args()
    path = sorted(RESULTS.glob("name_game_opus48_llmonly_2*.csv"))[-1]
    df = add_switch(pd.read_csv(path), ["condition", "group_key", "actor_id"])

    print("SWITCHING RATE (chose a different letter than previous round)\n")
    print(f"{'condition':38s} {'rounds 2-7':>11s} {'rounds 8-14':>12s} {'overall':>9s}")
    for c in ORDER:
        s = df[df.condition == c].dropna(subset=["switched"])
        pre = s[s["round"] <= 7].switched.mean()
        post = s[s["round"] >= 8].switched.mean()
        print(f"{LABELS[c]:38s} {pre:11.3f} {post:12.3f} {s.switched.mean():9.3f}")

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    for c in ORDER:
        s = df[df.condition == c].dropna(subset=["switched"])
        traj = s.groupby("round").switched.mean()
        ax.plot(traj.index, traj.values, color="black", lw=0.9,
                label=LABELS[c], **STYLE[c])

    if COMBINED.exists() and not args.conditions_only:
        comb = pd.read_csv(COMBINED)
        comb = comb[comb.actor_id_stable == True]  # noqa: E712
        comb = add_switch(comb, ["group_key", "actor_id"])
        for actor, lab, st in [
            ("Human", "Human (oTree, revealed at R8)", dict(marker="D", mfc="white", ls=":")),
            ("LLM",   "GPT-5.5 (oTree, full schedule)", dict(marker="v", mfc="0.85", ls=":")),
        ]:
            s = comb[comb.actor_type == actor].dropna(subset=["switched"])
            traj = s.groupby("round").switched.mean()
            ax.plot(traj.index, traj.values, color="0.45", ms=3.2, lw=0.8,
                    label=lab, **st)

    ax.axvline(7.5, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Round")
    ax.set_ylabel("Proportion switching letter")
    ax.set_xticks(range(2, 15))
    ax.set_xlim(1.5, 14.5)
    ax.set_ylim(-0.02, 1.02)   # full 0-1 proportion scale
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=7 if args.conditions_only else 6.5,
              loc="upper left" if args.conditions_only else "upper right")
    fig.tight_layout()

    stamp = path.stem.replace("name_game_opus48_llmonly_", "")
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_opus48_switching{'_conditions' if args.conditions_only else ''}_{stamp}.{ext}"
        fig.savefig(out, dpi=300)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
