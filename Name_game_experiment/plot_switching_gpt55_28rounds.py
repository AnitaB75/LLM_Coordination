#!/usr/bin/env python3
"""
Switching rate over trials, by condition, for the 28-round gpt-5.5 experiment.

SWITCHING = chose a different letter than in the previous round (defined for
rounds 2-28). Distinct from M/N adoption level: it counts transitions, so it
captures the failed-and-reverted attempts as well as the successful ones.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("academic")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

ORDER = ["human", "other_llm"]
LABELS = {"human": "Human framing", "other_llm": "Other-LLM framing"}
STYLE = {
    "human":     dict(marker="o", mfc="black", ls="-",  ms=3.4),
    "other_llm": dict(marker="s", mfc="white", ls="--", ms=3.4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="switching", choices=["switching", "adoption"],
                    help="switching = changed letter vs previous round; "
                         "adoption = chose M or N")
    args = ap.parse_args()
    path = sorted(RESULTS.glob("name_game_gpt55_28rounds_2*.csv"))[-1]
    df = pd.read_csv(path).sort_values(["condition", "group_key", "actor_id", "round"])
    df["prev"] = df.groupby(["condition", "group_key", "actor_id"]).letter.shift()
    df["switched"] = (df.letter != df["prev"]).where(df["prev"].notna())

    print("SWITCHING RATE (chose a different letter than previous round)\n")
    print(f"{'condition':22s} {'R2-7':>8s} {'R8-14':>8s} {'R15-28':>8s} {'overall':>9s}")
    for c in ORDER:
        s = df[df.condition == c].dropna(subset=["switched"])
        w = lambda lo, hi: s[(s["round"] >= lo) & (s["round"] <= hi)].switched.mean()
        print(f"{LABELS[c]:22s} {w(2,7):8.3f} {w(8,14):8.3f} {w(15,28):8.3f} "
              f"{s.switched.mean():9.3f}")

    col = "switched" if args.metric == "switching" else "switch_mn"
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for c in ORDER:
        s = df[df.condition == c]
        if args.metric == "switching":
            s = s.dropna(subset=["switched"])
        traj = s.groupby("round")[col].mean()
        ax.plot(traj.index, traj.values, color="black", lw=0.9,
                label=LABELS[c], **STYLE[c])

    ax.axvline(7.5, color="black", lw=0.6, ls=":")
    ax.axvline(14.5, color="0.6", lw=0.6, ls="--")   # where the 14-round game ended
    ax.set_xlabel("Round")
    ax.set_ylabel("Proportion switching letter" if args.metric == "switching"
                  else "Proportion choosing M or N")
    ax.set_xticks(range(2, 29, 2))
    ax.set_xlim(0.5 if args.metric == "adoption" else 1.5, 28.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()

    stamp = path.stem.replace("name_game_gpt55_28rounds_", "")
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_gpt55_28rounds_{args.metric}_{stamp}.{ext}"
        fig.savefig(out, dpi=300)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
