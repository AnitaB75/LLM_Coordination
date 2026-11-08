#!/usr/bin/env python3
"""
Analysis for Name_Game_gpt55_28rounds.py.

THE TEST: if entrenchment is a payback-horizon calculation, post-change M/N
adoption should be low in rounds 8-14 (matching the 14-round game, where little
horizon remains) and substantially higher in rounds 15-28. If it is
convention-lock, both windows stay flat.

Reports adoption per condition split by window, group-level tipping counts, and
the round of first defection. Plots the 28-round trajectory with the existing
14-round gpt-5.5 benchmark overlaid.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("academic")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
COMBINED = Path.home() / "Downloads" / "name_game_combined_human_llm_mixed.csv"

ORDER = ["human", "other_llm"]
LABELS = {"human": "Human framing", "other_llm": "Other-LLM framing"}
STYLE = {
    "human":     dict(marker="o", mfc="black", ls="-",  ms=3.2),
    "other_llm": dict(marker="s", mfc="white", ls="--", ms=3.2),
}


def main():
    path = sorted(RESULTS.glob("name_game_gpt55_28rounds_2*.csv"))[-1]
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {len(df)} rows, {df.group_key.nunique()} groups\n")

    print("M/N ADOPTION BY WINDOW  (the payback-horizon test)")
    print(f"{'condition':22s} {'R8-14':>8s} {'R15-28':>8s} {'all post':>9s} {'n post':>7s}")
    for c in ORDER:
        s = df[(df.condition == c) & (df.post_change == 1)]
        if not len(s):
            continue
        early = s[s["round"] < 15].switch_mn.mean()
        late = s[s["round"] >= 15].switch_mn.mean()
        print(f"{LABELS[c]:22s} {early:8.3f} {late:8.3f} {s.switch_mn.mean():9.3f} {len(s):7d}")

    print("\nGROUP-LEVEL OUTCOME (modal letter, final round)")
    last = df["round"].max()
    for c in ORDER:
        s = df[(df.condition == c) & (df["round"] == last)]
        if not len(s):
            continue
        modes = s.groupby("group_key").letter.agg(lambda x: x.mode().iat[0])
        unanimous = s.groupby("group_key").letter.nunique().eq(1).sum()
        print(f"{LABELS[c]:22s} {dict(modes.value_counts())}  unanimous={unanimous}/{modes.size}")

    print("\nFIRST M/N DEFECTION PER GROUP (round; '-' = never)")
    for c in ORDER:
        s = df[df.condition == c]
        out = []
        for g, sub in s.groupby("group_key"):
            d = sub[sub.switch_mn == 1]
            out.append(str(int(d["round"].min())) if len(d) else "-")
        print(f"{LABELS[c]:22s} {', '.join(out)}")

    # ---- trajectory ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for c in ORDER:
        s = df[df.condition == c]
        if not len(s):
            continue
        traj = s.groupby("round").switch_mn.mean()
        ax.plot(traj.index, traj.values, color="black", lw=0.9,
                label=f"{LABELS[c]} (28 rounds)", **STYLE[c])

    if COMBINED.exists():
        comb = pd.read_csv(COMBINED)
        comb = comb[comb.actor_type == "LLM"]
        comb["mn"] = comb.choice_category.isin(["M", "N"]).astype(int)
        traj = comb.groupby("round").mn.mean()
        ax.plot(traj.index, traj.values, color="0.5", lw=0.8, ms=3.0,
                marker="v", mfc="0.85", ls=":", label="gpt-5.5, 14-round game")

    ax.axvline(7.5, color="black", lw=0.6, ls=":")
    ax.axvline(14.5, color="0.6", lw=0.6, ls="--")   # end of the original game
    ax.set_xlabel("Round")
    ax.set_ylabel("Proportion choosing M or N")
    ax.set_xticks(range(2, 29, 2))
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()

    stamp = path.stem.replace("name_game_gpt55_28rounds_", "")
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_gpt55_28rounds_{stamp}.{ext}"
        fig.savefig(out, dpi=300)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
