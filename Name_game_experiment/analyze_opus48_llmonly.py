#!/usr/bin/env python3
"""
Analysis for Name_Game_opus48_llmonly.py.

Reads the newest name_game_opus48_llmonly_*.csv and reports, per condition:
  - post-change (rounds 8-14) M/N adoption rate
  - round-8 switch rate (the immediate response to the payoff change)
  - group-level convergence (did the group settle on M/N?)
and plots the M/N trajectory across rounds 1-14, with the human and LLM
benchmarks from the existing combined dataset overlaid.

Figure follows the academic style (~/.matplotlib/stylelib/academic.mplstyle):
series distinguished by marker AND linetype so it survives greyscale.
"""

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
    "static_adaptive":  dict(marker="o", mfc="black",  ls="-"),
    "reveal8_adaptive": dict(marker="s", mfc="white",  ls="--"),
    "static_nothink":   dict(marker="^", mfc="0.6",    ls="-."),
}


def newest(pattern):
    files = sorted(RESULTS.glob(pattern))
    if not files:
        raise SystemExit(f"no file matching {pattern} in {RESULTS}")
    return files[-1]


def main():
    path = newest("name_game_opus48_llmonly_2*.csv")
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {len(df)} rows, "
          f"{df.group_key.nunique()} groups, {df.condition.nunique()} conditions\n")

    post = df[df.post_change == 1]
    print("POST-CHANGE (rounds 8-14) M/N ADOPTION")
    print(f"{'condition':36s} {'rate':>7s} {'n':>6s} {'groups':>7s}")
    for c in ORDER:
        s = post[post.condition == c]
        if len(s):
            print(f"{LABELS[c]:36s} {s.switch_mn.mean():7.3f} {len(s):6d} {s.group_key.nunique():7d}")

    print("\nROUND-8 SWITCH RATE (immediate response)")
    r8 = df[df["round"] == 8]
    for c in ORDER:
        s = r8[r8.condition == c]
        if len(s):
            print(f"{LABELS[c]:36s} {s.switch_mn.mean():7.3f} {len(s):6d}")

    print("\nGROUP-LEVEL OUTCOME (modal letter in final round 14)")
    for c in ORDER:
        s = df[(df.condition == c) & (df["round"] == 14)]
        if not len(s):
            continue
        modes = s.groupby("group_key").letter.agg(lambda x: x.mode().iat[0])
        counts = modes.value_counts().to_dict()
        unanimous = s.groupby("group_key").letter.nunique().eq(1).sum()
        print(f"{LABELS[c]:36s} {counts}  unanimous={unanimous}/{modes.size}")

    # ---- trajectory figure -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    for c in ORDER:
        s = df[df.condition == c]
        if not len(s):
            continue
        traj = s.groupby("round").switch_mn.mean()
        ax.plot(traj.index, traj.values, color="black", ms=3.4, lw=0.9,
                label=LABELS[c], **STYLE[c])

    # benchmarks from the existing combined dataset
    if COMBINED.exists():
        comb = pd.read_csv(COMBINED)
        comb["mn"] = comb.choice_category.isin(["M", "N"]).astype(int)
        for actor, lab, st in [
            ("Human", "Human (oTree, revealed at R8)", dict(marker="D", mfc="white", ls=":")),
            ("LLM",   "GPT-5.5 (oTree, full schedule)", dict(marker="v", mfc="0.85", ls=":")),
        ]:
            s = comb[comb.actor_type == actor]
            traj = s.groupby("round").mn.mean()
            ax.plot(traj.index, traj.values, color="0.45", ms=3.2, lw=0.8,
                    label=lab, **st)

    ax.axvline(7.5, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Round")
    ax.set_ylabel("Proportion choosing M or N")
    ax.set_xticks(range(1, 15))
    ax.set_ylim(-0.03, 1.0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()

    stamp = path.stem.replace("name_game_opus48_llmonly_", "")
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_opus48_llmonly_{stamp}.{ext}"
        fig.savefig(out, dpi=300)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
