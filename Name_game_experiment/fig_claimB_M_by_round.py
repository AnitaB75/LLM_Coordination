#!/usr/bin/env python3
"""
Proportion of players choosing M, by round (1-14), for each model x partner framing.

Note this plots M ONLY, not M-or-N. That matters for one cell: Claude Opus 4.8
under Other LLM framing splits its switching 202 M / 36 N, so its M curve sits
below the M-or-N curve used for the primary DV. Every other cell is ~all M.

Cells and provenance are the same as the Claim B analysis.
"""

from pathlib import Path
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("academic")

HERE = Path(__file__).resolve().parent
OUT = HERE / "claimB_output" / "figures_academic"
OUT.mkdir(parents=True, exist_ok=True)

FRAMINGS = ["human", "other_llm", "self"]
FLAB = {"human": "Human", "other_llm": "Other LLM", "self": "Self"}
PANELS = [("gpt-5.4", "GPT-5.4"), ("Opus 4.8", "Claude Opus 4.8")]   # GPT left
STYLE = {
    "human":     dict(marker="s", mfc="white", ls="--", color="black"),
    "other_llm": dict(marker="^", mfc="0.55",  ls="-.", color="0.30"),
    "self":      dict(marker="o", mfc="black", ls="-",  color="black"),
}

SRC = {
    ("Opus 4.8", "hist"): HERE / "results/name_game_3cond_claude_20260718_142407.csv",
    ("gpt-5.4",  "hist"): HERE / "results/name_game_3cond_gpt_20260718_143234.csv",
    ("Opus 4.8", "new"):  Path(sorted(g for g in glob.glob(str(HERE/"results/name_game_otherllm_claude_*.csv")) if "parselog" not in g)[-1]),
    ("gpt-5.4",  "new"):  Path(sorted(g for g in glob.glob(str(HERE/"results/name_game_otherllm_gpt_*.csv")) if "parselog" not in g)[-1]),
}

def longify(path, model, conds):
    d = pd.read_csv(path)
    pc = [f"p{i}" for i in range(1, 7)]
    L = d.melt(id_vars=["trial", "partner_condition", "round"], value_vars=pc,
               var_name="player", value_name="choice")
    L["model"] = model
    return L[L.partner_condition.isin(conds)]

parts = []
for m in ["Opus 4.8", "gpt-5.4"]:
    parts.append(longify(SRC[(m, "hist")], m, ["self", "human"]))
    parts.append(longify(SRC[(m, "new")], m, ["other_llm"]))
L = pd.concat(parts, ignore_index=True)
L["is_M"] = (L.choice == "M").astype(int)

curve = (L.groupby(["model", "partner_condition", "round"]).is_M.mean()
           .rename("p_M").reset_index())
curve.to_csv(OUT / "M_by_round.csv", index=False)

print("Proportion choosing M, by round (rows = model x framing)")
print(curve.pivot_table(index=["model", "partner_condition"], columns="round",
                        values="p_M").round(3).to_string())
print("\nsanity: players per model x framing x round =",
      L.groupby(["model", "partner_condition", "round"]).size().unique())

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5.5, 3.0))
for ax, (m, disp), letter in zip(axes, PANELS, "AB"):
    for c in FRAMINGS:
        s = curve[(curve.model == m) & (curve.partner_condition == c)].sort_values("round")
        st = STYLE[c]
        ax.plot(s["round"], s.p_M, marker=st["marker"], mfc=st["mfc"], mec=st["color"],
                color=st["color"], ls=st["ls"], ms=3.5, lw=0.9, label=FLAB[c])
    ax.axvline(7.5, color="0.65", lw=0.6, ls=":")      # M becomes available at R8
    ax.set_xticks([1, 4, 8, 11, 14])
    ax.set_xlim(0.4, 14.6); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Round")
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
    ax.text(0.5, 1.06, disp, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
axes[0].set_ylabel("Proportion choosing M")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.09), handlelength=2.4)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"claimB_M_by_round.{ext}", dpi=300, bbox_inches="tight")
print(f"\n-> {OUT/'claimB_M_by_round.{png,pdf}'}")
