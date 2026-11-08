#!/usr/bin/env python3
"""
Publication figures for Claim B. VISUALISATION ONLY.

Reads exclusively from claimB_output/ (player_level_DV.csv, descriptive_table.csv,
round_by_round.csv, planned_contrasts.csv). No experiment is rerun and no
statistic in the report is altered.

Games, not players, are the independent unit (ICC(player|game) = .92-.99), so
every panel plots GAME-level observations and every interval is the
game-clustered CI already used in the report.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("academic")

HERE = Path(__file__).resolve().parent
SRC = HERE / "claimB_output"
OUT = SRC / "figures_academic"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(11)

FRAMINGS = ["human", "other_llm", "self"]
XLAB = ["Human", "Other LLM", "Self"]
MODELS = ["Opus 4.8", "gpt-5.4"]
DISPLAY = {"Opus 4.8": "Claude Opus 4.8", "gpt-5.4": "GPT-5.4"}
MSTYLE = {                                    # line type + marker, never colour alone
    "Opus 4.8": dict(marker="o", mfc="black", ls="-",  color="black"),
    "gpt-5.4":  dict(marker="s", mfc="white", ls="--", color="black"),
}
FSTYLE = {
    "human":     dict(marker="s", mfc="white", ls="--", color="black"),
    "other_llm": dict(marker="^", mfc="0.55",  ls="-.", color="0.30"),
    "self":      dict(marker="o", mfc="black", ls="-",  color="black"),
}

dv = pd.read_csv(SRC / "player_level_DV.csv")
desc = pd.read_csv(SRC / "descriptive_table.csv")
rr = pd.read_csv(SRC / "round_by_round.csv")
con = pd.read_csv(SRC / "planned_contrasts.csv")

# Game-level observations: mean DV of the 6 players in each game.
game = (dv.groupby(["model", "partner_condition", "game_uid"], as_index=False)
          .DV.mean().rename(columns={"DV": "game_dv"}))

def cell(df, m, c):
    return df[(df.model == m) & (df.partner_condition == c)]

def dcell(m, c):
    return desc[(desc.model == m) & (desc.framing == c)].iloc[0]

def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  saved {stem}.png / .pdf")

# ------------------------------------------------------ technical checks ----
print("TECHNICAL CHECKS")
ok = True
for m in MODELS:
    for c in FRAMINGS:
        g = cell(game, m, c); d = dcell(m, c)
        n_games, n_players = len(g), len(cell(dv, m, c))
        # balanced 6 players/game => mean of game means == mean of player DVs
        agree = abs(g.game_dv.mean() - d["mean"]) < 1e-12
        if not (n_games == 10 and n_players == 60 and agree):
            ok = False
        print(f"  {DISPLAY[m]:17s} {c:10s} games={n_games:2d} players={n_players:3d} "
              f"game-mean={g.game_dv.mean():.4f} report-mean={d['mean']:.4f} "
              f"{'OK' if agree else 'MISMATCH'}")
print(f"  all cells 10 games / 60 players and means reproduce the report: {ok}")
print("  CIs plotted are the report's game-clustered bootstrap (ci_lo/ci_hi);")
print("  the naive player-level CI is never used.\n")

# =================================================== FIGURE 1  PRIMARY ======
print("FIGURE 1  primary result")
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5.5, 3.0))
for ax, m, letter in zip(axes, MODELS, "AB"):
    xs = np.arange(3)
    means = [dcell(m, c)["mean"] for c in FRAMINGS]
    lo = [dcell(m, c)["mean"] - dcell(m, c)["ci_lo"] for c in FRAMINGS]
    hi = [dcell(m, c)["ci_hi"] - dcell(m, c)["mean"] for c in FRAMINGS]
    for j, c in enumerate(FRAMINGS):                       # one point per GAME
        g = cell(game, m, c).game_dv.values
        ax.scatter(j + RNG.uniform(-0.11, 0.11, len(g)), g, s=9,
                   facecolors="none", edgecolors="0.55", linewidths=0.6, zorder=1)
    ax.plot(xs, means, ls="-", lw=0.7, color="black", zorder=2)
    ax.errorbar(xs, means, yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)],
                fmt="o", ms=4.5, mfc="black", mec="black", color="black",
                ls="none", lw=0.9, capsize=2.5, capthick=0.8, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(XLAB)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Partner framing")
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
    ax.text(0.5, 1.06, DISPLAY[m], transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
axes[0].set_ylabel("Proportion of rounds 8–14\nchoosing M or N")
fig.tight_layout()
save(fig, "claimB_primary")

# =============================================== FIGURE 2  INTERACTION ======
print("FIGURE 2  model x framing interaction")
fig, ax = plt.subplots(figsize=(3.6, 3.0))
xs = np.arange(3)
for m in MODELS:
    means = np.array([dcell(m, c)["mean"] for c in FRAMINGS])
    lo = means - np.array([dcell(m, c)["ci_lo"] for c in FRAMINGS])
    hi = np.array([dcell(m, c)["ci_hi"] for c in FRAMINGS]) - means
    st = MSTYLE[m]
    ax.errorbar(xs, means, yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)],
                marker=st["marker"], mfc=st["mfc"], mec=st["color"], color=st["color"],
                ls=st["ls"], ms=4.5, lw=1.0, capsize=2.5, capthick=0.8, elinewidth=0.8,
                label=DISPLAY[m])
ax.set_xticks(xs); ax.set_xticklabels(XLAB)
ax.set_xlim(-0.35, 2.35); ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("Partner framing")
ax.set_ylabel("Proportion of rounds 8–14\nchoosing M or N")
ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=2.4)
fig.tight_layout()
save(fig, "claimB_interaction")

# ============================================ FIGURE 3  ROUND DYNAMICS ======
print("FIGURE 3  round-by-round dynamics (descriptive)")
rounds = [str(r) for r in range(8, 15)]
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5.5, 2.9))
for ax, m, letter in zip(axes, MODELS, "AB"):
    for c in FRAMINGS:
        row = rr[(rr.model == m) & (rr.partner_condition == c)].iloc[0]
        st = FSTYLE[c]
        ax.plot(range(8, 15), [row[r] for r in rounds], marker=st["marker"],
                mfc=st["mfc"], mec=st["color"], color=st["color"], ls=st["ls"],
                ms=3.5, lw=0.9, label=XLAB[FRAMINGS.index(c)])
    ax.set_xticks(range(8, 15)); ax.set_xlim(7.7, 14.3); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Round")
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
    ax.text(0.5, 1.06, DISPLAY[m], transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
axes[0].set_ylabel("Proportion choosing M or N")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.09), handlelength=2.4)
fig.tight_layout()
save(fig, "claimB_round_dynamics")

# =============================================== FIGURE 4  CONTRASTS ========
print("FIGURE 4  game-level planned contrasts")
# Game-clustered bootstrap on the DIFFERENCE of game means -- the same resampling
# scheme the report used for its descriptive CIs, applied to the contrast. The
# confirmatory p-values remain the report's exact permutation tests.
NB = 20000
def boot_diff(m, a, b):
    ga = cell(game, m, a).game_dv.values
    gb = cell(game, m, b).game_dv.values
    d = np.empty(NB)
    for i in range(NB):
        d[i] = (RNG.choice(ga, len(ga), replace=True).mean()
                - RNG.choice(gb, len(gb), replace=True).mean())
    return np.percentile(d, 2.5), np.percentile(d, 97.5)

PAIRS = [("self", "human"), ("self", "other_llm"), ("other_llm", "human")]
PLAB = {("self","human"): "Self − Human",
        ("self","other_llm"): "Self − Other LLM",
        ("other_llm","human"): "Other LLM − Human"}
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(5.5, 2.2))
for ax, m, letter in zip(axes, MODELS, "AB"):
    ys = np.arange(len(PAIRS))[::-1]
    for y, (a, b) in zip(ys, PAIRS):
        row = con[(con.model == m) & (con.contrast == f"{a} - {b}")].iloc[0]
        lo, hi = boot_diff(m, a, b)
        est = row["gm_diff"]
        ax.plot([lo, hi], [y, y], color="black", lw=0.9, solid_capstyle="butt")
        for e in (lo, hi):
            ax.plot([e, e], [y - 0.11, y + 0.11], color="black", lw=0.8)
        ax.plot(est, y, marker="o", ms=4.5, mfc="black", mec="black", ls="none")
    ax.axvline(0, color="0.45", lw=0.7, ls=":")
    ax.set_yticks(ys); ax.set_ylim(-0.6, len(PAIRS) - 0.4)
    ax.set_yticklabels([PLAB[p] for p in PAIRS] if letter == "A" else [])
    ax.set_xlim(-0.45, 1.15); ax.set_xlabel("Difference in game-level mean")
    ax.text(0.5, 1.09, DISPLAY[m], transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
    ax.text(-0.02 if letter == "A" else -0.02, 1.09, letter, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10)
fig.tight_layout()
save(fig, "claimB_contrasts")

# ==================================================== PREVIEW MONTAGE =======
print("PREVIEW montage")
from PIL import Image
names = ["claimB_primary", "claimB_interaction", "claimB_round_dynamics", "claimB_contrasts"]
imgs = [Image.open(OUT / f"{n}.png").convert("RGB") for n in names]
W = max(i.width for i in imgs)
scaled = [i.resize((W, int(i.height * W / i.width)), Image.LANCZOS) for i in imgs]
pad = 40
H = sum(i.height for i in scaled) + pad * (len(scaled) + 1)
sheet = Image.new("RGB", (W + 2 * pad, H), "white")
y = pad
for i in scaled:
    sheet.paste(i, (pad, y)); y += i.height + pad
sheet.save(OUT / "claimB_figures_preview.png", dpi=(150, 150))
print(f"  saved claimB_figures_preview.png  ({sheet.width}x{sheet.height})")
print(f"\nAll figures -> {OUT}")
