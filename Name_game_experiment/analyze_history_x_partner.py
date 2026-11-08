#!/usr/bin/env python3
"""
Analysis for the 2 (history) x 4 (partner framing) no-feedback name game.

PRIMARY COMPARISON
  Proportion choosing M or N at the FIRST round where M and N are available --
  round 8 under `history`, round 1 under `no_history` -- split by partner
  framing, with 95% CIs. This is the round at which the two history conditions
  face an identical decision (J=10 vs M=N=15, no feedback ever received); the
  only difference is whether seven J-only rounds preceded it.

  Also reported: the self-minus-human gap within each history condition, and
  the difference of those two gaps (the history x framing interaction), which
  is what the prediction is actually about.

SECONDARY
  Full round-by-round P(M or N) curves, by framing and history condition.

Sampling unit: a game (6 players x 14 rounds). Because the game has no feedback
and no communication, the six players never influence one another, so binomial
CIs over players are defensible; the interaction CI is nevertheless bootstrapped
over GAMES, which is the conservative choice.

Usage:
  python3 analyze_history_x_partner.py [results/name_game_history_x_partner_*.csv]
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("academic")

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "history_x_partner_output"

PARTNERS = ["self", "human", "other_llm", "neutral"]
PARTNER_LABELS = {"self": "Self", "human": "Human",
                  "other_llm": "Other LLM", "neutral": "Neutral"}
HISTORIES = ["history", "no_history"]
HISTORY_LABELS = {"history": "History (J-only R1-7)", "no_history": "No history"}
FIRST_MN_ROUND = {"history": 8, "no_history": 1}

# Marker / linetype carry the series, so the figures survive greyscale.
HIST_STYLE = {
    "history":    dict(marker="o", mfc="black", ls="-",  color="black"),
    "no_history": dict(marker="s", mfc="white", ls="--", color="black"),
}
PARTNER_STYLE = {
    "self":      dict(marker="o", mfc="black", ls="-",   color="black"),
    "human":     dict(marker="s", mfc="white", ls="--",  color="black"),
    "other_llm": dict(marker="^", mfc="0.55",  ls="-.",  color="0.35"),
    "neutral":   dict(marker="D", mfc="white", ls=":",   color="0.55"),
}

RNG = np.random.default_rng(20260909)
N_BOOT = 10000


# ------------------------------------------------------------ statistics ----

def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float, float]:
    """Wilson score interval. Returns (p, lo, hi)."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def boot_game_diff(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[float, float, float]:
    """
    Difference in P(M or N) between two sets of rows, bootstrapped by resampling
    whole GAMES within each set. Returns (diff, lo, hi).
    """
    def games(df):
        return [g["mn"].to_numpy() for _, g in df.groupby("trial", sort=True)]

    ga, gb = games(df_a), games(df_b)
    if not ga or not gb:
        return np.nan, np.nan, np.nan
    obs = df_a["mn"].mean() - df_b["mn"].mean()
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        a = np.concatenate([ga[j] for j in RNG.integers(0, len(ga), len(ga))])
        b = np.concatenate([gb[j] for j in RNG.integers(0, len(gb), len(gb))])
        draws[i] = a.mean() - b.mean()
    return obs, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ------------------------------------------------------------------ load ----

def load(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    n_all = len(df)
    if "parse_status" in df.columns:
        bad = int((df["parse_status"] == "failed").sum())
        if bad:
            print(f"Excluding {bad} row(s) with parse_status=failed.")
        df = df[df["parse_status"] != "failed"]
    df = df[df["choice"].notna() & (df["choice"] != "")]
    if len(df) != n_all:
        print(f"Kept {len(df)}/{n_all} rows.")
    df["mn"] = df["choice"].isin(["M", "N"]).astype(int)
    df["m"] = (df["choice"] == "M").astype(int)
    df["config"] = df["model"] + " (" + df["reasoning_effort"].astype(str) + ")"
    return df


# --------------------------------------------------------------- primary ----

def primary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, dcfg in df.groupby("config", sort=True):
        for hist in HISTORIES:
            sub_h = dcfg[(dcfg["history_condition"] == hist)
                         & (dcfg["round"] == FIRST_MN_ROUND[hist])]
            for partner in PARTNERS:
                sub = sub_h[sub_h["partner_condition"] == partner]
                n = len(sub)
                k = int(sub["mn"].sum())
                p, lo, hi = wilson(k, n)
                rows.append({
                    "config": config, "history_condition": hist,
                    "partner_condition": partner,
                    "first_mn_round": FIRST_MN_ROUND[hist],
                    "n_choices": n, "n_games": sub["trial"].nunique(),
                    "k_mn": k, "p_mn": p, "ci_lo": lo, "ci_hi": hi,
                    "p_m": sub["m"].mean() if n else np.nan,
                })
    return pd.DataFrame(rows)


def contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """self - human within each history condition, and the interaction."""
    rows = []
    for config, dcfg in df.groupby("config", sort=True):
        gaps = {}
        for hist in HISTORIES:
            sub = dcfg[(dcfg["history_condition"] == hist)
                       & (dcfg["round"] == FIRST_MN_ROUND[hist])]
            a = sub[sub["partner_condition"] == "self"]
            b = sub[sub["partner_condition"] == "human"]
            d, lo, hi = boot_game_diff(a, b)
            gaps[hist] = (a, b)
            rows.append({"config": config, "contrast": f"self - human [{hist}]",
                         "estimate": d, "ci_lo": lo, "ci_hi": hi})
        # interaction: (self-human | history) - (self-human | no_history)
        if all(h in gaps for h in HISTORIES):
            def gap_draws(pair):
                a, b = pair
                ga = [g["mn"].to_numpy() for _, g in a.groupby("trial", sort=True)]
                gb = [g["mn"].to_numpy() for _, g in b.groupby("trial", sort=True)]
                if not ga or not gb:
                    return None
                out = np.empty(N_BOOT)
                for i in range(N_BOOT):
                    x = np.concatenate([ga[j] for j in RNG.integers(0, len(ga), len(ga))])
                    y = np.concatenate([gb[j] for j in RNG.integers(0, len(gb), len(gb))])
                    out[i] = x.mean() - y.mean()
                return out

            dh, dn = gap_draws(gaps["history"]), gap_draws(gaps["no_history"])
            if dh is not None and dn is not None:
                a, b = gaps["history"]
                c, d_ = gaps["no_history"]
                obs = ((a["mn"].mean() - b["mn"].mean())
                       - (c["mn"].mean() - d_["mn"].mean()))
                draws = dh - dn
                rows.append({
                    "config": config,
                    "contrast": "interaction (self-human): history - no_history",
                    "estimate": obs,
                    "ci_lo": float(np.percentile(draws, 2.5)),
                    "ci_hi": float(np.percentile(draws, 97.5)),
                })
    return pd.DataFrame(rows)


def curve_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, hist, partner, rnd), sub in df.groupby(
            ["config", "history_condition", "partner_condition", "round"], sort=True):
        p, lo, hi = wilson(int(sub["mn"].sum()), len(sub))
        rows.append({"config": config, "history_condition": hist,
                     "partner_condition": partner, "round": rnd,
                     "n_choices": len(sub), "p_mn": p, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- figures ----

def save(fig, stem: str):
    OUTDIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  -> {OUTDIR / (stem + '.{pdf,png}')}")


def fig_primary(tab: pd.DataFrame):
    configs = sorted(tab["config"].unique())
    fig, axes = plt.subplots(1, len(configs), sharey=True,
                             figsize=(2.0 * len(configs) + 1.4, 3.2))
    axes = np.atleast_1d(axes)
    x = np.arange(len(PARTNERS))
    for ax, config in zip(axes, configs):
        sub = tab[tab["config"] == config]
        for i, hist in enumerate(HISTORIES):
            s = sub[sub["history_condition"] == hist].set_index("partner_condition")
            s = s.reindex(PARTNERS)
            off = (i - 0.5) * 0.16
            st = HIST_STYLE[hist]
            # Clip at 0: at p exactly 0 or 1 the Wilson bound can land a float
            # epsilon the wrong side of the point estimate.
            lo_err = np.clip(s["p_mn"] - s["ci_lo"], 0, None)
            hi_err = np.clip(s["ci_hi"] - s["p_mn"], 0, None)
            ax.errorbar(
                x + off, s["p_mn"], yerr=[lo_err, hi_err],
                fmt=st["marker"], mfc=st["mfc"], mec=st["color"], color=st["color"],
                ls="none", ms=4, lw=0.8, capsize=2.5, capthick=0.8,
                label=HISTORY_LABELS[hist],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([PARTNER_LABELS[p] for p in PARTNERS], rotation=30, ha="right")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.5, len(PARTNERS) - 0.5)
        ax.text(0.5, 1.02, config, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("P(chose M or N)")
    for ax in axes:
        ax.set_xlabel("Partner framing")
    # Legend below the axes: several cells sit at 1.0, so the upper corners are full.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig1_first_available_round")
    return fig


def fig_curves(curves: pd.DataFrame):
    configs = sorted(curves["config"].unique())
    n_rows, n_cols = len(configs), len(HISTORIES)
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True,
                             figsize=(5.5, 1.7 * n_rows + 0.9), squeeze=False)
    for r, config in enumerate(configs):
        for c, hist in enumerate(HISTORIES):
            ax = axes[r][c]
            sub = curves[(curves["config"] == config)
                         & (curves["history_condition"] == hist)]
            for partner in PARTNERS:
                s = sub[sub["partner_condition"] == partner].sort_values("round")
                st = PARTNER_STYLE[partner]
                ax.plot(s["round"], s["p_mn"], marker=st["marker"], mfc=st["mfc"],
                        mec=st["color"], color=st["color"], ls=st["ls"],
                        ms=3, lw=0.8, label=PARTNER_LABELS[partner])
            if hist == "history":
                ax.axvline(8, color="0.6", lw=0.6, ls=":")
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks([1, 4, 8, 11, 14])
            if r == 0:
                ax.text(0.5, 1.03, HISTORY_LABELS[hist], transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=8)
            if c == n_cols - 1:
                ax.text(1.02, 0.5, config, transform=ax.transAxes,
                        ha="left", va="center", fontsize=7, rotation=90)
    for ax in axes[-1]:
        ax.set_xlabel("Round")
    for row in axes:
        row[0].set_ylabel("P(M or N)")
    axes[0][0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_round_by_round")
    return fig


# ------------------------------------------------------------------ main ----

def main():
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob(str(HERE / "results" / "name_game_history_x_partner_*.csv")))
        paths = [p for p in paths if "parse_failures" not in p]
    if not paths:
        raise SystemExit("No results CSV found. Pass one as an argument.")
    print("Reading:", *paths, sep="\n  ")

    df = load(paths)
    OUTDIR.mkdir(exist_ok=True)

    tab = primary_table(df)
    curves = curve_table(df)
    con = contrasts(df)

    tab.to_csv(OUTDIR / "primary_first_available_round.csv", index=False)
    curves.to_csv(OUTDIR / "round_by_round.csv", index=False)
    con.to_csv(OUTDIR / "contrasts.csv", index=False)

    print("\nPRIMARY: P(M or N) at the first round M/N are available "
          "(R8 history, R1 no_history), 95% Wilson CI")
    for config, sub in tab.groupby("config", sort=True):
        print(f"\n  {config}")
        print(f"    {'framing':10s} {'history':>22s}   {'no_history':>22s}")
        for partner in PARTNERS:
            cells = []
            for hist in HISTORIES:
                r = sub[(sub["partner_condition"] == partner)
                        & (sub["history_condition"] == hist)]
                if len(r):
                    r = r.iloc[0]
                    cells.append(f"{r['p_mn']:.3f} [{r['ci_lo']:.2f},{r['ci_hi']:.2f}] "
                                 f"n={int(r['n_choices'])}")
                else:
                    cells.append("--")
            print(f"    {PARTNER_LABELS[partner]:10s} {cells[0]:>22s}   {cells[1]:>22s}")

    print("\nCONTRASTS (bootstrap over games, 95% CI)")
    for _, r in con.iterrows():
        print(f"  {r['config']:26s} {r['contrast']:44s} "
              f"{r['estimate']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")

    print("\nFigures:")
    fig_primary(tab)
    fig_curves(curves)
    print(f"\nTables -> {OUTDIR}")


if __name__ == "__main__":
    main()
