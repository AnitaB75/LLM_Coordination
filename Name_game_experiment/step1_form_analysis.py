"""
STEP 1 analysis - form-choice design.

Reads the call-level CSV from `step1_form_choice.py` and writes:

  step1_form_summary_<ts>.csv      one row per model x stimulus pair
  step1_form_comparison_<ts>.csv   GPT and Claude side by side, one row per pair
  step1_form_match_<ts>.png/.pdf   dyadic match probability with 95% bootstrap CIs

Per model x stimulus pair (N valid trials, counts n_k):

  unbiased dyadic match   sum_k n_k (n_k - 1) / (N (N - 1))
      probability two DISTINCT trials agree; unbiased for the population rate,
      unlike the plug-in version
  plug-in match           sum_k p_k^2                    (reported as a check)
  Shannon entropy         -sum_k p_k log2 p_k            (bits)
  normalized entropy      H / log2(K) with K = 2, so equal to H in bits here
  bootstrap CI            5,000 resamples of the N rows, percentile 95%, fixed seed

Reported on BOTH codings:
  IDENTITY  - the underlying stimulus, after undoing the AB/BA counterbalance.
              This is the quantity of interest.
  POSITION  - the raw LEFT/RIGHT answer. High position agreement together with
              identity agreement near 0.5 means the model defaulted to a side
              rather than coordinating on form. The earlier probes in this
              project never tabulated this, so the distinction was invisible.

Run:  python3 step1_form_analysis.py [step1_form_calls_<ts>.csv]
      (defaults to the newest step1_form_calls_*.csv in step1_output/)
"""
import os, sys, csv, glob, math, random
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Academic figure style: Arial, plain black text, no decorative weight. Models are
# distinguished by marker (filled / open) as well as tone, so the figure survives
# greyscale printing.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 0,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "step1_output")
BOOT_B = 5000
BOOT_SEED = 20260804
PAIR_ORDER = ["proto", "inkblot", "rot", "noise", "mirror"]
# filled black vs open — greyscale-safe
MSTYLE = {"gpt":    dict(marker="o", mfc="black", mec="black"),
          "claude": dict(marker="s", mfc="white", mec="black")}
MNICE = {"gpt": "GPT-5.4", "claude": "Claude Opus 4.8"}


def unbiased_match(counts):
    """sum_k n_k (n_k - 1) / (N (N - 1)) — P(two distinct trials agree)."""
    n = sum(counts.values())
    if n < 2:
        return float("nan")
    return sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))


def plugin_match(counts):
    n = sum(counts.values())
    return sum((v / n) ** 2 for v in counts.values()) if n else float("nan")


def entropy_bits(counts):
    n = sum(counts.values())
    if not n:
        return float("nan")
    return -sum((v / n) * math.log2(v / n) for v in counts.values() if v)


def boot_ci(labels, seed, b=BOOT_B):
    if len(labels) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(labels)
    draws = sorted(unbiased_match(Counter(rng.choices(labels, k=n))) for _ in range(b))
    return draws[int(0.025 * b)], draws[int(0.975 * b)]


def modal(counts):
    if not counts:
        return "", float("nan"), False
    top = max(counts.values())
    winners = sorted(k for k, v in counts.items() if v == top)
    return winners[0], top / sum(counts.values()), len(winners) > 1


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(OUTDIR, path)
    else:
        found = sorted(glob.glob(os.path.join(OUTDIR, "step1_form_calls_*.csv")))
        if not found:
            sys.exit(f"no step1_form_calls_*.csv in {OUTDIR} — run step1_form_choice.py first")
        path = found[-1]
    ts = os.path.basename(path).replace("step1_form_calls_", "").replace(".csv", "")

    rows = list(csv.DictReader(open(path)))
    invalid = [r for r in rows if r["valid_response"] != "True"]
    seen, valid = set(), []
    for r in rows:
        if r["valid_response"] == "True" and r["run_id"] not in seen:
            seen.add(r["run_id"])
            valid.append(r)

    by_cell = defaultdict(list)
    for r in valid:
        by_cell[(r["model"], r["stimulus_pair"])].append(r)

    models = sorted({r["model"] for r in valid}, key=lambda m: m != "gpt")
    pairs = [p for p in PAIR_ORDER if any((m, p) in by_cell for m in models)]

    summary = []
    for model in models:
        for pair in pairs:
            cell = by_cell.get((model, pair), [])
            if not cell:
                continue
            ident = [r["choice_identity"] for r in cell]
            side = [r["response_side"] for r in cell]
            ic, sc = Counter(ident), Counter(side)
            opts = sorted({r["left_identity"] for r in cell} | {r["right_identity"] for r in cell})
            mo, ms, tie = modal(ic)
            lo, hi = boot_ci(ident, f"{BOOT_SEED}|{model}|{pair}|id")
            plo, phi = boot_ci(side, f"{BOOT_SEED}|{model}|{pair}|pos")
            summary.append(dict(
                model=model, model_id=cell[0]["model_id"], stimulus_pair=pair,
                n_valid=len(cell),
                option_1=opts[0], n_option_1=ic.get(opts[0], 0),
                p_option_1=round(ic.get(opts[0], 0) / len(cell), 4),
                option_2=opts[1] if len(opts) > 1 else "",
                n_option_2=ic.get(opts[1], 0) if len(opts) > 1 else 0,
                p_option_2=round(ic.get(opts[1], 0) / len(cell), 4) if len(opts) > 1 else "",
                modal_option=mo, modal_share=round(ms, 4), modal_tied=tie,
                match_unbiased=round(unbiased_match(ic), 4),
                match_unbiased_lo=round(lo, 4), match_unbiased_hi=round(hi, 4),
                match_plugin=round(plugin_match(ic), 4),
                entropy_bits=round(entropy_bits(ic), 4),
                entropy_normalized=round(entropy_bits(ic) / math.log2(2), 4),
                # position diagnostic
                p_left=round(sc.get("LEFT", 0) / len(cell), 4),
                match_position=round(unbiased_match(sc), 4),
                match_position_lo=round(plo, 4), match_position_hi=round(phi, 4),
                n_AB_variant=sum(1 for r in cell if r["variant"] == "AB"),
                n_BA_variant=sum(1 for r in cell if r["variant"] == "BA")))

    sum_path = os.path.join(OUTDIR, f"step1_form_summary_{ts}.csv")
    with open(sum_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    idx = {(r["model"], r["stimulus_pair"]): r for r in summary}
    comp_path = os.path.join(OUTDIR, f"step1_form_comparison_{ts}.csv")
    fields = ["stimulus_pair",
              "gpt_modal_option", "gpt_modal_share", "gpt_match", "gpt_match_lo",
              "gpt_match_hi", "gpt_entropy", "gpt_p_left",
              "claude_modal_option", "claude_modal_share", "claude_match", "claude_match_lo",
              "claude_match_hi", "claude_entropy", "claude_p_left",
              "match_difference_gpt_minus_claude", "same_modal_option"]
    comp = []
    for pair in pairs:
        g, c = idx.get(("gpt", pair)), idx.get(("claude", pair))
        if not (g and c):
            continue
        comp.append({
            "stimulus_pair": pair,
            "gpt_modal_option": g["modal_option"], "gpt_modal_share": g["modal_share"],
            "gpt_match": g["match_unbiased"], "gpt_match_lo": g["match_unbiased_lo"],
            "gpt_match_hi": g["match_unbiased_hi"], "gpt_entropy": g["entropy_bits"],
            "gpt_p_left": g["p_left"],
            "claude_modal_option": c["modal_option"], "claude_modal_share": c["modal_share"],
            "claude_match": c["match_unbiased"], "claude_match_lo": c["match_unbiased_lo"],
            "claude_match_hi": c["match_unbiased_hi"], "claude_entropy": c["entropy_bits"],
            "claude_p_left": c["p_left"],
            "match_difference_gpt_minus_claude": round(g["match_unbiased"] - c["match_unbiased"], 4),
            "same_modal_option": g["modal_option"] == c["modal_option"]})
    with open(comp_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(comp)

    # ---- plot ----
    order = sorted(pairs, key=lambda p: -max(idx[(m, p)]["match_unbiased"]
                                             for m in models if (m, p) in idx))
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=300)
    off = {models[0]: 0.14, models[-1]: -0.14} if len(models) > 1 else {models[0]: 0.0}
    for model in models:
        ys, xs, los, his = [], [], [], []
        for i, pair in enumerate(order):
            r = idx.get((model, pair))
            if not r:
                continue
            ys.append(i + off[model]); xs.append(r["match_unbiased"])
            los.append(r["match_unbiased"] - r["match_unbiased_lo"])
            his.append(r["match_unbiased_hi"] - r["match_unbiased"])
        ax.errorbar(xs, ys, xerr=[los, his], ms=4.5, mew=0.8, capsize=2.5,
                    elinewidth=0.8, capthick=0.8, ecolor="black", color="black",
                    linestyle="none", label=MNICE[model], **MSTYLE[model])
    ax.axvline(0.5, ls=(0, (4, 3)), lw=0.7, color="black", zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.05)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Within-model dyadic match probability")
    # "chance" sits just above the axes at the dashed line: x in data coords,
    # y in axes fraction, so it never collides with a data row.
    from matplotlib.transforms import blended_transform_factory
    ax.text(0.5, 1.01, "chance", transform=blended_transform_factory(ax.transData, ax.transAxes),
            ha="center", va="bottom", fontsize=8)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4,
              borderaxespad=0.6, labelspacing=0.35)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    png = os.path.join(OUTDIR, f"step1_form_match_{ts}.png")
    pdf = os.path.join(OUTDIR, f"step1_form_match_{ts}.pdf")
    fig.savefig(png); fig.savefig(pdf)
    plt.close(fig)

    # ---- console ----
    print(f"source: {os.path.basename(path)}   rows={len(rows)}  valid trials={len(valid)}  "
          f"unparseable first responses={len(invalid)}")
    print("\n=== identity coding (the quantity of interest) ===")
    print(f"{'model':8}{'pair':9}{'n':>4}{'modal':>13}{'share':>8}{'match':>8}{'95% CI':>16}{'H(bits)':>9}")
    for r in summary:
        ci = f"[{r['match_unbiased_lo']:.3f},{r['match_unbiased_hi']:.3f}]"
        print(f"{r['model']:8}{r['stimulus_pair']:9}{r['n_valid']:>4}{r['modal_option']:>13}"
              f"{r['modal_share']:>8.3f}{r['match_unbiased']:>8.3f}{ci:>16}{r['entropy_bits']:>9.3f}"
              + ("   TIED" if r["modal_tied"] else ""))

    print("\n=== position coding (side-default diagnostic) ===")
    print(f"{'model':8}{'pair':9}{'P(LEFT)':>9}{'match_pos':>11}   flag")
    for r in summary:
        flag = ""
        if r["match_position"] >= 0.75 and r["match_unbiased"] < 0.6:
            flag = "<-- defaulted to a SIDE, not a form"
        print(f"{r['model']:8}{r['stimulus_pair']:9}{r['p_left']:>9.3f}"
              f"{r['match_position']:>11.3f}   {flag}")

    print("\nwrote:")
    for p in (sum_path, comp_path, png, pdf):
        print(f"  {p}")


if __name__ == "__main__":
    main()
