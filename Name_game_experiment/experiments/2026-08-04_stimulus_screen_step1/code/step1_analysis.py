"""
STEP 1 analysis - choice distribution and within-model coordination rate.

Reads the call-level CSV from `step1_choice_distribution.py` and writes:

  step1_summary_<ts>.csv      one row per model x stimulus pair
  step1_comparison_<ts>.csv   GPT and Claude side by side, one row per pair
  step1_match_<ts>.png/.pdf   dyadic match probability with 95% bootstrap CIs

Quantities, per model x stimulus pair (N valid trials, counts n_k over identities):

  unbiased dyadic match   sum_k n_k (n_k - 1) / (N (N - 1))
      the probability two DISTINCT trials from this cell agree; unbiased for the
      population match rate, unlike the plug-in version
  plug-in match           sum_k p_k^2                    (reported as a check)
  Shannon entropy         -sum_k p_k log2 p_k            (bits)
  normalized entropy      H / log2(K) with K = 2, so identical to H in bits here
  bootstrap CI            5,000 resamples of the N rows, percentile 95%, fixed seed

Both codings are reported:
  IDENTITY  - the underlying stimulus, after undoing the A/B counterbalance. This is
              the quantity of interest.
  POSITION  - the raw A/B label. High position agreement together with identity
              agreement near 0.5 means the model coordinated on the LABEL, not the
              stimulus (see the position-confound note in the collection script).

Run:  python3 step1_analysis.py [step1_calls_<ts>.csv]
      (defaults to the newest step1_calls_*.csv in step1_output/)
"""
import os, sys, csv, glob, math, random
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "step1_output")
BOOT_B = 5000
BOOT_SEED = 20260804
PAIR_ORDER = ["proto", "nd2", "inkblot", "rot", "noise", "mirror"]
MCOL = {"gpt": "#eb6834", "claude": "#2a78d6"}
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
    h = 0.0
    for v in counts.values():
        if v:
            p = v / n
            h -= p * math.log2(p)
    return h


def boot_ci(labels, seed, b=BOOT_B):
    """Percentile 95% CI for the unbiased match probability."""
    if len(labels) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(labels)
    draws = []
    for _ in range(b):
        draws.append(unbiased_match(Counter(rng.choices(labels, k=n))))
    draws.sort()
    return draws[int(0.025 * b)], draws[int(0.975 * b)]


def modal(counts):
    if not counts:
        return "", float("nan"), False
    top = max(counts.values())
    winners = sorted(k for k, v in counts.items() if v == top)
    n = sum(counts.values())
    return winners[0], top / n, len(winners) > 1


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(OUTDIR, path)
    else:
        found = sorted(glob.glob(os.path.join(OUTDIR, "step1_calls_*.csv")))
        if not found:
            sys.exit(f"no step1_calls_*.csv in {OUTDIR} — run step1_choice_distribution.py first")
        path = found[-1]
    ts = os.path.basename(path).replace("step1_calls_", "").replace(".csv", "")

    rows = list(csv.DictReader(open(path)))
    valid = [r for r in rows if r["valid_response"] == "True"]
    invalid = [r for r in rows if r["valid_response"] != "True"]
    # one trial contributes one valid row (the final attempt); guard against dupes
    seen, kept = set(), []
    for r in valid:
        if r["run_id"] not in seen:
            seen.add(r["run_id"])
            kept.append(r)
    valid = kept

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
            pos = [r["response_position"] for r in cell]
            ic, pc = Counter(ident), Counter(pos)
            opts = sorted({r["option_A_identity"] for r in cell} | {r["option_B_identity"] for r in cell})
            mo, ms, tie = modal(ic)
            lo, hi = boot_ci(ident, f"{BOOT_SEED}|{model}|{pair}")
            plo, phi = boot_ci(pos, f"{BOOT_SEED}|{model}|{pair}|pos")
            row = dict(
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
                # position-confound diagnostic
                match_unbiased_position=round(unbiased_match(pc), 4),
                match_position_lo=round(plo, 4), match_position_hi=round(phi, 4),
                p_position_A=round(pc.get("A", 0) / len(cell), 4),
                n_AB_variant=sum(1 for r in cell if r["variant"] == "AB"),
                n_BA_variant=sum(1 for r in cell if r["variant"] == "BA"))
            summary.append(row)

    sum_path = os.path.join(OUTDIR, f"step1_summary_{ts}.csv")
    with open(sum_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # ---- side-by-side comparison ----
    idx = {(r["model"], r["stimulus_pair"]): r for r in summary}
    comp_path = os.path.join(OUTDIR, f"step1_comparison_{ts}.csv")
    fields = ["stimulus_pair", "gpt_modal_option", "gpt_modal_share", "gpt_match",
              "gpt_match_lo", "gpt_match_hi", "gpt_entropy",
              "claude_modal_option", "claude_modal_share", "claude_match",
              "claude_match_lo", "claude_match_hi", "claude_entropy",
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
            "claude_modal_option": c["modal_option"], "claude_modal_share": c["modal_share"],
            "claude_match": c["match_unbiased"], "claude_match_lo": c["match_unbiased_lo"],
            "claude_match_hi": c["match_unbiased_hi"], "claude_entropy": c["entropy_bits"],
            "match_difference_gpt_minus_claude": round(g["match_unbiased"] - c["match_unbiased"], 4),
            "same_modal_option": g["modal_option"] == c["modal_option"]})
    with open(comp_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(comp)

    # ---- plot ----
    order = sorted(pairs, key=lambda p: -max(idx[(m, p)]["match_unbiased"]
                                             for m in models if (m, p) in idx))
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=190)
    off = {models[0]: 0.16, models[-1]: -0.16} if len(models) > 1 else {models[0]: 0.0}
    for model in models:
        ys, xs, los, his = [], [], [], []
        for i, pair in enumerate(order):
            r = idx.get((model, pair))
            if not r:
                continue
            ys.append(i + off[model]); xs.append(r["match_unbiased"])
            los.append(r["match_unbiased"] - r["match_unbiased_lo"])
            his.append(r["match_unbiased_hi"] - r["match_unbiased"])
        ax.errorbar(xs, ys, xerr=[los, his], fmt="o", ms=6, capsize=0, elinewidth=5,
                    ecolor=MCOL[model] + "44", color=MCOL[model], label=MNICE[model],
                    linestyle="none")
    ax.axvline(0.5, ls="--", lw=0.8, color="#898781")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontweight="bold")
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.invert_yaxis()
    ax.text(0.5, -0.52, " chance", color="#898781", fontsize=8, va="top")
    ax.set_xlim(0.42, 1.03)
    ax.set_xlabel("within-model dyadic match probability (unbiased)")
    ax.set_title("Step 1 — within-model coordination by stimulus",
                 fontweight="bold", loc="left", pad=34)
    n_txt = ", ".join(f"{MNICE[m]} n={idx[(m, order[0])]['n_valid']}"
                      for m in models if (m, order[0]) in idx)
    ax.text(0, 1.015, f"one-shot, no communication · {n_txt} per stimulus\n"
                      f"bars = 95% bootstrap CI ({BOOT_B:,} resamples)",
            transform=ax.transAxes, fontsize=8, color="#52514e", va="bottom",
            linespacing=1.45)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="x", color="#e1e0d9", lw=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    png = os.path.join(OUTDIR, f"step1_match_{ts}.png")
    pdf = os.path.join(OUTDIR, f"step1_match_{ts}.pdf")
    fig.savefig(png); fig.savefig(pdf)
    plt.close(fig)

    # ---- console report ----
    print(f"source: {os.path.basename(path)}   rows={len(rows)}  valid trials={len(valid)}  "
          f"invalid first responses={len(invalid)}")
    print("\n=== identity coding (the quantity of interest) ===")
    hdr = f"{'model':8}{'pair':9}{'n':>4}{'modal':>13}{'share':>8}{'match':>8}{'95% CI':>16}{'H(bits)':>9}"
    print(hdr)
    for r in summary:
        ci = f"[{r['match_unbiased_lo']:.3f},{r['match_unbiased_hi']:.3f}]"
        print(f"{r['model']:8}{r['stimulus_pair']:9}{r['n_valid']:>4}{r['modal_option']:>13}"
              f"{r['modal_share']:>8.3f}{r['match_unbiased']:>8.3f}{ci:>16}{r['entropy_bits']:>9.3f}"
              + ("   TIED" if r["modal_tied"] else ""))

    print("\n=== position coding (label-following diagnostic) ===")
    print(f"{'model':8}{'pair':9}{'P(said A)':>11}{'match_pos':>11}   flag")
    for r in summary:
        flag = ""
        if r["match_unbiased_position"] >= 0.75 and r["match_unbiased"] < 0.6:
            flag = "<-- answering by LABEL, not stimulus"
        print(f"{r['model']:8}{r['stimulus_pair']:9}{r['p_position_A']:>11.3f}"
              f"{r['match_unbiased_position']:>11.3f}   {flag}")

    print("\nwrote:")
    for p in (sum_path, comp_path, png, pdf):
        print(f"  {p}")


if __name__ == "__main__":
    main()
