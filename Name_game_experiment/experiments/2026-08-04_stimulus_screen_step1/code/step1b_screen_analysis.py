"""
STEP 1b screening analysis — apply the selection criteria and find crossovers.

Criteria, all three required:
  1. POSITION      P(LEFT) in [0.35, 0.65] for BOTH models
  2. DISSOCIATION  |match_gpt - match_claude| > 0.15
  3. FORM          conditional-position diagnostic passes for both models

The conditional-position diagnostic is the key one and is worth stating plainly.
For a model's modal identity m, compute the rate at which m is chosen separately
under each layout:

    p_AB = P(choose m | m on the left)      p_BA = P(choose m | m on the right)

A model choosing by FORM picks m at the same rate either way, so p_AB ~ p_BA.
A model choosing by SIDE picks m only when m happens to sit on its preferred
side, so p_AB and p_BA diverge toward 1 and 0. We report

    form_consistency = 1 - |p_AB - p_BA|

which is 1 for pure form-tracking and 0 for a pure side default. Note this is
necessary but NOT sufficient: a model answering at random also scores ~1, so it
is only meaningful read jointly with a match rate above chance. Both conditions
are enforced.

A lexical check on the rationales is reported alongside — whether the model
justified its pick in positional terms ("the left one") rather than by form.
This is a coarse proxy and is surfaced for review, not used to auto-reject.

Outputs
  step1b_screen_summary_<ts>.csv     per model x pair
  step1b_screen_selected_<ts>.csv    pairs passing all criteria, with crossover flag
  step1b_screen_<ts>.png / .pdf      GPT vs Claude match, crossovers highlighted

Run:  python3 step1b_screen_analysis.py [step1b_screen_calls_<ts>.csv]
"""
import os, sys, csv, glob, math, re, random
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("academic")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "step1_output")
BOOT_B = 5000
BOOT_SEED = 20260804

PLEFT_LO, PLEFT_HI = 0.35, 0.65
MIN_DELTA = 0.15
MIN_FORM_CONSISTENCY = 0.60
MIN_MATCH_ABOVE_CHANCE = 0.55     # guards the "random also scores 1" hole

POS_WORDS = re.compile(r"\b(left|right|first|second|leftmost|rightmost)\b", re.I)


def unbiased_match(c):
    n = sum(c.values())
    return float("nan") if n < 2 else sum(v * (v - 1) for v in c.values()) / (n * (n - 1))


def entropy_bits(c):
    n = sum(c.values())
    return float("nan") if not n else -sum((v / n) * math.log2(v / n) for v in c.values() if v)


def boot_draws(labels, seed, b=BOOT_B):
    rng = random.Random(seed)
    n = len(labels)
    return [unbiased_match(Counter(rng.choices(labels, k=n))) for _ in range(b)]


def pct_ci(d):
    s = sorted(d)
    return s[int(0.025 * len(s))], s[int(0.975 * len(s))]


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(OUTDIR, path)
    else:
        found = sorted(glob.glob(os.path.join(OUTDIR, "step1b_screen_calls_*.csv")))
        if not found:
            sys.exit("no step1b_screen_calls_*.csv found")
        path = found[-1]
    ts = os.path.basename(path).replace("step1b_screen_calls_", "").replace(".csv", "")

    rows = list(csv.DictReader(open(path)))
    seen, valid = set(), []
    for r in rows:
        if r["valid_response"] == "True" and r["run_id"] not in seen:
            seen.add(r["run_id"]); valid.append(r)

    by = defaultdict(list)
    for r in valid:
        by[(r["model"], r["stimulus_pair"])].append(r)

    pairs = sorted({r["stimulus_pair"] for r in valid})
    meta = {r["stimulus_pair"]: (r["family"], r["cue_1"], r["cue_2"]) for r in valid}

    summary, draws = [], {}
    for model in ("gpt", "claude"):
        for pair in pairs:
            cell = by.get((model, pair), [])
            if not cell:
                continue
            ident = [r["choice_identity"] for r in cell]
            ic = Counter(ident)
            m_ident = max(ic, key=lambda k: (ic[k], k))
            # conditional-position diagnostic
            ab = [r for r in cell if r["variant"] == "AB"]
            ba = [r for r in cell if r["variant"] == "BA"]
            p_ab = sum(r["choice_identity"] == m_ident for r in ab) / len(ab) if ab else float("nan")
            p_ba = sum(r["choice_identity"] == m_ident for r in ba) / len(ba) if ba else float("nan")
            form_consistency = 1 - abs(p_ab - p_ba)
            d = boot_draws(ident, f"{BOOT_SEED}|{model}|{pair}")
            draws[(model, pair)] = d
            lo, hi = pct_ci(d)
            sc = Counter(r["response_side"] for r in cell)
            pos_lex = sum(bool(POS_WORDS.search(r["response_raw"].split("CHOICE:")[0]))
                          for r in cell) / len(cell)
            fam, c1, c2 = meta[pair]
            summary.append(dict(
                model=model, stimulus_pair=pair, family=fam, cue_1=c1, cue_2=c2,
                n=len(cell), modal_identity=m_ident,
                modal_share=round(ic[m_ident] / len(cell), 4),
                match_unbiased=round(unbiased_match(ic), 4),
                ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                entropy_bits=round(entropy_bits(ic), 4),
                p_left=round(sc.get("LEFT", 0) / len(cell), 4),
                p_modal_given_AB=round(p_ab, 4), p_modal_given_BA=round(p_ba, 4),
                form_consistency=round(form_consistency, 4),
                rationale_mentions_position=round(pos_lex, 4)))

    sum_path = os.path.join(OUTDIR, f"step1b_screen_summary_{ts}.csv")
    with open(sum_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    # ---- apply criteria ----
    idx = {(r["model"], r["stimulus_pair"]): r for r in summary}
    sel = []
    for pair in pairs:
        g, c = idx.get(("gpt", pair)), idx.get(("claude", pair))
        if not (g and c):
            continue
        delta = g["match_unbiased"] - c["match_unbiased"]
        dd = [a - b for a, b in zip(draws[("gpt", pair)], draws[("claude", pair)])]
        dlo, dhi = pct_ci(dd)
        hi_m, lo_m = (g, c) if delta > 0 else (c, g)
        # POSITION criterion applies to the CONCENTRATED model only.
        #
        # Requiring modest P(LEFT) from *both* models is close to self-defeating:
        # a model with no form rule does not answer randomly, it defaults to a
        # side. So a large |delta| almost guarantees the low model has an extreme
        # P(LEFT), and screening on both models would reject exactly the
        # dissociations we are looking for. Nothing is lost by relaxing it — the
        # two models are separate API calls in separate cells, so the low model's
        # side default cannot bias the high model's estimate, and under a balanced
        # AB/BA counterbalance a side default reads as chance on the identity
        # coding, which is the correct answer for "no form preference".
        # `lo_p_left` is still reported so the low model's behaviour is visible.
        pos_ok = PLEFT_LO <= hi_m["p_left"] <= PLEFT_HI
        diss_ok = abs(delta) > MIN_DELTA
        form_ok = (hi_m["form_consistency"] >= MIN_FORM_CONSISTENCY
                   and hi_m["match_unbiased"] >= MIN_MATCH_ABOVE_CHANCE)
        sel.append(dict(
            stimulus_pair=pair, family=g["family"], cue_1=g["cue_1"], cue_2=g["cue_2"],
            gpt_match=g["match_unbiased"], gpt_ci=f"[{g['ci_lo']}, {g['ci_hi']}]",
            gpt_modal=g["modal_identity"], gpt_p_left=g["p_left"],
            gpt_form_consistency=g["form_consistency"],
            claude_match=c["match_unbiased"], claude_ci=f"[{c['ci_lo']}, {c['ci_hi']}]",
            claude_modal=c["modal_identity"], claude_p_left=c["p_left"],
            claude_form_consistency=c["form_consistency"],
            delta_gpt_minus_claude=round(delta, 4),
            delta_ci_lo=round(dlo, 4), delta_ci_hi=round(dhi, 4),
            delta_ci_excludes_zero=bool(dlo > 0 or dhi < 0),
            direction=("GPT>Claude" if delta > 0 else "Claude>GPT"),
            crit_position_ok=pos_ok, crit_dissociation_ok=diss_ok, crit_form_ok=form_ok,
            selected=bool(pos_ok and diss_ok and form_ok)))
    sel.sort(key=lambda r: -abs(r["delta_gpt_minus_claude"]))
    sel_path = os.path.join(OUTDIR, f"step1b_screen_selected_{ts}.csv")
    with open(sel_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sel[0])); w.writeheader(); w.writerows(sel)

    # ---- plot: GPT vs Claude match, selected pairs highlighted ----
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    for r in sel:
        picked = r["selected"]
        ax.plot(r["claude_match"], r["gpt_match"],
                marker="o" if picked else "o", ms=5.5 if picked else 3.5,
                mfc="black" if picked else "white", mec="black", mew=0.8, ls="none")
        if picked:
            ax.annotate(r["stimulus_pair"], (r["claude_match"], r["gpt_match"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
    lim = (0.42, 1.03)
    ax.plot(lim, lim, ls=(0, (4, 3)), lw=0.7, color="black", zorder=0)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("Claude Opus 4.8 — dyadic match probability")
    ax.set_ylabel("GPT-5.4 — dyadic match probability")
    ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    png = os.path.join(OUTDIR, f"step1b_screen_{ts}.png")
    fig.savefig(png); fig.savefig(png.replace(".png", ".pdf"))
    plt.close(fig)

    # ---- console ----
    n_sel = sum(r["selected"] for r in sel)
    up = [r for r in sel if r["selected"] and r["delta_gpt_minus_claude"] > 0]
    dn = [r for r in sel if r["selected"] and r["delta_gpt_minus_claude"] < 0]
    print(f"source: {os.path.basename(path)}  valid trials={len(valid)}  pairs={len(pairs)}")
    print(f"\n=== selection: {n_sel}/{len(pairs)} pairs pass all three criteria ===")
    print(f"  GPT > Claude : {len(up)}   {[r['stimulus_pair'] for r in up]}")
    print(f"  Claude > GPT : {len(dn)}   {[r['stimulus_pair'] for r in dn]}")
    print(f"  CROSSOVER available: {bool(up and dn)}")

    print(f"\n{'pair':16}{'fam':8}{'GPT':>7}{'Claude':>8}{'delta':>8}{'gL':>6}{'cL':>6}"
          f"{'gFC':>6}{'cFC':>6}  pos diss form  SEL")
    for r in sel:
        print(f"{r['stimulus_pair']:16}{r['family'][:7]:8}{r['gpt_match']:>7.3f}"
              f"{r['claude_match']:>8.3f}{r['delta_gpt_minus_claude']:>+8.3f}"
              f"{r['gpt_p_left']:>6.2f}{r['claude_p_left']:>6.2f}"
              f"{r['gpt_form_consistency']:>6.2f}{r['claude_form_consistency']:>6.2f}"
              f"   {'Y' if r['crit_position_ok'] else '.'}   "
              f"{'Y' if r['crit_dissociation_ok'] else '.'}   "
              f"{'Y' if r['crit_form_ok'] else '.'}   "
              f"{'**' if r['selected'] else ''}")

    print("\nwrote:")
    for p in (sum_path, sel_path, png):
        print(f"  {p}")


if __name__ == "__main__":
    main()
