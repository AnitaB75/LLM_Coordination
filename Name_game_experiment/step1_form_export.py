"""
Build the Step 1 deliverable bundle from an existing raw call-level CSV.

Pure post-processing — reads `step1_output/step1_form_calls_<ts>.csv` and writes a
self-contained `step1_deliverables_<ts>/` folder. Makes no API calls and does not
modify the raw file.

Outputs
  01_raw_calls.csv                  one row per API attempt, + included_in_analysis
  02_summary_model_x_stimulus.csv   per model x stimulus, all statistics
  03_comparison_gpt_vs_claude.csv   one row per stimulus, incl. CI on the difference
  04_run_config.md                  models, settings, full prompt, deviations
  05_match_plot.png / .pdf          copied from the analysis run
  README.md                         what each file is and how to verify it

Run:  python3 step1_form_export.py [step1_form_calls_<ts>.csv]
"""
import os, sys, csv, glob, math, json, random, shutil
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "step1_output")
BOOT_B = 5000
BOOT_SEED = 20260804
PAIR_ORDER = ["proto", "inkblot", "rot", "noise", "mirror"]


def unbiased_match(counts):
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


def boot_draws(labels, seed, b=BOOT_B):
    """Bootstrap draws of the unbiased match probability."""
    rng = random.Random(seed)
    n = len(labels)
    return [unbiased_match(Counter(rng.choices(labels, k=n))) for _ in range(b)]


def pct_ci(draws):
    d = sorted(draws)
    return d[int(0.025 * len(d))], d[int(0.975 * len(d))]


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
            sys.exit("no step1_form_calls_*.csv found")
        path = found[-1]
    ts = os.path.basename(path).replace("step1_form_calls_", "").replace(".csv", "")
    bundle = os.path.join(OUTDIR, f"step1_deliverables_{ts}")
    os.makedirs(bundle, exist_ok=True)

    rows = list(csv.DictReader(open(path)))

    # ---------- 1. raw calls, with included_in_analysis ----------
    # One trial (run_id) may have two attempts if the first was unparseable.
    # The row used downstream is the last valid attempt for that run_id.
    used = {}
    for i, r in enumerate(rows):
        if r["valid_response"] == "True":
            used[r["run_id"]] = i          # last valid attempt wins
    raw_cols = ["model", "model_id", "stimulus_pair", "run_id", "attempt_number",
                "prompt_sha8", "variant", "left_identity", "right_identity",
                "response_raw", "reasoning", "response_side", "choice_identity",
                "valid_response", "superseded_by_retry", "included_in_analysis",
                "api_settings", "timestamp"]
    raw_path = os.path.join(bundle, "01_raw_calls.csv")
    with open(raw_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=raw_cols)
        w.writeheader()
        for i, r in enumerate(rows):
            out = {c: r.get(c, "") for c in raw_cols}
            out["attempt_number"] = r["attempt"]
            out["included_in_analysis"] = (used.get(r["run_id"]) == i)
            w.writerow(out)

    analysed = [rows[i] for i in sorted(used.values())]
    by_cell = defaultdict(list)
    for r in analysed:
        by_cell[(r["model"], r["stimulus_pair"])].append(r)
    # attempt-level failure bookkeeping, per cell
    attempts_by_cell = defaultdict(list)
    for r in rows:
        attempts_by_cell[(r["model"], r["stimulus_pair"])].append(r)

    models = ["gpt", "claude"]
    pairs = [p for p in PAIR_ORDER if any((m, p) in by_cell for m in models)]

    # ---------- 2. summary ----------
    summary, draws_cache = [], {}
    for model in models:
        for pair in pairs:
            cell = by_cell.get((model, pair), [])
            if not cell:
                continue
            atts = attempts_by_cell[(model, pair)]
            ident = [r["choice_identity"] for r in cell]
            side = [r["response_side"] for r in cell]
            ic, sc = Counter(ident), Counter(side)
            opts = sorted({r["left_identity"] for r in cell} | {r["right_identity"] for r in cell})
            mo, ms, tie = modal(ic)
            d = boot_draws(ident, f"{BOOT_SEED}|{model}|{pair}|id")
            draws_cache[(model, pair)] = d
            lo, hi = pct_ci(d)
            plo, phi = pct_ci(boot_draws(side, f"{BOOT_SEED}|{model}|{pair}|pos"))
            summary.append(dict(
                model=model, model_id=cell[0]["model_id"], stimulus_pair=pair,
                n_planned_trials=len(cell), n_api_attempts=len(atts), n_valid=len(cell),
                n_invalid_attempts=sum(1 for r in atts if r["valid_response"] != "True"),
                n_retried_trials=sum(1 for r in atts if r["superseded_by_retry"] == "True"),
                n_api_errors=sum(1 for r in atts if r["response_raw"].startswith("__ERR__")),
                identity_1=opts[0], n_identity_1=ic.get(opts[0], 0),
                p_identity_1=round(ic.get(opts[0], 0) / len(cell), 4),
                identity_2=opts[1], n_identity_2=ic.get(opts[1], 0),
                p_identity_2=round(ic.get(opts[1], 0) / len(cell), 4),
                modal_identity=mo, modal_share=round(ms, 4), modal_tied=tie,
                match_unbiased=round(unbiased_match(ic), 6),
                match_unbiased_ci_lo=round(lo, 6), match_unbiased_ci_hi=round(hi, 6),
                match_plugin=round(plugin_match(ic), 6),
                entropy_bits=round(entropy_bits(ic), 6),
                entropy_normalized=round(entropy_bits(ic) / math.log2(2), 6),
                p_left=round(sc.get("LEFT", 0) / len(cell), 4),
                match_position_unbiased=round(unbiased_match(sc), 6),
                match_position_ci_lo=round(plo, 6), match_position_ci_hi=round(phi, 6),
                n_AB_variant=sum(1 for r in cell if r["variant"] == "AB"),
                n_BA_variant=sum(1 for r in cell if r["variant"] == "BA"),
                bootstrap_B=BOOT_B, bootstrap_seed=BOOT_SEED))
    sum_path = os.path.join(bundle, "02_summary_model_x_stimulus.csv")
    with open(sum_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)

    # ---------- 3. comparison, incl. CI on the between-model difference ----------
    idx = {(r["model"], r["stimulus_pair"]): r for r in summary}
    comp = []
    for pair in pairs:
        g, c = idx.get(("gpt", pair)), idx.get(("claude", pair))
        if not (g and c):
            continue
        # paired by draw index over INDEPENDENT per-model resamples: the two models
        # were run as separate API calls, so there is no trial-level pairing to
        # preserve; a shared index just combines two independent replicates.
        gd, cd = draws_cache[("gpt", pair)], draws_cache[("claude", pair)]
        diff = [a - b for a, b in zip(gd, cd)]
        dlo, dhi = pct_ci(diff)
        comp.append(dict(
            stimulus_pair=pair,
            gpt_modal_identity=g["modal_identity"], gpt_modal_share=g["modal_share"],
            gpt_match=g["match_unbiased"], gpt_ci_lo=g["match_unbiased_ci_lo"],
            gpt_ci_hi=g["match_unbiased_ci_hi"], gpt_p_left=g["p_left"],
            claude_modal_identity=c["modal_identity"], claude_modal_share=c["modal_share"],
            claude_match=c["match_unbiased"], claude_ci_lo=c["match_unbiased_ci_lo"],
            claude_ci_hi=c["match_unbiased_ci_hi"], claude_p_left=c["p_left"],
            diff_gpt_minus_claude=round(g["match_unbiased"] - c["match_unbiased"], 6),
            diff_ci_lo=round(dlo, 6), diff_ci_hi=round(dhi, 6),
            diff_ci_excludes_zero=bool(dlo > 0 or dhi < 0),
            same_modal_identity=g["modal_identity"] == c["modal_identity"]))
    comp_path = os.path.join(bundle, "03_comparison_gpt_vs_claude.csv")
    with open(comp_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comp[0]))
        w.writeheader(); w.writerows(comp)

    # ---------- 5. plot ----------
    for ext in ("png", "pdf"):
        src = os.path.join(OUTDIR, f"step1_form_match_{ts}.{ext}")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(bundle, f"05_match_plot.{ext}"))

    print(f"bundle -> {bundle}")
    for f in sorted(os.listdir(bundle)):
        print(f"  {f}")
    return bundle, ts, rows, analysed, summary, comp


if __name__ == "__main__":
    main()
