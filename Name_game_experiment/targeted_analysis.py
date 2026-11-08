"""
STEP 1c analysis - per-cell metrics, between-model disparity, and selection.

Runs on either stage. The stage only changes which files are written and whether
the selection logic runs; every statistic is computed by the same code in
`targeted_common.py`, so screen and confirmation numbers are directly comparable.

  --stage screen   -> targeted_screen_summary.csv
                      targeted_screen_comparison.csv
                      targeted_screen_selected.csv      <- picks the confirmation battery
  --stage confirm  -> targeted_confirmation_summary.csv
                      targeted_confirmation_comparison.csv

SELECTION (screen stage only; never applied to confirmation data)

  A candidate is a PRIMARY / clean form disparity when all five hold:
      strong model dyadic match  >= 0.85
      weak   model dyadic match  <= 0.70
      |match difference|         >= 0.20
      strong model form consistency >= 0.90
      weak   model form consistency >= 0.80

  The last one is the load-bearing criterion. A model with no form rule does not
  answer at random - it defaults to a side, which under balanced AB/BA reads as
  ~0.50 on the identity coding. That looks like indifference but is not: it is a
  position fallback wearing a post-hoc form rationale. Requiring the WEAK model to
  keep form consistency >= 0.80 means both models are genuinely tracking form and
  one is simply more concentrated - which is the estimand we want.

  Candidates meeting the first four but failing the weak-model form criterion are
  kept as SECONDARY and flagged clean_form_disparity=False. They are a real and
  reportable phenomenon - a difference in KIND, one model coordinating on form
  while the other cannot and falls back on position - but they are not evidence of
  differing focal strength, and are never counted as clean disparities.

  Nothing is dropped. Every candidate appears in the output with its reason.
"""
import os, sys, csv, glob, argparse
from collections import Counter, defaultdict

import targeted_common as tc

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "step1c_output")

STRONG_MATCH_MIN = 0.85
WEAK_MATCH_MAX = 0.70
MIN_DELTA = 0.20
STRONG_FC_MIN = 0.90
WEAK_FC_MIN = 0.80
# Below this the weak model is not weakly form-sensitive, it is choosing a side.
# Separated from SECONDARY because "FC 0.67" and "FC 0.00" are different claims:
# the first is a partial fallback, the second is a pure position default whose
# ~0.50 match rate carries no information about form at all.
WEAK_FC_FLOOR = 0.35

# Controls carried into confirmation, per the design. Baselines are the Step 1
# n=60 form-choice run (gpt/claude): proto 1.000/1.000, rot 0.696/0.741,
# mirror 0.573/0.497, noise 0.506/0.494.
CONTROLS = [("proto", "shared-easy positive control"),
            ("rot", "intermediate shared coordination"),
            ("mirror", "structured ambiguous control"),
            ("noise", "floor control")]

# Step 1b disparities re-run fresh alongside the new candidates, so the whole
# battery has confirmation-sample estimates rather than a mix of old and new.
INCUMBENTS = [("s_arrow", "Claude>GPT"), ("s_convexity", "Claude>GPT"),
              ("c_open_cplx", "Claude>GPT"), ("s_contrast", "GPT>Claude"),
              ("s_density", "GPT>Claude")]

MAX_BATTERY = 17


def resolve(stage, argpath):
    if argpath:
        p = argpath if os.path.isabs(argpath) else os.path.join(OUTDIR, argpath)
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")
        return p
    found = sorted(glob.glob(os.path.join(OUTDIR, f"targeted_{stage}_calls_*.csv")))
    if not found:
        sys.exit(f"no targeted_{stage}_calls_*.csv in {OUTDIR}")
    return found[-1]


def build_summary(valid, diag, reg):
    by = defaultdict(list)
    for r in valid:
        by[(r["model"], r["stimulus_pair"])].append(r)

    ids = sorted({r["stimulus_pair"] for r in valid})
    summary, draws = [], {}
    for model in ("gpt", "claude"):
        for sid in ids:
            cell = by.get((model, sid), [])
            if not cell:
                continue
            e = reg[sid]
            m = tc.cell_metrics(cell, e)
            d = tc.boot_draws([r["choice_identity"] for r in cell],
                              f"{tc.BOOT_SEED}|{model}|{sid}")
            draws[(model, sid)] = d
            lo, hi = tc.pct_ci(d)
            summary.append(dict(
                model=model, model_id=cell[0]["model_id"], stimulus_pair=sid,
                family=e["family"], manipulation=e["manipulation"], source=e["source"],
                identity_1=e["identity_1"], identity_2=e["identity_2"], **m,
                ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                n_parse_retry=diag["parse_fail"].get((model, sid), 0),
                n_api_error=diag["api_err"].get((model, sid), 0)))
    return summary, draws, ids


def build_comparison(summary, draws, ids, reg):
    idx = {(r["model"], r["stimulus_pair"]): r for r in summary}
    out = []
    for sid in ids:
        g, c = idx.get(("gpt", sid)), idx.get(("claude", sid))
        if not (g and c):
            continue
        delta = round(g["match_unbiased"] - c["match_unbiased"], 4)
        dd = [a - b for a, b in zip(draws[("gpt", sid)], draws[("claude", sid)])]
        dlo, dhi = tc.pct_ci(dd)
        strong, weak = (g, c) if delta > 0 else (c, g)
        out.append(dict(
            stimulus_pair=sid, family=g["family"], manipulation=g["manipulation"],
            source=g["source"],
            gpt_match=g["match_unbiased"], gpt_ci_lo=g["ci_lo"], gpt_ci_hi=g["ci_hi"],
            gpt_plugin=g["match_plugin"], gpt_modal=g["modal_identity"],
            gpt_p_left=g["p_left"], gpt_form_consistency=g["form_consistency"],
            gpt_p_modal_left=g["p_modal_given_modal_left"],
            gpt_p_modal_right=g["p_modal_given_modal_right"], gpt_n=g["n_valid"],
            claude_match=c["match_unbiased"], claude_ci_lo=c["ci_lo"], claude_ci_hi=c["ci_hi"],
            claude_plugin=c["match_plugin"], claude_modal=c["modal_identity"],
            claude_p_left=c["p_left"], claude_form_consistency=c["form_consistency"],
            claude_p_modal_left=c["p_modal_given_modal_left"],
            claude_p_modal_right=c["p_modal_given_modal_right"], claude_n=c["n_valid"],
            delta_gpt_minus_claude=delta,
            delta_ci_lo=round(dlo, 4), delta_ci_hi=round(dhi, 4),
            delta_ci_excludes_zero=bool(dlo > 0 or dhi < 0),
            direction=("GPT>Claude" if delta > 0 else "Claude>GPT"),
            strong_model=("gpt" if delta > 0 else "claude"),
            strong_match=strong["match_unbiased"], weak_match=weak["match_unbiased"],
            strong_form_consistency=strong["form_consistency"],
            weak_form_consistency=weak["form_consistency"]))
    out.sort(key=lambda r: -abs(r["delta_gpt_minus_claude"]))
    return out


def classify(r):
    """(status, clean, reason) for one comparison row. Nothing is silently dropped."""
    d = abs(r["delta_gpt_minus_claude"])
    sm, wm = r["strong_match"], r["weak_match"]
    sfc, wfc = r["strong_form_consistency"], r["weak_form_consistency"]

    fails = []
    if sm < STRONG_MATCH_MIN:
        fails.append(f"strong match {sm:.3f} < {STRONG_MATCH_MIN}")
    if wm > WEAK_MATCH_MAX:
        fails.append(f"weak match {wm:.3f} > {WEAK_MATCH_MAX}")
    if d < MIN_DELTA:
        fails.append(f"|delta| {d:.3f} < {MIN_DELTA}")
    if sfc < STRONG_FC_MIN:
        fails.append(f"strong form consistency {sfc:.2f} < {STRONG_FC_MIN}")

    if fails:
        if d < MIN_DELTA and sm >= 0.95 and wm >= 0.95:
            return "REJECT", False, "both models at ceiling; no disparity to detect"
        if d < MIN_DELTA and sm <= 0.60:
            return "REJECT", False, "both models at floor; no shared focal point"
        return "REJECT", False, "; ".join(fails)

    wpl = r["gpt_p_left"] if r["strong_model"] == "claude" else r["claude_p_left"]
    if wfc < WEAK_FC_FLOOR:
        return ("POSITION_DEFAULT", False,
                f"weak model form consistency {wfc:.2f} < {WEAK_FC_FLOOR} "
                f"(P(LEFT)={wpl:.2f}): the weak model is essentially choosing a SIDE, "
                "not expressing a weaker form preference. Its ~0.50 match rate is a "
                "position default, NOT indifference between the forms. Reported in "
                "full, but this is not a focality disparity and is not recommended "
                "for the battery")
    if wfc < WEAK_FC_MIN:
        return ("SECONDARY", False,
                f"weak model form consistency {wfc:.2f} < {WEAK_FC_MIN} "
                f"(P(LEFT)={wpl:.2f}): difference in KIND - weak model PARTIALLY "
                "falls back on position, not a clean focality difference")
    return ("PRIMARY", True,
            f"clean: both models track form (FC {sfc:.2f} / {wfc:.2f}), "
            f"strong {sm:.3f} vs weak {wm:.3f}, delta {d:.3f}")


def select(comparison, reg):
    rows = []
    for r in comparison:
        status, clean, reason = classify(r)
        rows.append(dict(
            stimulus_id=r["stimulus_pair"], family=r["family"],
            manipulation=r["manipulation"], source=r["source"],
            screen_status=status, clean_form_disparity=clean,
            disparity_direction=r["direction"],
            gpt_match=r["gpt_match"], claude_match=r["claude_match"],
            delta_gpt_minus_claude=r["delta_gpt_minus_claude"],
            delta_ci=f"[{r['delta_ci_lo']}, {r['delta_ci_hi']}]",
            gpt_form_consistency=r["gpt_form_consistency"],
            claude_form_consistency=r["claude_form_consistency"],
            role="", selection_status="NOT_SELECTED", reason=reason))

    idx = {r["stimulus_id"]: r for r in rows}
    chosen = []

    def take(sid, role, note):
        if sid in idx:
            idx[sid]["role"] = role
            idx[sid]["selection_status"] = "SELECT"
            idx[sid]["reason"] = note + " | " + idx[sid]["reason"]
        else:
            e = reg.get(sid, {})
            rows.append(dict(
                stimulus_id=sid, family=e.get("family", ""), manipulation=e.get("manipulation", ""),
                source=e.get("source", ""), screen_status="NOT_SCREENED",
                clean_form_disparity="", disparity_direction="",
                gpt_match="", claude_match="", delta_gpt_minus_claude="", delta_ci="",
                gpt_form_consistency="", claude_form_consistency="",
                role=role, selection_status="SELECT", reason=note))
        chosen.append(sid)

    # 1. incumbents from Step 1b, re-measured fresh
    for sid, direction in INCUMBENTS:
        take(sid, f"incumbent ({direction})",
             "carried from Step 1b for a fresh confirmation-sample estimate")

    # 2. new candidates, best first within each direction.
    # Rank PRIMARY > SECONDARY > POSITION_DEFAULT, then by |delta|. Position
    # defaults are carried into confirmation rather than dropped - whether a pure
    # side default reproduces on a fresh sample is itself worth knowing, and
    # dropping them here would hide exactly the artefact this screen exists to
    # expose. They are labelled as such everywhere they appear.
    TIER = {"PRIMARY": 0, "SECONDARY": 1, "POSITION_DEFAULT": 2}
    for direction in ("Claude>GPT", "GPT>Claude"):
        pool = [r for r in rows
                if r["source"] == "targeted" and r["disparity_direction"] == direction
                and r["screen_status"] in TIER and r["stimulus_id"] not in chosen]
        pool.sort(key=lambda r: (TIER[r["screen_status"]],
                                 -abs(float(r["delta_gpt_minus_claude"]))))
        for r in pool[:4]:
            if len(chosen) + len(CONTROLS) >= MAX_BATTERY:
                break
            take(r["stimulus_id"], f"new candidate ({direction}, {r['screen_status']})",
                 f"screen {r['screen_status']}, ranked by tier then |delta|")

    # 3. controls, always
    for sid, role in CONTROLS:
        take(sid, f"control - {role}", "retained as a control regardless of screen result")

    order = {"SELECT": 0, "NOT_SELECTED": 1}
    rows.sort(key=lambda r: (order[r["selection_status"]],
                             -abs(float(r["delta_gpt_minus_claude"] or 0))))
    return rows


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def qc(valid, diag, summary, n_expected):
    print("\n--- QC ---")
    print(f"rows in file {diag['n_rows']}, unique run_ids {diag['n_runs']}, "
          f"valid trials {len(valid)}")
    dup = len(valid) - len({r["run_id"] for r in valid})
    print(f"duplicated run_ids among valid trials : {dup}  {'OK' if dup == 0 else 'FAIL'}")
    print(f"trials lost (both attempts unparseable): {len(diag['losses'])}"
          f"  {'OK' if not diag['losses'] else 'see below'}")
    if diag["losses"]:
        for r in diag["losses"]:
            print(f"    LOST {r}")
    bad_n = [(s["model"], s["stimulus_pair"], s["n_valid"])
             for s in summary if s["n_valid"] != n_expected]
    print(f"cells with n != {n_expected}: {len(bad_n)}  {bad_n if bad_n else 'OK'}")
    bad_b = [(s["model"], s["stimulus_pair"], s["n_AB"], s["n_BA"])
             for s in summary if s["n_AB"] != s["n_BA"]]
    print(f"cells with AB != BA         : {len(bad_b)}  {bad_b if bad_b else 'OK'}")
    tot_err = sum(diag["api_err"].values())
    tot_retry = sum(diag["parse_fail"].values())
    print(f"API errors {tot_err} | unparseable first responses (retried) {tot_retry}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("screen", "confirm"), required=True)
    ap.add_argument("calls", nargs="?", default=None)
    ap.add_argument("--n-expected", type=int, default=None)
    args = ap.parse_args()

    path = resolve(args.stage, args.calls)
    reg = tc.registry()
    valid, diag = tc.load_valid(path)
    summary, draws, ids = build_summary(valid, diag, reg)
    comparison = build_comparison(summary, draws, ids, reg)

    n_exp = args.n_expected or (30 if args.stage == "screen" else 60)
    print(f"source: {os.path.basename(path)}   stimuli={len(ids)}   valid={len(valid)}")
    qc(valid, diag, summary, n_exp)

    tag = "screen" if args.stage == "screen" else "confirmation"
    written = [write(os.path.join(OUTDIR, f"targeted_{tag}_summary.csv"), summary),
               write(os.path.join(OUTDIR, f"targeted_{tag}_comparison.csv"), comparison)]

    print(f"\n{'stimulus':17}{'GPT':>7}{'gFC':>6}{'Claude':>8}{'cFC':>6}{'delta':>8}"
          f"  {'95% CI':>18}  status")
    for r in comparison:
        status = classify(r)[0] if args.stage == "screen" else ""
        print(f"{r['stimulus_pair']:17}{r['gpt_match']:>7.3f}{r['gpt_form_consistency']:>6.2f}"
              f"{r['claude_match']:>8.3f}{r['claude_form_consistency']:>6.2f}"
              f"{r['delta_gpt_minus_claude']:>+8.3f}"
              f"  [{r['delta_ci_lo']:+.3f}, {r['delta_ci_hi']:+.3f}]  {status}")

    if args.stage == "screen":
        sel = select(comparison, reg)
        written.append(write(os.path.join(OUTDIR, "targeted_screen_selected.csv"), sel))
        prim = [r for r in sel if r["screen_status"] == "PRIMARY"]
        sec = [r for r in sel if r["screen_status"] == "SECONDARY"]
        battery = [r for r in sel if r["selection_status"] == "SELECT"]
        print(f"\n=== screen result ===")
        print(f"  PRIMARY   (clean form disparity) : {len(prim)}  "
              f"{[r['stimulus_id'] for r in prim]}")
        print(f"  SECONDARY (position fallback)    : {len(sec)}  "
              f"{[r['stimulus_id'] for r in sec]}")
        print(f"  confirmation battery: {len(battery)}")
        for r in battery:
            print(f"    {r['stimulus_id']:17}{r['role']}")

    print("\nwrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
