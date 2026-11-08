"""
STEP 1c report - assemble the screen and confirmation results into Markdown.

Reads everything from the CSVs in step1c_output/ so the report cannot drift from
the data. Writes targeted_stimulus_screen_report.md.

Run:  python3 targeted_report.py
"""
import os, csv, datetime
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "step1c_output")

# Step 1b screen values for the incumbents, so their replication is judged
# against where they actually came from rather than against nothing.
# Source: step1_output/step1b_screen_selected_20260804_153626.csv (n=30/cell).
STEP1B = {"s_arrow": -0.4966, "s_convexity": -0.2643, "c_open_cplx": -0.2759,
          "s_contrast": +0.3701, "s_density": +0.1862}

# Step 1 form-choice values for the controls (n=60/cell), for the same reason.
# Source: step1_output/step1_form_summary_20260804_124918.csv
STEP1_CONTROLS = {"proto": (1.000, 1.000), "rot": (0.6955, 0.7407),
                  "mirror": (0.5729, 0.4966), "noise": (0.5056, 0.4938)}

FAMILY_LABEL = {
    "A_semantic": "A · semantic / canonical orientation",
    "B_glyph": "B · canonical abstract glyphs",
    "C_contrast": "C · contrast / visual weight",
    "D_density": "D · density / numerosity / texture",
    "E_closure": "E · convexity / closure",
}


def read(name):
    p = os.path.join(OUT, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def fl(r, k, d=float("nan")):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return d


def fm(v, n=3):
    return "—" if v != v else f"{v:.{n}f}"


def main():
    comp = read("targeted_screen_comparison.csv")
    sel = read("targeted_screen_selected.csv")
    conf = read("targeted_confirmation_comparison.csv")
    csum = read("targeted_confirmation_summary.csv")
    ssum = read("targeted_screen_summary.csv")

    status = {r["stimulus_id"]: r["screen_status"] for r in sel}
    reason = {r["stimulus_id"]: r["reason"] for r in sel}
    role = {r["stimulus_id"]: r["role"] for r in sel}
    picked = [r for r in sel if r["selection_status"] == "SELECT"]

    L = []
    A = L.append
    A("# Targeted stimulus screen (Step 1c)\n")
    A(f"_Generated {datetime.date.today().isoformat()} from `step1c_output/`._\n")
    A("Goal: extend the coordination battery with stimuli showing clear, **genuine** "
      "GPT-vs-Claude focality disparities — cases where both models coordinate on "
      "FORM and one is simply more concentrated — as opposed to cases where the "
      "weaker model has no form rule and quietly defaults to a side.\n")

    # ---------------------------------------------------------------- method
    A("## Procedure\n")
    A("One-shot coordination framing imported verbatim from `step1_form_choice.py` "
      "(prompt sha8 `3003ae1b`), the same module Step 1 and Step 1b used, so all "
      "three datasets are comparable. Model settings unchanged: GPT-5.4 with "
      "`reasoning.effort=low`, Claude Opus 4.8 with `max_tokens=2000`, adaptive "
      "thinking, `output_config.effort=low`. Neither API exposes a sampling seed; "
      "the fixed seeds here govern AB/BA allocation only.\n")
    A("- **Screen** — 27 new pairs × 2 models × n=30, exact 15/15 AB/BA.\n"
      "- **Confirmation** — selected battery × 2 models × n=60 **fresh** trials, "
      "exact 30/30 AB/BA, separate allocation seed.\n")
    A("Screen and confirmation are never pooled for the confirmatory estimates: the "
      "screen chose the stimuli, so reusing it would bias them upward (winner's "
      "curse). Every confirmation number below is from the fresh sample alone.\n")

    A("### The position-fallback diagnostic\n")
    A("For a model's modal identity *m*, form consistency is\n")
    A("```\nform_consistency = 1 − | P(choose m | m shown LEFT) − P(choose m | m shown RIGHT) |\n```\n")
    A("A model choosing by form picks *m* at the same rate wherever it sits, scoring "
      "~1. A model choosing by side picks *m* only when *m* lands on its preferred "
      "side, driving the two conditionals to 1 and 0 and the score to 0. Under "
      "balanced AB/BA a pure side default reads as ≈0.50 on the identity coding — "
      "**indistinguishable from indifference on the match rate alone**. That is why "
      "a low match rate is never reported here as indifference without this "
      "diagnostic agreeing.\n")

    A("### Selection criteria\n")
    A("| criterion | threshold |\n|---|---|\n"
      "| strong model dyadic match | ≥ 0.85 |\n"
      "| weak model dyadic match | ≤ 0.70 |\n"
      "| absolute difference | ≥ 0.20 |\n"
      "| strong model form consistency | ≥ 0.90 |\n"
      "| **weak model form consistency** | **≥ 0.80** |\n")
    A("Meeting all five ⇒ **PRIMARY**, `clean_form_disparity = TRUE`. Meeting the "
      "first four but not the last ⇒ **SECONDARY**, flagged as a difference in "
      "*kind* (partial position fallback), never counted as a clean focality "
      "difference.\n")

    # ------------------------------------------------------------- screen
    A("## Screen results — all 27 candidates\n")
    A("`gFC` / `cFC` = form consistency. Sorted by |Δ|.\n")
    A("| stimulus | family | GPT | gFC | Claude | cFC | Δ (GPT−Claude) | 95% CI | status |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in comp:
        sid = r["stimulus_pair"]
        A(f"| `{sid}` | {r['family']} | {fm(fl(r,'gpt_match'))} | {fm(fl(r,'gpt_form_consistency'),2)} "
          f"| {fm(fl(r,'claude_match'))} | {fm(fl(r,'claude_form_consistency'),2)} "
          f"| {fl(r,'delta_gpt_minus_claude'):+.3f} "
          f"| [{fl(r,'delta_ci_lo'):+.3f}, {fl(r,'delta_ci_hi'):+.3f}] "
          f"| {status.get(sid,'—')} |")
    A("")

    prim = [r for r in comp if status.get(r["stimulus_pair"]) == "PRIMARY"]
    sec = [r for r in comp if status.get(r["stimulus_pair"]) == "SECONDARY"]

    A("### Clean disparities (PRIMARY)\n")
    if prim:
        for r in prim:
            sid = r["stimulus_pair"]
            A(f"- **`{sid}`** ({r['direction']}) — {reason.get(sid,'')}")
    else:
        A("_None. No new candidate met all five criteria in the screen._")
    A("")

    A("### Position-fallback disparities (SECONDARY)\n")
    if sec:
        A("Real and reportable, but a difference in kind rather than a focality "
          "difference — the weaker model is partially choosing a side.\n")
        for r in sec:
            sid = r["stimulus_pair"]
            A(f"- **`{sid}`** ({r['direction']}) — weak-model form consistency "
              f"{fm(fl(r,'weak_form_consistency'),2)}; {reason.get(sid,'')}")
    else:
        A("_None._")
    A("")

    # ------------------------------------------------------- family verdicts
    A("### Which families were productive\n")
    byfam = defaultdict(list)
    for r in comp:
        byfam[r["family"]].append(r)
    A("| family | n | median \\|Δ\\| | at ceiling (both ≥0.95) | at floor (both ≤0.60) | PRIMARY | SECONDARY |")
    A("|---|---|---|---|---|---|---|")
    for fam in sorted(byfam):
        rs = byfam[fam]
        ds = sorted(abs(fl(r, "delta_gpt_minus_claude")) for r in rs)
        med = ds[len(ds) // 2] if ds else float("nan")
        ceil_n = sum(fl(r, "gpt_match") >= 0.95 and fl(r, "claude_match") >= 0.95 for r in rs)
        floor_n = sum(fl(r, "gpt_match") <= 0.60 and fl(r, "claude_match") <= 0.60 for r in rs)
        p = sum(status.get(r["stimulus_pair"]) == "PRIMARY" for r in rs)
        s = sum(status.get(r["stimulus_pair"]) == "SECONDARY" for r in rs)
        A(f"| {FAMILY_LABEL.get(fam, fam)} | {len(rs)} | {fm(med)} | {ceil_n} | {floor_n} | {p} | {s} |")
    A("")

    # -------------------------------------------------------- confirmation
    A("## Confirmation — fresh n=60 per model per stimulus\n")
    if not conf:
        A("_Not yet run._\n")
    else:
        A("All figures below are from the confirmation sample only.\n")
        A("| stimulus | role | GPT (95% CI) | gFC | Claude (95% CI) | cFC | Δ (95% CI) | replicated |")
        A("|---|---|---|---|---|---|---|---|")
        scr = {r["stimulus_pair"]: r for r in comp}
        for r in sorted(conf, key=lambda r: -abs(fl(r, "delta_gpt_minus_claude"))):
            sid = r["stimulus_pair"]
            d = fl(r, "delta_gpt_minus_claude")
            d0 = (fl(scr[sid], "delta_gpt_minus_claude") if sid in scr
                  else STEP1B.get(sid, float("nan")))
            if d0 != d0:
                rep = "n/a (control)"
            else:
                same_dir = (d > 0) == (d0 > 0)
                rep = ("**yes** " if same_dir and abs(d) >= 0.20 else
                       "direction only " if same_dir else "**no** ")
                rep += f"({d0:+.3f} → {d:+.3f})"
            A(f"| `{sid}` | {role.get(sid,'')} "
              f"| {fm(fl(r,'gpt_match'))} [{fm(fl(r,'gpt_ci_lo'))}, {fm(fl(r,'gpt_ci_hi'))}] "
              f"| {fm(fl(r,'gpt_form_consistency'),2)} "
              f"| {fm(fl(r,'claude_match'))} [{fm(fl(r,'claude_ci_lo'))}, {fm(fl(r,'claude_ci_hi'))}] "
              f"| {fm(fl(r,'claude_form_consistency'),2)} "
              f"| {d:+.3f} [{fl(r,'delta_ci_lo'):+.3f}, {fl(r,'delta_ci_hi'):+.3f}] | {rep} |")
        A("")

        # ---- recommended battery, judged on the FRESH sample only ----
        import targeted_analysis as ta
        rec = []
        for r in conf:
            rr = dict(r)
            for k in ("delta_gpt_minus_claude", "strong_match", "weak_match",
                      "strong_form_consistency", "weak_form_consistency",
                      "gpt_p_left", "claude_p_left"):
                rr[k] = fl(r, k)
            rec.append((r["stimulus_pair"], ta.classify(rr)[0], rr))
        st_conf = {sid: s for sid, s, _ in rec}

        A("### Confirmation status under the same five criteria\n")
        A("Applying the criteria to the fresh sample is **reporting**, not "
          "selecting — the stimuli were already fixed by the screen. Nothing below "
          "was chosen using confirmation data.\n")
        A("| stimulus | screen | confirmation | verdict |\n|---|---|---|---|")
        for sid, s, rr in sorted(rec, key=lambda t: -abs(t[2]["delta_gpt_minus_claude"])):
            scr_s = status.get(sid, "— (control/incumbent)")
            note = ("holds" if s == scr_s else
                    "**upgraded**" if s == "PRIMARY" else
                    "**downgraded**" if scr_s == "PRIMARY" else "changed tier")
            if sid in STEP1_CONTROLS:
                note = "control"
            A(f"| `{sid}` | {scr_s} | {s} | {note} |")
        A("")

        clean = [(sid, rr) for sid, s, rr in rec if s == "PRIMARY"]
        A(f"**Clean disparities surviving confirmation: {len(clean)}** — "
          + (", ".join(f"`{s}`" for s, _ in clean) if clean else "none") + ".\n")

        A("### Controls vs their Step 1 baselines\n")
        A("| control | GPT now | GPT Step 1 | Claude now | Claude Step 1 |\n|---|---|---|---|---|")
        cmap = {r["stimulus_pair"]: r for r in conf}
        for sid, (g0, c0) in STEP1_CONTROLS.items():
            if sid in cmap:
                A(f"| `{sid}` | {fm(fl(cmap[sid],'gpt_match'))} | {g0:.3f} "
                  f"| {fm(fl(cmap[sid],'claude_match'))} | {c0:.3f} |")
        A("")

    A("## Figure\n")
    A("`targeted_stimulus_screen.png` / `.pdf` — (A) GPT vs Claude match for all "
      "candidates; (B) disparity effect sizes with bootstrap 95% CIs; (C) the "
      "form-consistency diagnostic, separating clean disparities from position "
      "fallback; (D) confirmation results for the selected battery.\n")

    A("## Outputs\n")
    for n in ("targeted_stimulus_manifest.csv", "targeted_screen_calls_*.csv",
              "targeted_screen_summary.csv", "targeted_screen_comparison.csv",
              "targeted_screen_selected.csv", "targeted_confirmation_calls_*.csv",
              "targeted_confirmation_summary.csv", "targeted_confirmation_comparison.csv",
              "targeted_stimulus_screen.png / .pdf"):
        A(f"- `{n}`")
    A("")

    path = os.path.join(OUT, "targeted_stimulus_screen_report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
