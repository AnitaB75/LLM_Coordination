"""
Build one self-contained folder per LLM Name Game experiment.

Each experiment gets:

    experiments/<launch-date>_<slug>/
        code/            <- runner(s), stimulus generator, analysis for this pilot
        data/            <- result CSVs + run logs
        visualizations/  <- figures produced from this pilot
        README.md        <- what it tested, what it found, how to re-run

Launch dates come from the timestamp embedded in each result filename
(`*_YYYYMMDD_HHMMSS.csv`), which is written by the runner at launch.

Stimulus images live in the shared `stimuli/` tree and are NOT duplicated into
each folder (they are large and shared across pilots); each README names the
stimulus sets it used.

NON-DESTRUCTIVE: only copies. Nothing in the source tree is moved or deleted.

Run:  python3 tools/build_experiment_folders.py [--force]
"""
import os, glob, shutil, sys, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
OUT_ROOT = os.path.join(ROOT, "experiments")
FORCE = "--force" in sys.argv

E = [
    dict(date="2026-05-07", slug="llm_feedback_MN",
         title="LLM feedback - M/N payoff shift",
         tests="The original LLM run: 6 agents, 14 rounds, J then M/N. Includes the "
               "human-partner framing variant.",
         finding="Baseline for everything that follows.",
         code=["Name_Game_LLM_feedback_MN.py", "Name_Game_LLM_feedback_MN_HUMAN_ONLY.py",
               "Name_Game_LLM_feedback_MIXED.py"],
         data_glob=[os.path.join(REPO, "results", "name_game_LLM_feedback_MN*.csv")],
         analysis=[os.path.join(REPO, "Name_game_analysis_scripts", "analyze_name_game_results.py")],
         stimuli="none (text letters)"),
    dict(date="2026-06-27", slug="bot_follow_human",
         title="Bot follows human",
         tests="Whether an LLM agent follows a human partner's lead.",
         finding="See run_bot_follow_human.log.",
         code=["Name_Game_bot_follow_human.py"],
         data_glob=["results/name_game_bot_follow_human_*.csv", "run_bot_follow_human.log"],
         viz=["llm_jn_choices.png"], analysis=["plot_jn_choices.R"],
         stimuli="none (text letters)"),
    dict(date="2026-07-17", slug="claude_3cond",
         title="Claude, 3 partner conditions",
         tests="Single-model precursor to the both-models 3-condition run.",
         finding="Superseded by the 2026-07-18 both-models version.",
         code=["Name_Game_Claude_3cond.py"],
         data_glob=["results/name_game_claude_3cond_*.csv", "run_claude_3cond.log",
                    "run_claude_mixed.log", "run_mixed.log"],
         stimuli="none (text letters)"),
    dict(date="2026-07-18", slug="belief_3cond",
         title="Belief manipulation x 3 partner conditions",
         tests="self / mixed(self+human) / human framings, n=10 per cell, both models.",
         finding="Abandoning J is a SELF-RECOGNITION effect: self -> M 10/10 for both "
                 "models; human -> J. Claude monotonic (10/3/0); GPT high then drops.",
         code=["Name_Game_both_3cond.py"],
         data_glob=["results/name_game_3cond_*.csv", "run_3cond_*.log"],
         viz=["figures/fig2_belief_manipulation.png"],
         stimuli="none (text letters)"),
    dict(date="2026-07-19", slug="generic_ai_mix",
         title="Generic-AI partner mix",
         tests="Framing partners as 'a mix of AI models + humans'.",
         finding="mixed_ai ~= human for both models. Generic-AI partners are NOT "
                 "treated like copies-of-self.",
         code=["Name_Game_both_mixedAI.py"],
         data_glob=["results/name_game_mixedAI_*.csv", "run_mixedAI_*.log"],
         stimuli="none (text letters)"),
    dict(date="2026-07-19", slug="hack_random_strings",
         title="Hack: random strings",
         tests="Replace M/N with random 3-letter strings, self-play.",
         finding="5/5 hacked. Models alphabetize noise (Claude) or take first-listed (GPT).",
         code=["Name_Game_strings_selfplay.py"],
         data_glob=["results/name_game_strings_selfplay_*.csv", "run_strings_*.log"],
         stimuli="none (generated strings)"),
    dict(date="2026-07-19", slug="hack_car_makes",
         title="Hack: car makes",
         tests="Replace M/N with six car-make names.",
         finding="5/5 hacked - alphabetical order / brand prominence.",
         code=["Name_Game_carmakes_selfplay.py"],
         data_glob=["results/name_game_carmakes_selfplay_*.csv", "run_carmakes_*.log"],
         stimuli="none (text names)"),
    dict(date="2026-07-19", slug="visual_baseline",
         title="Visual baseline (position available)",
         tests="Circle/square and blob pairs with position information intact.",
         finding="Models coordinate on POSITION (leftmost), even for circle/square.",
         code=["visual_probe.py", "make_shape_stimuli.py"],
         data_glob=["results/visual_probe_2026*.csv", "run_visual_probe.log"],
         stimuli="stimuli/visual"),
    dict(date="2026-07-19", slug="visual_position_removed",
         title="Visual, position randomized",
         tests="Same shapes, told position is randomized per player.",
         finding="circle->circle 12/12 (prototypicality). Mirror blobs weak; models disagree.",
         code=["visual_probe_posrand.py"],
         data_glob=["results/visual_probe_posrand_*.csv", "run_visual_posrand.log"],
         stimuli="stimuli/visual"),
    dict(date="2026-07-19", slug="inkblots_and_noise",
         title="Inkblots + noise textures",
         tests="Internally-symmetric blobs and random textures, position randomized.",
         finding="Inkblots SPLIT the models - GPT hacks via global symmetry, Claude "
                 "cannot. Noise is weak for both.",
         code=["visual2_posrand_probe.py"],
         data_glob=["results/visual2_posrand_*.csv", "run_visual2_posrand.log"],
         stimuli="stimuli/visual2"),
    dict(date="2026-07-19", slug="nondescript_mountains",
         title="Nondescript mountains",
         tests="Matched procedural peaks, with and without position randomization.",
         finding="~90% - both models pick the 'most distinctive peak' and agree.",
         code=["mountain_probe.py", "mountain_posrand_probe.py"],
         data_glob=["results/mountain_*.csv", "run_mountain*.log"],
         stimuli="stimuli/mountains"),
    dict(date="2026-07-19", slug="five_landscapes",
         title="Five landscapes",
         tests="Five visually distinct scenes; pick one.",
         finding="100% mountains (salience). The most hackable stimulus set tested.",
         code=["landscape_probe.py"],
         data_glob=["results/landscape_probe_*.csv", "run_landscape.log"],
         stimuli="stimuli/landscapes"),
    dict(date="2026-07-19", slug="generic_cars",
         title="Generic car silhouettes",
         tests="Six unnamed car silhouettes - names stripped, shape cue remains.",
         finding="~65% van (largest).",
         code=["car_probe.py"],
         data_glob=["results/car_probe_*.csv", "run_car_probe.log"],
         stimuli="stimuli/cars"),
    dict(date="2026-07-19", slug="roundabout_solution",
         title="Roundabout (the un-hackable design)",
         tests="Private per-seat rotation+mirror; convention exit = 10, two symmetric "
               "15-point exits.",
         finding="0/10 leaks - un-hackable by construction. The group holds the "
                 "convention because the two 15-exits cannot be coordinated a priori.",
         code=["Name_Game_roundabout_selfplay.py", "analyze_roundabout.py"],
         data_glob=["results/roundabout_selfplay_*.csv", "run_roundabout_*.log"],
         viz=["figures/fig3_hack_and_solution.png"],
         stimuli="stimuli/diagrams"),
    dict(date="2026-07-19", slug="roundabout_devalued",
         title="Roundabout, convention devalued",
         tests="Same design, but the convention drops to 3 points in rounds 8-14 - a 5x "
               "incentive to switch.",
         finding="Still 0/10 leaks. Self-play does NOT initiate the costly first move, "
                 "unlike humans in the mutual-observability study.",
         code=["Name_Game_roundabout_devalued.py"],
         data_glob=["results/roundabout_devalued_*.csv", "run_rb_dev_*.log"],
         stimuli="stimuli/diagrams"),
    # --- Prediction game: the LLM observes, it does not play -----------------
    # Lives in ../Prediction_game/, so paths reach up out of Name_game_experiment.
    dict(date="2026-06-15", slug="prediction_game_baseline",
         title="Prediction game - baseline",
         tests="gpt-5.4 as an OBSERVER, not a player: shown rounds 1-7 of all-J play "
               "by 6 humans, asked to predict rounds 9-14. Six round-8 evidence "
               "conditions (0M/1M/2M/3M/4M/splitMN) x 20 runs. Paired with a "
               "play-condition run of the same payoff structure.",
         finding="Baseline for the underestimation hypothesis - does an LLM expect "
                 "humans to hold the J convention longer than they actually do?",
         code=["../Prediction_game/experiment/Name_Game_prediction_MN.py",
               "../Prediction_game/experiment/Name_Game_play_MN.py"],
         analysis=["../Prediction_game/analysis/analyze_prediction_results.R",
                   "../Prediction_game/analysis/visualize_prediction_results.R"],
         data_glob=["../Prediction_game/results/name_game_prediction_MN_20260615_*",
                    "../Prediction_game/results/name_game_play_MN_20260615_*",
                    "../Prediction_game/results/run_20260615_*.log",
                    "../Prediction_game/results/play_run_20260615_*.log"],
         stimuli="none (text letters, history shown in prompt)"),
    dict(date="2026-06-19", slug="prediction_round9_gpt54",
         title="Prediction pilot - round 9, gpt-5.4",
         tests="Single-round (round 9) prediction pilot, gpt-5.4. Two runs.",
         finding="Pilot preceding the round-8/9 variant sweep.",
         code=["../Prediction_game/experiment/pilot_round9_gpt54.py"],
         data_glob=["../Prediction_game/results/pilot_round9_gpt54_20260619_*"],
         stimuli="none (text letters)"),
    dict(date="2026-06-23", slug="prediction_round9_variants",
         title="Prediction pilots - round 9 variants (model x self-framing)",
         tests="Round-9 prediction across variants: Claude, gpt-5.4 predicting ITS OWN "
               "future choices (`_self`), and an round-8-prediction variant.",
         finding="Varies who is being predicted (humans vs the model itself) while "
                 "holding the round structure fixed.",
         code=["../Prediction_game/experiment/pilot_round9_claude.py",
               "../Prediction_game/experiment/pilot_round9_gpt54_self.py",
               "../Prediction_game/experiment/pilot_round8pred_gpt54.py",
               "../Prediction_game/experiment/pilot_round89_gpt54_self.py"],
         analysis=["../Prediction_game/analysis/analyze_prediction_results.R"],
         data_glob=["../Prediction_game/results/pilot_round9_claude_20260623_*",
                    "../Prediction_game/results/pilot_round9_gpt54_self_20260623_*",
                    "../Prediction_game/results/pilot_round8pred_gpt54_20260623_*",
                    "../Prediction_game/results/pilot_round89_gpt54_self_20260623_*"],
         stimuli="none (text letters)"),
    dict(date="2026-06-26", slug="prediction_round9_reward",
         title="Prediction pilot - round 9, reward manipulation (Claude)",
         tests="Round-9 prediction with the reward level varied, Claude.",
         finding="See reward_pilot_run.log.",
         code=["../Prediction_game/experiment/pilot_round9_claude_reward.py"],
         data_glob=["../Prediction_game/results/pilot_round9_claude_reward_20260626_*",
                    "../Prediction_game/results/reward_pilot_run.log"],
         stimuli="none (text letters)"),
    dict(date="2026-07-09", slug="prediction_game_fable",
         title="Prediction game - Fable",
         tests="The 2026-06-15 baseline prediction design re-run on Fable.",
         finding="Cross-model check on the prediction baseline.",
         code=["../Prediction_game/experiment/Name_Game_prediction_MN_fable.py"],
         analysis=["../Prediction_game/analysis/analyze_prediction_results.R"],
         data_glob=["../Prediction_game/results/name_game_prediction_MN_fable_20260709_*",
                    "../Prediction_game/results/fable_run.log"],
         stimuli="none (text letters)"),

    # --- Qualtrics: real human participants predicting -----------------------
    dict(date="2026-07-02", slug="human_predictions_study1",
         title="Human predictions, study 1 (Qualtrics)",
         tests="HUMAN participants shown the same Name Game history the LLM observer "
               "saw, asked to predict how the group behaves after the payoff shift. "
               "The human side of the underestimation comparison.",
         finding="Human baseline against which the LLM prediction runs are scored.",
         code=["../qualtrics_human_study/generate_stimuli.py"],
         analysis=["../Prediction_game/analysis/analyze_human_predictions.R"],
         analysis_dir="../qualtrics_human_study/data/study1_predictions",
         stimuli="qualtrics_human_study/stimuli (shared, not duplicated)"),
    dict(date="2026-07-07", slug="human_predictions_study2",
         title="Human predictions, study 2 (Qualtrics)",
         tests="Second wave, adding the reward-level (15/20/50) and AI-framing "
               "stimulus variants.",
         finding="Extends study 1 with payoff-magnitude and partner-framing arms.",
         code=["../qualtrics_human_study/generate_stimuli_rewards.py",
               "../qualtrics_human_study/generate_stimuli_ai.py",
               "../qualtrics_human_study/generate_stimuli_midplacement.py"],
         analysis=["../Prediction_game/analysis/analyze_human_predictions.R"],
         analysis_dir="../qualtrics_human_study/data/study2_predictions",
         stimuli="qualtrics_human_study/stimuli/{reward_15,reward_20,reward_50,ai_framing,midplacement}"),

    # --- Convention entrenchment series (August) -----------------------------
    dict(date="2026-08-04", slug="entrenchment_pilot",
         title="Convention entrenchment - pilot (M/N)",
         tests="Does the number of J-only rounds BEFORE an unannounced payoff shift "
               "determine whether the group abandons J? Conditions j7/j4/j1 (7, 4, or "
               "1 J-only round), post-shift phase held at 7 rounds. Human framing "
               "throughout. Both providers, 3 trials.",
         finding="First pass at the entrenchment question; superseded by the singleM "
                 "variant, which removes the M-vs-N coordination problem.",
         code=["Name_Game_entrenchment_human.py"],
         analysis=["analyze_entrenchment.R", "fig_entrenchment.R"],
         data_glob=["results/name_game_entrenchment_claude_20260804_*.csv",
                    "results/name_game_entrenchment_gpt_20260804_*.csv",
                    "entrenchment_claude.log", "entrenchment_gpt.log"],
         viz=["../fig_entrenchment_human.png"],
         stimuli="none (text letters)"),
    dict(date="2026-08-04", slug="entrenchment_singleM",
         title="Convention entrenchment - single-M variant",
         tests="Same entrenchment design with ONE high-value letter (M) instead of two "
               "(M/N), removing the confound that switching requires agreeing on WHICH "
               "new letter. Includes the first payoff sweep.",
         finding="Isolates entrenchment from the M-vs-N coordination problem.",
         code=["Name_Game_entrenchment_human.py"],
         analysis=["analyze_entrenchment.R"],
         data_glob=["results/name_game_entrenchment_singleM_gpt_20260804_*.csv",
                    "entrenchment_singleM_claude.log", "entrenchment_singleM_gpt.log",
                    "entrenchment_sweep_gpt.log"],
         stimuli="none (text letters)"),
    dict(date="2026-08-05", slug="entrenchment_payoff_sweep",
         title="Convention entrenchment - payoff sweep (gpt-5.5)",
         tests="Convention length (j7/j4/j1) crossed with the new option's payoff "
               "(M = 15, 20, or 40 vs J = 10; gaps of 1.5x, 2x, 4x). 9 trials per run. "
               "Extended with a second sweep on 2026-08-06.",
         finding="Maps where the convention breaks as a function of both how "
                 "entrenched it is and how much is on offer.",
         code=["Name_Game_entrenchment_human.py"],
         analysis=["fig_academic.R", "fig_two_panel.R", "fig_pilot_summary.R"],
         data_glob=["results/name_game_entrenchment_singleM_gpt55_2026080[56]_*.csv",
                    "entrenchment_sweep_gpt55.log", "entrenchment_sweep_gpt55_b.log"],
         viz=["../fig_entrenchment_academic.png", "../fig_entrenchment_2panel.png",
              "../fig_pilot_summary.png"],
         stimuli="none (text letters)"),
    dict(date="2026-08-06", slug="entrenchment_replication_grid",
         title="Convention entrenchment - full replication grid",
         tests="The powered version: convention length (j7/j4/j1) x model (gpt-5.4, "
               "gpt-5.5) x payoff (M = 15, 20, 40), 10 trials per cell.",
         finding="Replication and power for the entrenchment x payoff surface; "
                 "fig_defection_timecourse shows the round-by-round cascade.",
         code=["Name_Game_entrenchment_human.py"],
         analysis=["analyze_replication.R", "fig_full_grid.R",
                   "fig_defection_timecourse.R", "fig_simple.R"],
         data_glob=["results/name_game_entrenchment_singleM_gpt5[45]_j*_*.csv",
                    "rep_j*.log", "rep54_j*.log", "rep_gpt5[45]_j*.log"],
         viz=["../fig_replication.png", "../fig_full_grid.png",
              "../fig_defection_timecourse.png", "../fig_simple.png"],
         stimuli="none (text letters)"),

    # --- Cross-model stimulus screening --------------------------------------
    dict(date="2026-08-04", slug="stimulus_screen_step1",
         title="Step 1 / 1b - cross-model stimulus screen",
         tests="GPT vs Claude choosing from image pairs, measuring BETWEEN-MODEL match "
               "rate. Step 1 fixes the form-choice procedure; Step 1b screens a 30-pair "
               "battery for stimuli that dissociate the two models (position bias in "
               "[0.35, 0.65] for both, |match_gpt - match_claude| > 0.15, crossovers in "
               "both directions).",
         finding="Selects the stimulus set used by downstream cross-model coordination "
                 "runs. Screening study, not a coordination game in itself.",
         code=["step1_form_choice.py", "step1_analysis.py", "step1_choice_distribution.py",
               "step1_form_analysis.py", "step1_form_export.py",
               "step1b_screen.py", "step1b_screen_analysis.py", "step1b_screen_figure.py",
               "make_screen_stimuli.py", "make_single_option_stimuli.py",
               "build_report.py", "build_summary_pdf.py", "theme_academic.R"],
         analysis_dir="step1_output",
         stimuli="stimuli/screen, stimuli/singles"),

    # The three self-coordination experiments (2026-07-30 selfcoord_identity,
    # 2026-08-01 confidence_calibration, 2026-08-02 partner_specific_coordination)
    # moved out on 2026-08-04 to ../../Self_Coordination/, which is now its own
    # project with its own built experiment folders.
]

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def copy_one(src, dest_dir):
    if not os.path.exists(src):
        return None
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(src))
    if os.path.isdir(src):
        shutil.copytree(src, dest, ignore=IGNORE, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dest)
    return os.path.basename(src)


def build(exp):
    name = f"{exp['date']}_{exp['slug']}"
    base = os.path.join(OUT_ROOT, name)
    if os.path.exists(base):
        if not FORCE:
            print(f"  skip (exists)  {name}")
            return
        shutil.rmtree(base)
    for sub in ("code", "data", "visualizations"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    got = {"code": [], "data": [], "visualizations": []}
    for f in exp.get("code", []):
        r = copy_one(os.path.join(ROOT, f), os.path.join(base, "code"))
        if r: got["code"].append(r)
    for f in exp.get("analysis", []):
        r = copy_one(os.path.join(ROOT, f), os.path.join(base, "code"))
        if r: got["code"].append(r)
    # An analysis directory is fanned out by file type so it lands in the right
    # subfolder rather than burying CSVs and figures under code/.
    if exp.get("analysis_dir"):
        adir = os.path.join(ROOT, exp["analysis_dir"])
        for fn in sorted(os.listdir(adir)):
            ext = os.path.splitext(fn)[1].lower()
            sub = ("data" if ext in (".csv", ".tsv", ".jsonl")
                   else "visualizations" if ext in (".pdf", ".png", ".svg")
                   else "code")
            r = copy_one(os.path.join(adir, fn), os.path.join(base, sub))
            if r: got[sub].append(r)
    for pat in exp.get("data_glob", []):
        for src in sorted(glob.glob(pat if os.path.isabs(pat) else os.path.join(ROOT, pat))):
            r = copy_one(src, os.path.join(base, "data"))
            if r: got["data"].append(r)
    for f in exp.get("viz", []):
        r = copy_one(os.path.join(ROOT, f), os.path.join(base, "visualizations"))
        if r: got["visualizations"].append(r)

    # Runners resolve stimuli relative to their own file, so the sets they use are
    # copied in beside them. The whole tree is only ~2.4 MB, and this is what makes
    # each folder runnable on its own.
    sets = [s.strip() for s in exp["stimuli"].split(",") if s.strip().startswith("stimuli/")]
    for s in sets:
        src = os.path.join(ROOT, s)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(base, "code", "stimuli", os.path.basename(s)),
                            ignore=IGNORE, dirs_exist_ok=True)
            got.setdefault("stimuli", []).append(os.path.basename(s))

    for sub in ("code", "data", "visualizations"):
        d = os.path.join(base, sub)
        if not os.listdir(d):
            open(os.path.join(d, ".gitkeep"), "w").close()

    write_readme(base, exp, got)
    missing = "" if all(got.values()) else "  (some sections empty)"
    print(f"  built  {name}   code:{len(got['code'])} data:{len(got['data'])} viz:{len(got['visualizations'])}{missing}")


def write_readme(base, exp, got):
    L = [f"# {exp['title']}", "", f"`{exp['date']}` · {exp['slug']}", "",
         "## What it tested", "", exp["tests"], "",
         "## What it found", "", exp["finding"], "",
         "## Layout", "",
         "| folder | contents |", "|---|---|",
         "| `code/` | runner(s) and analysis for this pilot |",
         "| `data/` | result CSVs and run logs |",
         "| `visualizations/` | figures |", "",
         f"Stimuli: {exp['stimuli']}", ""]
    if got.get("stimuli"):
        L += [f"The stimulus sets this pilot uses are copied into `code/stimuli/` "
              f"({', '.join(got['stimuli'])}), so the runner works from this folder "
              f"without reaching back into the shared tree.", ""]
    for k, label in (("code", "Code"), ("data", "Data"), ("visualizations", "Visualizations")):
        L += [f"## {label}", ""]
        L += ([f"- `{k}/{f}`" for f in got[k]] if got[k] else ["_none_"])
        L += [""]
    if got["code"]:
        runner = next((f for f in got["code"] if f.endswith(".py")), None)
        if runner:
            L += ["## Re-run", "", "```bash",
                  "# needs OPENAI_API_KEY and ANTHROPIC_API_KEY in the repo-root .env",
                  f"cd code && python3 {runner}", "```", ""]
    L += ["---", "",
          f"_Generated by `tools/build_experiment_folders.py` on "
          f"{datetime.date.today().isoformat()}. Source of truth is "
          f"`Name_game_experiment/`; this folder is a copy._", ""]
    with open(os.path.join(base, "README.md"), "w") as fh:
        fh.write("\n".join(L))


def write_index():
    L = ["# LLM Name Game experiments", "",
         "One folder per pilot, named by launch date (taken from the timestamp the "
         "runner wrote into its result filename). Each holds `code/`, `data/`, and "
         "`visualizations/`, plus the stimulus sets it needs — so any folder can be "
         "copied out and run on its own.", "",
         "| launched | experiment | what it found |", "|---|---|---|"]
    for exp in sorted(E, key=lambda x: x["date"]):
        first = exp["finding"].split(". ")[0].rstrip(".")
        L.append(f"| `{exp['date']}` | [{exp['title']}]({exp['date']}_{exp['slug']}/) | {first} |")
    L += ["", "Rebuild with `python3 tools/build_experiment_folders.py --force`.",
          "Source of truth stays `Name_game_experiment/`; these are copies.", ""]
    with open(os.path.join(OUT_ROOT, "README.md"), "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"building into {OUT_ROOT}")
    for exp in E:
        build(exp)
    write_index()
    print("  wrote experiments/README.md")
