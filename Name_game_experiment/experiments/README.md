# LLM Name Game experiments

One folder per pilot, named by launch date (taken from the timestamp the runner wrote into its result filename). Each holds `code/`, `data/`, and `visualizations/`, plus the stimulus sets it needs — so any folder can be copied out and run on its own.

| launched | experiment | what it found |
|---|---|---|
| `2026-05-07` | [LLM feedback - M/N payoff shift](2026-05-07_llm_feedback_MN/) | Baseline for everything that follows |
| `2026-06-15` | [Prediction game - baseline](2026-06-15_prediction_game_baseline/) | Baseline for the underestimation hypothesis - does an LLM expect humans to hold the J convention longer than they actually do? |
| `2026-06-19` | [Prediction pilot - round 9, gpt-5.4](2026-06-19_prediction_round9_gpt54/) | Pilot preceding the round-8/9 variant sweep |
| `2026-06-23` | [Prediction pilots - round 9 variants (model x self-framing)](2026-06-23_prediction_round9_variants/) | Varies who is being predicted (humans vs the model itself) while holding the round structure fixed |
| `2026-06-26` | [Prediction pilot - round 9, reward manipulation (Claude)](2026-06-26_prediction_round9_reward/) | See reward_pilot_run.log |
| `2026-06-27` | [Bot follows human](2026-06-27_bot_follow_human/) | See run_bot_follow_human.log |
| `2026-07-02` | [Human predictions, study 1 (Qualtrics)](2026-07-02_human_predictions_study1/) | Human baseline against which the LLM prediction runs are scored |
| `2026-07-07` | [Human predictions, study 2 (Qualtrics)](2026-07-07_human_predictions_study2/) | Extends study 1 with payoff-magnitude and partner-framing arms |
| `2026-07-09` | [Prediction game - Fable](2026-07-09_prediction_game_fable/) | Cross-model check on the prediction baseline |
| `2026-07-17` | [Claude, 3 partner conditions](2026-07-17_claude_3cond/) | Superseded by the 2026-07-18 both-models version |
| `2026-07-18` | [Belief manipulation x 3 partner conditions](2026-07-18_belief_3cond/) | Abandoning J is a SELF-RECOGNITION effect: self -> M 10/10 for both models; human -> J |
| `2026-07-19` | [Generic-AI partner mix](2026-07-19_generic_ai_mix/) | mixed_ai ~= human for both models |
| `2026-07-19` | [Hack: random strings](2026-07-19_hack_random_strings/) | 5/5 hacked |
| `2026-07-19` | [Hack: car makes](2026-07-19_hack_car_makes/) | 5/5 hacked - alphabetical order / brand prominence |
| `2026-07-19` | [Visual baseline (position available)](2026-07-19_visual_baseline/) | Models coordinate on POSITION (leftmost), even for circle/square |
| `2026-07-19` | [Visual, position randomized](2026-07-19_visual_position_removed/) | circle->circle 12/12 (prototypicality) |
| `2026-07-19` | [Inkblots + noise textures](2026-07-19_inkblots_and_noise/) | Inkblots SPLIT the models - GPT hacks via global symmetry, Claude cannot |
| `2026-07-19` | [Nondescript mountains](2026-07-19_nondescript_mountains/) | ~90% - both models pick the 'most distinctive peak' and agree |
| `2026-07-19` | [Five landscapes](2026-07-19_five_landscapes/) | 100% mountains (salience) |
| `2026-07-19` | [Generic car silhouettes](2026-07-19_generic_cars/) | ~65% van (largest) |
| `2026-07-19` | [Roundabout (the un-hackable design)](2026-07-19_roundabout_solution/) | 0/10 leaks - un-hackable by construction |
| `2026-07-19` | [Roundabout, convention devalued](2026-07-19_roundabout_devalued/) | Still 0/10 leaks |
| `2026-08-04` | [Convention entrenchment - pilot (M/N)](2026-08-04_entrenchment_pilot/) | First pass at the entrenchment question; superseded by the singleM variant, which removes the M-vs-N coordination problem |
| `2026-08-04` | [Convention entrenchment - single-M variant](2026-08-04_entrenchment_singleM/) | Isolates entrenchment from the M-vs-N coordination problem |
| `2026-08-04` | [Step 1 / 1b - cross-model stimulus screen](2026-08-04_stimulus_screen_step1/) | Selects the stimulus set used by downstream cross-model coordination runs |
| `2026-08-05` | [Convention entrenchment - payoff sweep (gpt-5.5)](2026-08-05_entrenchment_payoff_sweep/) | Maps where the convention breaks as a function of both how entrenched it is and how much is on offer |
| `2026-08-06` | [Convention entrenchment - full replication grid](2026-08-06_entrenchment_replication_grid/) | Replication and power for the entrenchment x payoff surface; fig_defection_timecourse shows the round-by-round cascade |

Rebuild with `python3 tools/build_experiment_folders.py --force`.
Source of truth stays `Name_game_experiment/`; these are copies.
