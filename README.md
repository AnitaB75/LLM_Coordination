# LLM Coordination

Research project investigating coordination behavior in large language models (LLMs) using a multi-agent Name Game paradigm. We ask whether LLMs can coordinate on shared conventions, how they respond to partner identity (self, human, other LLM, neutral), and whether coordination strategies shift when payoff structures change mid-game.

**Authors:** Rachel Calcott, Anita Braga

---

## Overview

Six GPT agents play a repeated coordination game across 14 rounds. Players simultaneously choose a letter and earn points only when all players match. In rounds 1–7, one letter (J) dominates; in rounds 8–14, two new higher-value letters (M, N) become available. This phase shift tests whether agents can abandon an established convention when a better one becomes available.

A key manipulation is **partner condition**: agents are told their co-players are other instances of themselves, human participants, other LLMs, or neutral "other participants." This allows us to test whether perceived partner identity shapes coordination strategy and willingness to switch conventions.

---

## Experiments

**[`Name_game_experiment/experiments/`](Name_game_experiment/experiments/) is the
index of every experiment in this project** — 27 of them, 2026-05-07 to 2026-08-06,
one dated folder each holding `code/`, `data/`, `visualizations/` and a README.
Start there.

They fall into four lines of work:

| line | what varies | folders |
|---|---|---|
| **Partner identity** | who the agent is told it is playing with — self / human / other-LLM / generic-AI mix | `llm_feedback_MN`, `claude_3cond`, `belief_3cond`, `generic_ai_mix`, `bot_follow_human` |
| **Stimulus hacking** | whether the focal point can be found in the stimulus itself, rather than agreed on | the `2026-07-19` set: strings, car makes, visual probes, mountains, landscapes, inkblots, roundabout |
| **Prediction** | the LLM observes rather than plays, and forecasts others' switching — with real human participants as the comparison | `prediction_game_*`, `prediction_round9_*`, `human_predictions_study1/2` |
| **Convention entrenchment** | how many rounds the old convention ran before an *unannounced* payoff shift, crossed with how much the new option pays | `entrenchment_pilot`, `entrenchment_singleM`, `entrenchment_payoff_sweep`, `entrenchment_replication_grid` |

Self-coordination (does a model coordinate with copies of itself via a self-model
or via shared salience?) split out on 2026-08-04 into its own project at
[`../Self_Coordination/`](../Self_Coordination/). Human-participant oTree studies
live in [`../Name_Game/`](../Name_Game/).

---

## Repository Structure

```
LLM_Coordination/
│
├── Name_game_experiment/          # The LLM Name Game — working tree
│   ├── experiments/               #   >> the dated experiment index (start here)
│   ├── tools/
│   │   └── build_experiment_folders.py   # manifest that builds experiments/
│   ├── results/                   #   raw result CSVs, timestamped by the runner
│   ├── stimuli/                   #   shared stimulus image sets
│   ├── step1_output/              #   cross-model stimulus screen deliverables
│   ├── Name_Game_*.py             #   runners
│   ├── *_probe.py, step1*.py      #   stimulus probes and screening
│   └── fig_*.R, analyze_*.R       #   figures and analysis
│
├── Prediction_game/               # LLM as observer/forecaster, not player
│   ├── experiment/  analysis/  results/
│
├── qualtrics_human_study/         # Human participants doing the prediction task
│   ├── data/{study1,study2}_predictions/
│   ├── stimuli/                   #   incl. reward_15/20/50, ai_framing, midplacement
│   └── generate_stimuli*.py
│
├── Name_game_analysis_scripts/    # Earlier analysis scripts (May run)
├── data/                          # Third-party reference datasets (git-ignored)
├── results/                       # Earliest raw output (May run, Qwen runs)
└── analyze_name_game_results.R    # Full R analysis (run from repo root)
```

Each `experiments/<date>_<slug>/` folder is a **copy** — the working tree above is
the source of truth. Rebuild the copies with:

```bash
cd Name_game_experiment && python3 tools/build_experiment_folders.py          # add new only
cd Name_game_experiment && python3 tools/build_experiment_folders.py --force  # rebuild all
```

To add an experiment, append an entry to the `E` list in
`tools/build_experiment_folders.py` and re-run it. The index table is generated
from that list, so it cannot drift.

---

## Experimental Design

| Parameter | Value |
|---|---|
| Model | GPT-5.4 |
| Players per trial | 6 agents |
| Rounds per trial | 14 |
| Rounds 1–7 payoff | J = 10 pts, all others = 0 |
| Rounds 8–14 payoff | J = 10 pts, M = 15 pts, N = 15 pts |
| Partner conditions | `self`, `human`, `other_llm`, `neutral` |
| Trials | 4 (one per condition) |

Each agent sees: the current round, available letters and their payoffs, all players' choices from the previous round, points earned, and their own reasoning from prior rounds.

---

## Setup

**Requirements:**
- Python 3.9+
- An OpenAI API key in a `.env` file in the repo root:

```
OPENAI_API_KEY=your_key_here
```

**Install dependencies:**
```bash
pip install openai python-dotenv
```

**Run the main experiment:**
```bash
python Name_game_experiment/Name_Game_LLM_feedback_MN.py
```

Results are saved automatically to `results/` as timestamped CSV files.

---

## Analysis

**R analysis** (full pipeline — matching rates, letter distributions, switching behavior, convergence, visualizations):
```r
source("analyze_name_game_results.R")
```

**Python analysis:**
```bash
python Name_game_analysis_scripts/analyze_name_game_results.py
```

Both scripts load from the `results/` folder and produce summary statistics and plots.

---

## Key Questions

1. Do LLMs coordinate successfully under the baseline payoff structure?
2. Do they switch to the higher-value convention when M and N become available in round 8?
3. Does perceived partner identity (self vs. human vs. other LLM) affect coordination strategy or switching behavior?
4. What reasoning do agents give for their choices, particularly at the phase-shift decision point?
