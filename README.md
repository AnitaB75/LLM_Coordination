# LLM Coordination

Research project investigating coordination behavior in large language models (LLMs) using a multi-agent Name Game paradigm. We ask whether LLMs can coordinate on shared conventions, how they respond to partner identity (self, human, other LLM, neutral), and whether coordination strategies shift when payoff structures change mid-game.

**Authors:** Rachel Calcott, Anita Braga

---

## Overview

Six GPT agents play a repeated coordination game across 14 rounds. Players simultaneously choose a letter and earn points only when all players match. In rounds 1–7, one letter (J) dominates; in rounds 8–14, two new higher-value letters (M, N) become available. This phase shift tests whether agents can abandon an established convention when a better one becomes available.

A key manipulation is **partner condition**: agents are told their co-players are other instances of themselves, human participants, other LLMs, or neutral "other participants." This allows us to test whether perceived partner identity shapes coordination strategy and willingness to switch conventions.

---

## Repository Structure

```
LLM_Coordination/
│
├── Name_game_experiment/          # Experiment scripts
│   ├── Name_Game_LLM_feedback_MN.py             # Main experiment (LLM-only, all 4 conditions)
│   ├── Name_Game_LLM_feedback_MN_HUMAN_ONLY.py  # Human partner condition variant
│   ├── Name_Game_reasoning_openAI_participants_implicit.py  # Earlier reasoning variant
│   └── Name_Game_two_letters_no_feedback_openAI_participants.py  # No-feedback variant
│
├── Name_game_analysis_scripts/    # Analysis scripts
│   ├── analyze_name_game_results.py              # Python analysis script
│   └── name_game_analysis/
│       ├── analyze_name_game_by_condition.R      # R analysis by partner condition
│       ├── name_game_LLM_feedback_MN_20260507_134935.csv
│       └── name_game_LLM_feedback_MN_HUMAN_20260507_212952.csv
│
├── results/                       # Raw output data
│   ├── name_game_LLM_feedback_MN_20260507_134935.csv     # LLM-only runs
│   ├── name_game_LLM_feedback_MN_HUMAN_20260507_212952.csv  # Human condition runs
│   ├── name_game_reasoning_qwen_results_part_1.csv       # Qwen model results
│   ├── name_game_reasoning_qwen_results_part_2.csv
│   └── name_game_reasoning_qwen_results_part_3.csv
│
├── analyze_name_game_results.R    # Full R analysis script (run from repo root)
├── name_game_analysis.png         # Summary visualization
└── .gitignore
```

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
