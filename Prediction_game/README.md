# Name Game — Prediction-Only Variant

A prediction-only extension of the [Name Game coordination study](../README.md).
Here the LLM is an **observer**, not a player: it never chooses a letter for
itself. Instead it sees a fixed, partial history of a game played by 6 humans
and is asked to **predict** how those humans will behave in the remaining rounds.

**Research question:** Do LLMs *underestimate* humans' willingness to abandon an
established convention (J) in favour of a newly-available, higher-value
convention (M) once partial evidence of switching appears?

**Authors:** Rachel Calcott, Anita Braga

---

## The game shown to the model

- 6 human players, no communication.
- **Rounds 1–7:** all 6 humans chose **J** every round.
- **Rounds 8–14 payoffs:** all-J = 10 pts, all-M = 15 pts, all-N = 15 pts; every
  other outcome (any mismatch) = 0 pts.
- **Round 8** is the experimental manipulation: its results vary how much
  evidence of switching toward M the model has observed before it predicts.

The model is asked, for each remaining round (9–14):

1. Probabilities (0–100) that a human chooses **J / M / N / Other**, summing to 100.
2. The single most likely **group outcome**: `all_J`, `all_M`, `all_N`, or `mixed`.

---

## Conditions (round-8 evidence)

| Condition    | Round-8 result            | Switching evidence            |
|--------------|---------------------------|-------------------------------|
| `r8_0M`      | 6 J, 0 M, 0 N             | none (convention held)        |
| `r8_1M`      | 5 J, 1 M, 0 N             | weak                          |
| `r8_2M`      | 4 J, 2 M, 0 N             | moderate                      |
| `r8_3M`      | 3 J, 3 M, 0 N             | even split J/M                |
| `r8_4M`      | 2 J, 4 M, 0 N             | strong                        |
| `r8_splitMN` | 4 J, 1 M, 1 N             | ambiguous (both M and N seen) |

`r8_0M` … `r8_4M` form an increasing-evidence gradient; `r8_splitMN` probes
whether a split between the two high-value letters (M and N) suppresses the
predicted switch.

---

## Design summary

| Parameter            | Value                                              |
|----------------------|----------------------------------------------------|
| Model                | `gpt-5.4` (OpenAI Responses API)                   |
| Role of the model    | Observer / forecaster (does **not** play)          |
| Conditions           | 6 (see table above)                                |
| Runs per condition   | 20                                                 |
| Total model calls    | 120                                                |
| Predicted rounds     | 9–14                                               |
| Output capture       | OpenAI Structured Outputs (strict JSON schema)     |
| Execution order      | Shuffled across (condition × run) to spread drift  |
| Rate-limit pacing    | 21 s between calls (≈ 3 RPM, matches player study)  |

---

## Repository structure

```
Prediction_game/
├── README.md                              # this file
├── experiment/
│   └── Name_Game_prediction_MN.py         # the experiment script
├── results/                               # output (timestamped, git-ignored data)
│   ├── name_game_prediction_MN_<ts>.csv   # tidy long table (1 row / condition×run×round)
│   └── name_game_prediction_MN_<ts>.jsonl # raw: full parsed prediction + reasoning trace
└── analysis/
    └── analyze_prediction_results.R       # loads newest CSV, summarises + plots
```

---

## Setup

Requires Python 3.9+ and an OpenAI API key in a `.env` file at the **repo root**
(`../.env`, already git-ignored — the key is never stored in source):

```
OPENAI_API_KEY=your_key_here
```

```bash
pip install openai python-dotenv
```

---

## Run the experiment

From the repo root:

```bash
python Prediction_game/experiment/Name_Game_prediction_MN.py
```

Results are written automatically to `Prediction_game/results/` as a timestamped
CSV (tidy, for analysis) and JSONL (raw, full fidelity including the model's
reasoning summary). At ~21 s/call × 120 calls the run takes ~40 minutes; adjust
`RUNS_PER_CONDITION` or `SLEEP_SECONDS` at the top of the script to change this.

---

## Analyse

```r
# from Prediction_game/analysis/ (or the repo root)
source("analyze_prediction_results.R")
```

The R script loads the most recent results CSV and reports:

- mean predicted **P(choose M)** by condition × round,
- the proportion of runs predicting a full switch (**all_M**) by condition × round,
- pooled headline metrics per condition,
- three plots (printed to the active device): P(choose M) trajectories,
  all_M probability trajectories, and a condition-level summary bar chart.

It also runs a data-quality check that each round's probabilities sum to 100.

---

## Output schema (tidy CSV)

| Column              | Meaning                                                       |
|---------------------|--------------------------------------------------------------|
| `model`             | Model id (`gpt-5.4`)                                          |
| `condition`         | Round-8 evidence condition                                   |
| `run`               | Replicate index within condition (1–20)                      |
| `round`             | Predicted round (9–14)                                        |
| `prob_J/M/N/Other`  | Predicted probability (0–100) a human picks that category    |
| `prob_sum`          | Sum of the four probabilities (QA; should equal 100)         |
| `group_outcome`     | Predicted group outcome: `all_J` / `all_M` / `all_N` / `mixed` |
| `reasoning_summary` | Model's reasoning-summary trace for the run (may be empty)    |
| `run_timestamp`     | Timestamp of the model call                                  |

---

## Relationship to the player study

This shares the payoff structure and the J→M/N phase shift of the original
[player-based Name Game](../Name_game_experiment/), but isolates the
*prediction* component: rather than measuring whether an LLM *switches*, it
measures whether an LLM *expects others to switch*. Comparing predicted
switching here against observed human/LLM switching in the player data tests the
underestimation hypothesis directly.
