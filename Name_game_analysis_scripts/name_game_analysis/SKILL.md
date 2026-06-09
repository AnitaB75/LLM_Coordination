# Name Game Coordination Analysis

## What this project is

A coordination game experiment studying how LLMs (and humans) converge on shared conventions when playing together. Participants choose a letter each round (J, M, or N) and earn points only when all group members match.

## Game structure

- **14 rounds** per trial, 6 players per group
- **Rounds 1–7**: only J earns points (10 pts if all match, 0 if not)
- **Rounds 8–14**: J still earns 10 pts; M and N now earn 15 pts — but only if all 6 players match
- The dashed vertical line at 7.5 in all plots marks this payoff shift
- Key question: do players switch away from the established J convention when M/N become more valuable?

## Partner conditions

| Condition    | Description                                      |
|-------------|--------------------------------------------------|
| `self`       | LLM playing with copies of itself                |
| `other_llm`  | LLM playing with a different LLM                 |
| `neutral`    | LLM playing with partners of unknown type        |
| `human`      | LLM playing with human participants              |

## Data files

| File | Contents |
|------|----------|
| `name_game_LLM_feedback_MN_20260507_134935.csv` | LLM conditions (self, other_llm, neutral) |
| `name_game_LLM_feedback_MN_HUMAN_20260507_212952.csv` | Human condition |

Both CSVs have the same schema. Key columns:
- `trial`: trial number
- `partner_condition`: one of the four conditions above
- `round`: 1–14
- `p1`–`p6`: letter choice (J/M/N) for each of the 6 players
- `p1_raw`–`p6_raw`: raw model output before parsing
- `p1_reasoning`–`p6_reasoning`: chain-of-thought reasoning text
- `matched`: whether all 6 players chose the same letter
- `points`: points earned that round
- `p1_total`–`p6_total`: cumulative score per player

## R analysis script

`analyze_name_game_by_condition.R` — loads both CSVs, binds them, and produces 5 visualizations:

1. **p1** — stacked bar chart: letter proportions by round, faceted by condition
2. **p2** — line chart: same data, lines per letter
3. **p3** — side-by-side bars: conditions in columns, letters dodged
4. **p4** — round 8 bar chart: letter choices at the critical switching point
5. **p5** — 2×2 panel line plot (main viz): proportion of each letter over all 14 rounds, one panel per condition

All plots save as PNGs in this folder. p5 is the primary figure.

## How to run

Open `analyze_name_game_by_condition.R` in RStudio and source it. Requires: `tidyverse`, `ggplot2`, `gridExtra`, `scales`.

## Python scripts (in parent folder)

- `Name_Game_LLM_feedback_MN.py` — runs the LLM-only experiment (generates the LLM CSV)
- `Name_Game_LLM_feedback_MN_HUMAN_ONLY.py` — runs the human condition experiment (generates the human CSV)
- `analyze_name_game_results.py` — earlier Python-based analysis (predates the R script)

## Adding new data

New CSVs go in this folder. In the R script, load them and add to the `bind_rows()` call at the top. If a new `partner_condition` value appears, it will automatically get its own panel in p5.
