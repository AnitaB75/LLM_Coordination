"""
Name Game — PREDICTION-ONLY variant
====================================

Research question
-----------------
Do LLMs UNDERESTIMATE humans' willingness to abandon an established convention
(J) in favour of a newly-available, higher-value convention (M) once partial
evidence of switching appears?

Unlike the player version of the Name Game (../../Name_game_experiment/),
the model here is an OBSERVER, not a player. It never chooses a letter for
itself. Instead it is shown a fixed, partial history of a game played by 6
humans and asked to PREDICT how those humans will behave in rounds 9-14.

Base game shown to the model
----------------------------
- 6 human players, no communication.
- Rounds 1-7: all 6 humans chose J every round.
- Rounds 8-14 payoff: all-J = 10 pts, all-M = 15 pts, all-N = 15 pts,
  every other outcome = 0 pts.
- Round 8 results are the experimental manipulation (see CONDITIONS below):
  they vary how much evidence of switching toward M the model has seen.

Experimental manipulation (round-8 evidence)
--------------------------------------------
  r8_0M       6 chose J, 0 chose M, 0 chose N   (no switching evidence)
  r8_1M       5 chose J, 1 chose M, 0 chose N
  r8_2M       4 chose J, 2 chose M, 0 chose N
  r8_3M       3 chose J, 3 chose M, 0 chose N
  r8_4M       2 chose J, 4 chose M, 0 chose N   (strong switching evidence)
  r8_splitMN  4 chose J, 1 chose M, 1 chose N   (ambiguous: M and N both appear)

Task asked of the model (per run)
---------------------------------
For each remaining round (9-14):
  * probabilities (0-100) that a human chooses J / M / N / Other, summing to 100
  * the single most likely group outcome: all_J / all_M / all_N / mixed

Output is captured with OpenAI Structured Outputs (a strict JSON schema) so the
numeric predictions parse cleanly. Results are written to ../results/ as:
  * a tidy long CSV  (one row per model x condition x run x round)
  * a raw JSONL      (one line per run: full parsed prediction + reasoning trace)

Setup
-----
Requires an OPENAI_API_KEY in a .env file at the repo root (already git-ignored).
    pip install openai python-dotenv

Run
---
    python Prediction_game/experiment/Name_Game_prediction_MN.py
"""

from dotenv import load_dotenv
from openai import OpenAI
import csv
import json
import os
import random
import time
from datetime import datetime

################################################################
# === Setup ===
################################################################
# Loads OPENAI_API_KEY from a .env file (searched in CWD and parent dirs).
# The repo-root .env is git-ignored; the key is never stored in source.
load_dotenv()

client = OpenAI()
MODEL = "gpt-5.4"

RUNS_PER_CONDITION = 20      # replicate predictions per condition
PREDICTED_ROUNDS = list(range(9, 15))  # rounds 9-14
SLEEP_SECONDS = 21           # rate-limit pacing (matches player experiment: ~3 RPM)
MAX_RETRIES = 3              # light retry on transient API errors

# Output directory: ../results relative to this script (CWD-independent).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

################################################################
# === Conditions (round-8 evidence) ===
################################################################
# Each condition is the count of the 6 humans choosing J / M / N in round 8.
CONDITIONS = {
    'r8_0M':      {'J': 6, 'M': 0, 'N': 0},
    'r8_1M':      {'J': 5, 'M': 1, 'N': 0},
    'r8_2M':      {'J': 4, 'M': 2, 'N': 0},
    'r8_3M':      {'J': 3, 'M': 3, 'N': 0},
    'r8_4M':      {'J': 2, 'M': 4, 'N': 0},
    'r8_splitMN': {'J': 4, 'M': 1, 'N': 1},
}

# Stable presentation order for analysis (ascending M evidence, split last).
CONDITION_ORDER = ['r8_0M', 'r8_1M', 'r8_2M', 'r8_3M', 'r8_4M', 'r8_splitMN']


def round8_results_text(counts):
    """Human-readable round-8 result line inserted into the prompt."""
    return (f"{counts['J']} chose J, "
            f"{counts['M']} chose M, and {counts['N']} chose N "
            f"(out of the 6 players).")


################################################################
# === Prompts ===
################################################################

SYSTEM_PROMPT = (
    "You are a careful behavioural forecaster observing a coordination game "
    "played by humans. Reason about how human participants adopt, keep, or "
    "abandon shared conventions, then report your prediction strictly in the "
    "required JSON schema. Do not play the game yourself."
)


def build_user_prompt(counts):
    return f"""You are observing a coordination game played by 6 human participants.

Rules:
- The players cannot communicate.
- They only earn points if all 6 choose the same letter.
- Rounds 1-7: if all 6 choose J, everyone earns 10 points. All other letters earn 0.
- Rounds 8-14: if all 6 choose J, everyone earns 10 points. If all 6 choose M, everyone earns 15 points. If all 6 choose N, everyone earns 15 points. All other outcomes earn 0.

Observed history:
- Rounds 1-7: all 6 human players chose J every round.
- Round 8: {round8_results_text(counts)}

Task:
Predict what the 6 human players will choose in each remaining round, rounds 9-14.

For each round, give probabilities from 0 to 100 for how many of the 6 humans will choose J, M, N, or Other. The probabilities within each round must sum to 100.

Also predict the most likely group outcome for each round, choosing exactly one of:
- all_J
- all_M
- all_N
- mixed

Provide a prediction for every round from 9 through 14 (six rounds in total)."""


################################################################
# === Structured-output schema ===
################################################################
# Strict JSON schema: every field required, additionalProperties disabled.
# (Array length and "sums to 100" are validated post-hoc, not by the schema.)
PREDICTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rounds": {
            "type": "array",
            "description": "One entry per round, for rounds 9 through 14.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "round": {"type": "integer", "description": "Round number (9-14)."},
                    "prob_J": {"type": "integer", "description": "0-100 probability a human chooses J."},
                    "prob_M": {"type": "integer", "description": "0-100 probability a human chooses M."},
                    "prob_N": {"type": "integer", "description": "0-100 probability a human chooses N."},
                    "prob_Other": {"type": "integer", "description": "0-100 probability a human chooses any other letter."},
                    "group_outcome": {
                        "type": "string",
                        "enum": ["all_J", "all_M", "all_N", "mixed"],
                        "description": "Single most likely group outcome for the round.",
                    },
                },
                "required": ["round", "prob_J", "prob_M", "prob_N", "prob_Other", "group_outcome"],
            },
        }
    },
    "required": ["rounds"],
}


################################################################
# === API call ===
################################################################

def get_prediction(counts):
    """Call the model once and return (parsed_prediction_dict, reasoning_text, raw_output_text)."""
    full_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(counts)},
    ]

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                input=full_messages,
                reasoning={"effort": "low", "summary": "auto"},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "name_game_prediction",
                        "schema": PREDICTION_SCHEMA,
                        "strict": True,
                    }
                },
            )

            # Capture the model's reasoning summary (may be empty), matching the
            # player experiment's trace-capture pattern.
            reasoning_text = ""
            for item in response.output:
                if item.type == "reasoning":
                    if item.summary:
                        reasoning_text = " ".join(part.text for part in item.summary).replace("\n", " ")
                    break

            raw = response.output_text
            parsed = json.loads(raw)
            return parsed, reasoning_text, raw

        except Exception as e:  # transient API / parse error -> retry
            last_err = e
            print(f"  [WARN] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(SLEEP_SECONDS)

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed; last error: {last_err}")


################################################################
# === Run experiment ===
################################################################

# Build the full job list (condition x replicate) and shuffle execution order so
# any drift in API behaviour over the run is spread evenly across conditions.
jobs = [(cond, run_idx)
        for cond in CONDITION_ORDER
        for run_idx in range(1, RUNS_PER_CONDITION + 1)]
random.shuffle(jobs)

print(f"=== Name Game PREDICTION experiment ===")
print(f"Model: {MODEL} | conditions: {len(CONDITIONS)} | runs/condition: {RUNS_PER_CONDITION}")
print(f"Total model calls: {len(jobs)}  (~{len(jobs) * SLEEP_SECONDS / 60:.0f} min at {SLEEP_SECONDS}s pacing)\n")

rows = []          # tidy long rows (one per round)
raw_records = []   # one record per run

for job_num, (condition, run_idx) in enumerate(jobs, start=1):
    counts = CONDITIONS[condition]
    print(f"[{job_num}/{len(jobs)}] condition={condition} run={run_idx}")

    parsed, reasoning_text, raw = get_prediction(counts)
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Index returned rounds by round number for safe lookup / validation.
    by_round = {r.get('round'): r for r in parsed.get('rounds', [])}

    for rnd in PREDICTED_ROUNDS:
        r = by_round.get(rnd)
        if r is None:
            print(f"  [WARN] missing round {rnd} in response (condition={condition}, run={run_idx})")
            continue
        prob_sum = r['prob_J'] + r['prob_M'] + r['prob_N'] + r['prob_Other']
        if prob_sum != 100:
            print(f"  [WARN] round {rnd} probabilities sum to {prob_sum} (not 100) — stored raw")
        rows.append({
            'model': MODEL,
            'condition': condition,
            'run': run_idx,
            'round': rnd,
            'prob_J': r['prob_J'],
            'prob_M': r['prob_M'],
            'prob_N': r['prob_N'],
            'prob_Other': r['prob_Other'],
            'prob_sum': prob_sum,
            'group_outcome': r['group_outcome'],
            'reasoning_summary': reasoning_text,
            'run_timestamp': run_ts,
        })

    raw_records.append({
        'model': MODEL,
        'condition': condition,
        'run': run_idx,
        'run_timestamp': run_ts,
        'round8_results': round8_results_text(counts),
        'reasoning_summary': reasoning_text,
        'prediction': parsed,
        'raw_output': raw,
    })

    time.sleep(SLEEP_SECONDS)

################################################################
# === Export ===
################################################################

os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

csv_path = os.path.join(RESULTS_DIR, f'name_game_prediction_MN_{timestamp}.csv')
jsonl_path = os.path.join(RESULTS_DIR, f'name_game_prediction_MN_{timestamp}.jsonl')

fieldnames = [
    'model', 'condition', 'run', 'round',
    'prob_J', 'prob_M', 'prob_N', 'prob_Other', 'prob_sum',
    'group_outcome', 'reasoning_summary', 'run_timestamp',
]
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open(jsonl_path, 'w') as f:
    for rec in raw_records:
        f.write(json.dumps(rec) + "\n")

print(f"\n=== COMPLETE ===")
print(f"Runs collected: {len(raw_records)} | tidy rows: {len(rows)}")
print(f"Tidy CSV : {csv_path}")
print(f"Raw JSONL: {jsonl_path}")
