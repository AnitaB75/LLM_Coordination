"""
Name Game — SELF-PLAY variant (companion to the prediction variant)
===================================================================

Research question
-----------------
When the model is itself a PLAYER (not an observer predicting humans), how
readily does it abandon the established J convention for the higher-value M,
given the same round-8 evidence? Comparing this against the prediction variant
(Name_Game_prediction_MN.py) tests whether the model's OWN willingness to switch
matches, exceeds, or falls below what it PREDICTS humans will do.

Difference from the prediction variant
--------------------------------------
- Prediction variant: model is an observer; outputs P(human chooses J/M/N/Other)
  and the predicted group outcome for rounds 9-14.
- Self-play variant (this file): model is one of the 6 players; outputs the
  letter IT will choose for each of rounds 9-14 to maximise its own points.

Design
------
One-shot plan: the model decides rounds 9-14 from the round-8 evidence, with no
new feedback between rounds (mirrors the prediction task's elicitation, so the
two arms are directly comparable). For the sequential, feedback-driven player
game see ../../Name_game_experiment/Name_Game_LLM_feedback_MN.py.

The model is told it had been coordinating on J (it chose J in round 8), so in
every condition it is one of the J-choosers; the round-8 counts describe all 6
players including itself.

Conditions, runs, model, pacing, and output conventions match the prediction
variant (6 conditions x 20 runs = 120 calls; gpt-5.4; structured JSON; results
written to ../results/ with a `name_game_play_MN_` prefix).

Setup / run
-----------
Requires OPENAI_API_KEY in a .env file at the repo root (git-ignored).
    pip install openai python-dotenv
    python Prediction_game/experiment/Name_Game_play_MN.py
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
load_dotenv()  # loads OPENAI_API_KEY from the git-ignored repo-root .env

client = OpenAI()
MODEL = "gpt-5.4"

RUNS_PER_CONDITION = 3            # quick version (full study used 20)
PLAY_ROUNDS = list(range(9, 15))  # rounds 9-14
SLEEP_SECONDS = 6                 # quick pacing (full study used 21s for ~3 RPM)
MAX_RETRIES = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

################################################################
# === Conditions (round-8 evidence) — identical to prediction variant ===
################################################################
CONDITIONS = {
    'r8_0M':      {'J': 6, 'M': 0, 'N': 0},
    'r8_1M':      {'J': 5, 'M': 1, 'N': 0},
    'r8_2M':      {'J': 4, 'M': 2, 'N': 0},
    'r8_3M':      {'J': 3, 'M': 3, 'N': 0},
    'r8_4M':      {'J': 2, 'M': 4, 'N': 0},
    'r8_splitMN': {'J': 4, 'M': 1, 'N': 1},
}
CONDITION_ORDER = ['r8_0M', 'r8_1M', 'r8_2M', 'r8_3M', 'r8_4M', 'r8_splitMN']


def round8_results_text(counts):
    return (f"{counts['J']} chose J, "
            f"{counts['M']} chose M, and {counts['N']} chose N "
            f"(out of the 6 players).")


################################################################
# === Prompts ===
################################################################

SYSTEM_PROMPT = (
    "You are a participant in a coordination game. Decide what to play to "
    "maximise your own points given the other players' behaviour, then report "
    "your planned choices strictly in the required JSON schema."
)


def build_user_prompt(counts):
    return f"""You are one of 6 players in a coordination game.

Rules:
- The players cannot communicate.
- You only earn points if all 6 players choose the same letter.
- Rounds 1-7: if all 6 choose J, everyone earns 10 points. All other letters earn 0.
- Rounds 8-14: if all 6 choose J, everyone earns 10 points. If all 6 choose M, everyone earns 15 points. If all 6 choose N, everyone earns 15 points. All other outcomes earn 0.

History so far:
- Rounds 1-7: all 6 players (including you) chose J every round.
- Round 8: {round8_results_text(counts)} You chose J.

Task:
Decide what YOU will choose in each remaining round, rounds 9-14, to maximise your own total points.

For each round give your own choice as exactly one of: J, M, N, or Other.
Provide a choice for every round from 9 through 14 (six rounds in total)."""


################################################################
# === Structured-output schema ===
################################################################
PLAY_SCHEMA = {
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
                    "choice": {
                        "type": "string",
                        "enum": ["J", "M", "N", "Other"],
                        "description": "The letter you will play this round.",
                    },
                },
                "required": ["round", "choice"],
            },
        }
    },
    "required": ["rounds"],
}


################################################################
# === API call ===
################################################################

def get_play(counts):
    """Call the model once; return (parsed_plan_dict, reasoning_text, raw_output_text)."""
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
                        "name": "name_game_play",
                        "schema": PLAY_SCHEMA,
                        "strict": True,
                    }
                },
            )

            reasoning_text = ""
            for item in response.output:
                if item.type == "reasoning":
                    if item.summary:
                        reasoning_text = " ".join(part.text for part in item.summary).replace("\n", " ")
                    break

            raw = response.output_text
            parsed = json.loads(raw)
            return parsed, reasoning_text, raw

        except Exception as e:
            last_err = e
            print(f"  [WARN] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(SLEEP_SECONDS)

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed; last error: {last_err}")


################################################################
# === Run experiment ===
################################################################

jobs = [(cond, run_idx)
        for cond in CONDITION_ORDER
        for run_idx in range(1, RUNS_PER_CONDITION + 1)]
random.shuffle(jobs)

print(f"=== Name Game SELF-PLAY experiment ===")
print(f"Model: {MODEL} | conditions: {len(CONDITIONS)} | runs/condition: {RUNS_PER_CONDITION}")
print(f"Total model calls: {len(jobs)}  (~{len(jobs) * SLEEP_SECONDS / 60:.0f} min at {SLEEP_SECONDS}s pacing)\n")

rows = []
raw_records = []

for job_num, (condition, run_idx) in enumerate(jobs, start=1):
    counts = CONDITIONS[condition]
    print(f"[{job_num}/{len(jobs)}] condition={condition} run={run_idx}")

    parsed, reasoning_text, raw = get_play(counts)
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    by_round = {r.get('round'): r for r in parsed.get('rounds', [])}
    for rnd in PLAY_ROUNDS:
        r = by_round.get(rnd)
        if r is None:
            print(f"  [WARN] missing round {rnd} (condition={condition}, run={run_idx})")
            continue
        rows.append({
            'model': MODEL,
            'condition': condition,
            'run': run_idx,
            'round': rnd,
            'choice': r['choice'],
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
        'plan': parsed,
        'raw_output': raw,
    })

    time.sleep(SLEEP_SECONDS)

################################################################
# === Export ===
################################################################

os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

csv_path = os.path.join(RESULTS_DIR, f'name_game_play_MN_{timestamp}.csv')
jsonl_path = os.path.join(RESULTS_DIR, f'name_game_play_MN_{timestamp}.jsonl')

fieldnames = ['model', 'condition', 'run', 'round', 'choice',
              'reasoning_summary', 'run_timestamp']
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
