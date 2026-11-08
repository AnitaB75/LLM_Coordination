"""
Name Game — PREDICTION-ONLY variant, CLAUDE FABLE 5 rerun
=========================================================

This is a faithful rerun of the gpt-5.4 prediction study
(`Name_Game_prediction_MN.py`) using Anthropic's Claude Fable 5
(`claude-fable-5`). The design — conditions, prompt, JSON schema, 20 runs per
condition, rounds 9-14 — is identical, so the two model runs are directly
comparable. Only the API client and a few Fable-specific parameters differ.

Fable-specific choices (see the Claude API reference):
- MODEL = "claude-fable-5".
- No temperature / top_p (rejected on Fable 5).
- Thinking is always on; we set display "summarized" so we can capture a
  reasoning-summary trace analogous to the gpt-5.4 run's reasoning_summary.
- Reasoning depth is controlled by output_config effort = "low" (matches the
  gpt-5.4 run's "low" reasoning effort).
- Structured output via output_config.format (strict json_schema) — the direct
  analog of OpenAI Structured Outputs; the SAME schema is reused.
- Server-side refusal fallbacks are intentionally NOT enabled: this experiment
  measures Fable's own predictions, so a silent fallback to another model would
  contaminate the dataset. Refusals (very unlikely for this benign game) are
  detected via stop_reason and retried, never silently answered by another model.

Output -> ../results/ as:
  * name_game_prediction_MN_fable_<ts>.csv   (one row per condition x run x round)
  * name_game_prediction_MN_fable_<ts>.jsonl (one line per run: full trace)

Setup: ANTHROPIC_API_KEY in the repo-root .env (git-ignored).
    pip install anthropic python-dotenv
Run:
    python Prediction_game/experiment/Name_Game_prediction_MN_fable.py
    RUNS_PER_CONDITION=1 python .../Name_Game_prediction_MN_fable.py   # smoke test
"""

import os, csv, json, random, time
from datetime import datetime
from dotenv import load_dotenv
import anthropic

################################################################
# === Setup ===
################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, "..", "..", ".env"))   # repo-root .env (ANTHROPIC_API_KEY)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

client = anthropic.Anthropic()
MODEL = "claude-fable-5"

RUNS_PER_CONDITION = int(os.environ.get("RUNS_PER_CONDITION", "20"))
PREDICTED_ROUNDS = list(range(9, 15))   # rounds 9-14
SLEEP_SECONDS = 2                        # light pacing; SDK auto-retries 429/5xx
MAX_RETRIES = 3
MAX_TOKENS = 8000                        # room for summarized thinking + JSON

################################################################
# === Conditions (round-8 evidence) — identical to the gpt-5.4 study ===
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
# === Prompts — verbatim from the gpt-5.4 study ===
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
# === Structured-output schema — identical to the gpt-5.4 study ===
################################################################
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
    """Call Fable once; return (parsed_dict, reasoning_summary, raw_json_text)."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_prompt(counts)}],
                # Thinking is always on for Fable; ask for a readable summary so we
                # can store a reasoning trace comparable to the gpt-5.4 run.
                thinking={"type": "adaptive", "display": "summarized"},
                # Strict JSON schema output + low reasoning effort (matches gpt-5.4 run).
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": PREDICTION_SCHEMA},
                },
            )

            if resp.stop_reason == "refusal":
                raise RuntimeError(f"model refusal: {getattr(resp, 'stop_details', None)}")

            reasoning_text = " ".join(
                b.thinking for b in resp.content if b.type == "thinking" and b.thinking
            ).replace("\n", " ").strip()
            raw = "".join(b.text for b in resp.content if b.type == "text")
            parsed = json.loads(raw)
            return parsed, reasoning_text, raw

        except Exception as e:
            last_err = e
            print(f"  [WARN] attempt {attempt}/{MAX_RETRIES} failed: {e}", flush=True)
            time.sleep(SLEEP_SECONDS * attempt)

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed; last error: {last_err}")


################################################################
# === Run experiment ===
################################################################
jobs = [(cond, run_idx)
        for cond in CONDITION_ORDER
        for run_idx in range(1, RUNS_PER_CONDITION + 1)]
random.shuffle(jobs)

print("=== Name Game PREDICTION experiment — Claude Fable 5 rerun ===", flush=True)
print(f"Model: {MODEL} | conditions: {len(CONDITIONS)} | runs/condition: {RUNS_PER_CONDITION}", flush=True)
print(f"Total model calls: {len(jobs)}\n", flush=True)

rows, raw_records = [], []
for job_num, (condition, run_idx) in enumerate(jobs, start=1):
    counts = CONDITIONS[condition]
    print(f"[{job_num}/{len(jobs)}] condition={condition} run={run_idx}", flush=True)

    parsed, reasoning_text, raw = get_prediction(counts)
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    by_round = {r.get('round'): r for r in parsed.get('rounds', [])}

    for rnd in PREDICTED_ROUNDS:
        r = by_round.get(rnd)
        if r is None:
            print(f"  [WARN] missing round {rnd} (condition={condition}, run={run_idx})", flush=True)
            continue
        prob_sum = r['prob_J'] + r['prob_M'] + r['prob_N'] + r['prob_Other']
        if prob_sum != 100:
            print(f"  [WARN] round {rnd} probs sum to {prob_sum} (not 100) — stored raw", flush=True)
        rows.append({
            'model': MODEL, 'condition': condition, 'run': run_idx, 'round': rnd,
            'prob_J': r['prob_J'], 'prob_M': r['prob_M'], 'prob_N': r['prob_N'],
            'prob_Other': r['prob_Other'], 'prob_sum': prob_sum,
            'group_outcome': r['group_outcome'],
            'reasoning_summary': reasoning_text, 'run_timestamp': run_ts,
        })

    raw_records.append({
        'model': MODEL, 'condition': condition, 'run': run_idx, 'run_timestamp': run_ts,
        'round8_results': round8_results_text(counts),
        'reasoning_summary': reasoning_text, 'prediction': parsed, 'raw_output': raw,
    })

    time.sleep(SLEEP_SECONDS)

################################################################
# === Export ===
################################################################
os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = os.path.join(RESULTS_DIR, f'name_game_prediction_MN_fable_{timestamp}.csv')
jsonl_path = os.path.join(RESULTS_DIR, f'name_game_prediction_MN_fable_{timestamp}.jsonl')

fieldnames = ['model', 'condition', 'run', 'round',
              'prob_J', 'prob_M', 'prob_N', 'prob_Other', 'prob_sum',
              'group_outcome', 'reasoning_summary', 'run_timestamp']
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(rows)
with open(jsonl_path, 'w') as f:
    for rec in raw_records:
        f.write(json.dumps(rec) + "\n")

print(f"\n=== COMPLETE ===", flush=True)
print(f"Runs collected: {len(raw_records)} | tidy rows: {len(rows)}", flush=True)
print(f"Tidy CSV : {csv_path}", flush=True)
print(f"Raw JSONL: {jsonl_path}", flush=True)

# Quick round-9 sanity summary (mean predicted prob_M by condition).
print("\nMean round-9 prob_M by condition:", flush=True)
for c in CONDITION_ORDER:
    r9 = [r for r in rows if r['condition'] == c and r['round'] == 9]
    if r9:
        print(f"  {c}: prob_M={sum(x['prob_M'] for x in r9)/len(r9):.0f} (n={len(r9)})", flush=True)
