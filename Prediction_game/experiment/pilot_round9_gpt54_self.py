"""
PILOT — Name Game round-9 prediction, gpt-5.4, SELF partner framing.

Identical to pilot_round9_gpt54.py EXCEPT the partner identity: the model is told
the 6 players are separate instances of ITSELF (the same AI model), rather than
human participants. This is the "self" partner-condition manipulation; everything
else (round 9 only, J/M/N counts summing to 6, conditions, N, reasoning) matches.

Output -> ../results/pilot_round9_gpt54_self_<ts>.{csv,jsonl}
"""

from dotenv import load_dotenv
from openai import OpenAI
import csv
import json
import os
import random
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, "..", "..", ".env"), override=True)   # repo-root .env

client = OpenAI()
MODEL = "gpt-5.4"

N_PER_CONDITION = 10         # balanced across the 4 conditions
PREDICT_ROUND = 9
SLEEP_SECONDS = 5            # light pacing for a small pilot
MAX_RETRIES = 6
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

# round-8 evidence: counts of the 6 players choosing J / M / N
CONDITIONS = {
    "r8_0M": {"J": 6, "M": 0, "N": 0},
    "r8_1M": {"J": 5, "M": 1, "N": 0},
    "r8_2M": {"J": 4, "M": 2, "N": 0},
    "r8_3M": {"J": 3, "M": 3, "N": 0},
}
CONDITION_ORDER = ["r8_0M", "r8_1M", "r8_2M", "r8_3M"]


def round8_text(c):
    return f"{c['J']} chose J, {c['M']} chose M, and {c['N']} chose N (out of the 6 players)."


SYSTEM_PROMPT = (
    "You are a careful behavioural forecaster observing a coordination game "
    "played by other instances of yourself (the same AI model). Reason about how "
    "copies of yourself adopt, keep, or abandon shared conventions, then report "
    "your prediction strictly in the required JSON schema. Do not play the game yourself."
)


def build_user_prompt(c):
    # SELF framing: the 6 players are separate instances of the model itself.
    return f"""You are observing a coordination game played by 6 players. All 6 players are separate instances of yourself — the same AI model you are.

Rules:
- The players cannot communicate.
- They only earn points if all 6 choose the same letter.
- Rounds 1-7: if all 6 choose J, everyone earns 10 points. All other letters earn 0.
- Rounds 8-14: if all 6 choose J, everyone earns 10 points. If all 6 choose M, everyone earns 15 points. If all 6 choose N, everyone earns 15 points. All other outcomes earn 0.

Observed history:
- Rounds 1-7: all 6 players (instances of yourself) chose J every round.
- Round 8: {round8_text(c)}

Task:
Predict what the 6 players (instances of yourself) will choose in round 9.
Give the NUMBER of players (out of 6) who will choose J, M, and N. The three numbers must sum to 6.
Also briefly explain the reasoning behind your prediction."""


PREDICTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pred_J": {"type": "integer", "description": "Number of the 6 players predicted to choose J (0-6)."},
        "pred_M": {"type": "integer", "description": "Number of the 6 players predicted to choose M (0-6)."},
        "pred_N": {"type": "integer", "description": "Number of the 6 players predicted to choose N (0-6)."},
        "reasoning": {"type": "string", "description": "Brief explanation of the prediction."},
    },
    "required": ["pred_J", "pred_M", "pred_N", "reasoning"],
}


def get_prediction(c):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(c)},
    ]
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=messages,
                reasoning={"effort": "low", "summary": "auto"},
                text={"format": {"type": "json_schema", "name": "round9_prediction",
                                 "schema": PREDICTION_SCHEMA, "strict": True}},
            )
            reasoning_summary = ""
            for item in resp.output:
                if item.type == "reasoning" and item.summary:
                    reasoning_summary = " ".join(p.text for p in item.summary).replace("\n", " ")
                    break
            raw = resp.output_text
            return json.loads(raw), reasoning_summary, raw
        except Exception as e:
            last_err = e
            print(f"  [WARN] attempt {attempt}/{MAX_RETRIES}: {e}", flush=True)
            time.sleep(min(SLEEP_SECONDS * attempt, 30))   # exponential-ish backoff
    raise RuntimeError(f"all retries failed: {last_err}")


# --- balanced random assignment: each participant gets ONE condition ----------
assignments = CONDITION_ORDER * N_PER_CONDITION
random.shuffle(assignments)

print(f"=== PILOT: round-9 prediction, {MODEL} (SELF framing, between-subjects) ===", flush=True)
print(f"Participants: {len(assignments)}  ({N_PER_CONDITION} per condition)\n", flush=True)

rows, raw_records = [], []
for pid, condition in enumerate(assignments, start=1):
    c = CONDITIONS[condition]
    print(f"[participant {pid}/{len(assignments)}] assigned={condition}", flush=True)
    try:
        parsed, reasoning_summary, raw = get_prediction(c)
    except Exception as e:
        print(f"  [SKIP] participant {pid} failed after retries: {e}", flush=True)
        continue
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    s = parsed["pred_J"] + parsed["pred_M"] + parsed["pred_N"]
    if s != 6:
        print(f"  [WARN] counts sum to {s} (not 6) — stored raw", flush=True)
    rows.append({
        "model": MODEL, "participant": pid, "condition": condition, "round": PREDICT_ROUND,
        "pred_J": parsed["pred_J"], "pred_M": parsed["pred_M"], "pred_N": parsed["pred_N"],
        "pred_sum": s, "reasoning": parsed["reasoning"].replace("\n", " "),
        "reasoning_summary": reasoning_summary, "run_timestamp": ts,
    })
    raw_records.append({"participant": pid, "condition": condition, "round8": round8_text(c),
                        "prediction": parsed, "reasoning_summary": reasoning_summary, "raw": raw})
    time.sleep(SLEEP_SECONDS)

os.makedirs(RESULTS_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(RESULTS_DIR, f"pilot_round9_gpt54_self_{ts}.csv")
jsonl_path = os.path.join(RESULTS_DIR, f"pilot_round9_gpt54_self_{ts}.jsonl")
fields = ["model", "participant", "condition", "round", "pred_J", "pred_M", "pred_N",
          "pred_sum", "reasoning", "reasoning_summary", "run_timestamp"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
with open(jsonl_path, "w") as f:
    for r in raw_records:
        f.write(json.dumps(r) + "\n")

print(f"\n=== DONE === {len(rows)} predictions", flush=True)
print(f"CSV: {csv_path}", flush=True)

# quick summary: mean predicted counts by condition
print("\nMean predicted #players by condition (round 9):", flush=True)
for cond in CONDITION_ORDER:
    cr = [r for r in rows if r["condition"] == cond]
    if cr:
        mj = sum(r["pred_J"] for r in cr) / len(cr)
        mm = sum(r["pred_M"] for r in cr) / len(cr)
        mn = sum(r["pred_N"] for r in cr) / len(cr)
        print(f"  {cond}: J={mj:.1f}  M={mm:.1f}  N={mn:.1f}  (n={len(cr)})", flush=True)
