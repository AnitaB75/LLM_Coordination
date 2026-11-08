"""
PILOT — Name Game ROUND-8 prediction, gpt-5.4 (human framing).

Variant of the round-9 pilot: instead of showing a round-8 result and asking for
round 9, the model sees ONLY rounds 1-7 (all J), is told M and N have just become
available (worth 15), and must predict ROUND 8 itself — the first round after the
payoff change, with NO switching evidence. There is therefore a single cell
(no round-8 manipulation).

Response: NUMBER of the 6 players choosing J / M / N (must sum to 6) + reasoning.
Output -> ../results/pilot_round8pred_gpt54_<ts>.{csv,jsonl}
"""

from dotenv import load_dotenv
from openai import OpenAI
import csv, json, os, time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, "..", "..", ".env"), override=True)

client = OpenAI()
MODEL = "gpt-5.4"
N_RUNS = 40
SLEEP_SECONDS = 5
MAX_RETRIES = 6
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

SYSTEM_PROMPT = (
    "You are a careful behavioural forecaster observing a coordination game "
    "played by humans. Reason about how human participants adopt, keep, or "
    "abandon shared conventions, then report your prediction strictly in the "
    "required JSON schema. Do not play the game yourself."
)

USER_PROMPT = """You are observing a coordination game played by 6 human participants.

Rules:
- The players cannot communicate.
- They only earn points if all 6 choose the same letter.
- Rounds 1-7: if all 6 choose J, everyone earns 10 points. All other letters earn 0.
- Rounds 8-14: if all 6 choose J, everyone earns 10 points. If all 6 choose M, everyone earns 15 points. If all 6 choose N, everyone earns 15 points. All other outcomes earn 0.

Observed history:
- Rounds 1-7: all 6 human players chose J every round.

Round 8 is the first round after M and N became available (each worth 15 points).

Task:
Predict what the 6 human players will choose in round 8.
Give the NUMBER of players (out of 6) who will choose J, M, and N. The three numbers must sum to 6.
Also briefly explain the reasoning behind your prediction."""

PREDICTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "pred_J": {"type": "integer", "description": "Number of the 6 players predicted to choose J (0-6)."},
        "pred_M": {"type": "integer", "description": "Number of the 6 players predicted to choose M (0-6)."},
        "pred_N": {"type": "integer", "description": "Number of the 6 players predicted to choose N (0-6)."},
        "reasoning": {"type": "string", "description": "Brief explanation of the prediction."},
    },
    "required": ["pred_J", "pred_M", "pred_N", "reasoning"],
}


def get_prediction():
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}]
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL, input=messages,
                reasoning={"effort": "low", "summary": "auto"},
                text={"format": {"type": "json_schema", "name": "round8_prediction",
                                 "schema": PREDICTION_SCHEMA, "strict": True}},
            )
            reasoning_summary = ""
            for item in resp.output:
                if item.type == "reasoning" and item.summary:
                    reasoning_summary = " ".join(p.text for p in item.summary).replace("\n", " ")
                    break
            return json.loads(resp.output_text), reasoning_summary, resp.output_text
        except Exception as e:
            last_err = e
            print(f"  [WARN] attempt {attempt}/{MAX_RETRIES}: {e}", flush=True)
            time.sleep(min(SLEEP_SECONDS * attempt, 30))
    raise RuntimeError(f"all retries failed: {last_err}")


print(f"=== PILOT: ROUND-8 prediction, {MODEL} (human framing) ===", flush=True)
print(f"Runs: {N_RUNS} (single cell, no round-8 evidence)\n", flush=True)

rows, raw_records = [], []
for run in range(1, N_RUNS + 1):
    print(f"[run {run}/{N_RUNS}]", flush=True)
    try:
        parsed, rsum, raw = get_prediction()
    except Exception as e:
        print(f"  [SKIP] run {run} failed: {e}", flush=True)
        continue
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    s = parsed["pred_J"] + parsed["pred_M"] + parsed["pred_N"]
    if s != 6:
        print(f"  [WARN] counts sum to {s} (not 6)", flush=True)
    rows.append({"model": MODEL, "run": run, "round": 8,
                 "pred_J": parsed["pred_J"], "pred_M": parsed["pred_M"], "pred_N": parsed["pred_N"],
                 "pred_sum": s, "reasoning": parsed["reasoning"].replace("\n", " "),
                 "reasoning_summary": rsum, "run_timestamp": ts})
    raw_records.append({"run": run, "prediction": parsed, "reasoning_summary": rsum, "raw": raw})
    time.sleep(SLEEP_SECONDS)

os.makedirs(RESULTS_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(RESULTS_DIR, f"pilot_round8pred_gpt54_{ts}.csv")
jsonl_path = os.path.join(RESULTS_DIR, f"pilot_round8pred_gpt54_{ts}.jsonl")
fields = ["model", "run", "round", "pred_J", "pred_M", "pred_N", "pred_sum",
          "reasoning", "reasoning_summary", "run_timestamp"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
with open(jsonl_path, "w") as f:
    for r in raw_records:
        f.write(json.dumps(r) + "\n")

print(f"\n=== DONE === {len(rows)} predictions", flush=True)
print(f"CSV: {csv_path}", flush=True)
if rows:
    n = len(rows)
    mj = sum(r["pred_J"] for r in rows) / n
    mm = sum(r["pred_M"] for r in rows) / n
    mn = sum(r["pred_N"] for r in rows) / n
    from collections import Counter
    print(f"\nRound-8 prediction (n={n}):")
    print(f"  mean J={mj:.2f} ({mj/6:.0%})  M={mm:.2f} ({mm/6:.0%})  N={mn:.2f} ({mn/6:.0%})")
    print("  pred_M distribution:", dict(sorted(Counter(r['pred_M'] for r in rows).items())))
