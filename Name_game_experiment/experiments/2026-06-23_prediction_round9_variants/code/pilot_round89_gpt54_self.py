"""
PILOT — Name Game ROUND-8 + ROUND-9 prediction, gpt-5.4, SELF framing.

The model sees ONLY rounds 1-7 (all J) plus the rules, is told the 6 players are
separate instances of ITSELF, and that M and N have just become available
(worth 15). It must predict BOTH round 8 (the first round after the change) and
round 9 — a 2-step forecast with no round-8 evidence given. Single cell.

Response: for each of rounds 8 and 9, the NUMBER of the 6 players choosing
J / M / N (each round must sum to 6) + reasoning.
Output -> ../results/pilot_round89_gpt54_self_<ts>.{csv,jsonl}
"""

from dotenv import load_dotenv
from openai import OpenAI
import csv, json, os, time
from datetime import datetime
from collections import Counter

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
    "played by other instances of yourself (the same AI model). Reason about how "
    "copies of yourself adopt, keep, or abandon shared conventions, then report "
    "your prediction strictly in the required JSON schema. Do not play the game yourself."
)

USER_PROMPT = """You are observing a coordination game played by 6 players. All 6 players are separate instances of yourself — the same AI model you are.

Rules:
- The players cannot communicate.
- They only earn points if all 6 choose the same letter.
- Rounds 1-7: if all 6 choose J, everyone earns 10 points. All other letters earn 0.
- Rounds 8-14: if all 6 choose J, everyone earns 10 points. If all 6 choose M, everyone earns 15 points. If all 6 choose N, everyone earns 15 points. All other outcomes earn 0.

Observed history:
- Rounds 1-7: all 6 players (instances of yourself) chose J every round.

Round 8 is the first round after M and N became available (each worth 15 points).

Task:
Predict what the 6 players (instances of yourself) will choose in round 8, and then what they will choose in round 9.
For each round, give the NUMBER of players (out of 6) who will choose J, M, and N. The three numbers must sum to 6 in each round.
Also briefly explain the reasoning behind your prediction."""

PREDICTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "pred8_J": {"type": "integer", "description": "Round 8: number of the 6 players choosing J (0-6)."},
        "pred8_M": {"type": "integer", "description": "Round 8: number choosing M (0-6)."},
        "pred8_N": {"type": "integer", "description": "Round 8: number choosing N (0-6)."},
        "pred9_J": {"type": "integer", "description": "Round 9: number choosing J (0-6)."},
        "pred9_M": {"type": "integer", "description": "Round 9: number choosing M (0-6)."},
        "pred9_N": {"type": "integer", "description": "Round 9: number choosing N (0-6)."},
        "reasoning": {"type": "string", "description": "Brief explanation of the prediction."},
    },
    "required": ["pred8_J", "pred8_M", "pred8_N", "pred9_J", "pred9_M", "pred9_N", "reasoning"],
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
                text={"format": {"type": "json_schema", "name": "round89_prediction",
                                 "schema": PREDICTION_SCHEMA, "strict": True}},
            )
            rsum = ""
            for item in resp.output:
                if item.type == "reasoning" and item.summary:
                    rsum = " ".join(p.text for p in item.summary).replace("\n", " ")
                    break
            return json.loads(resp.output_text), rsum, resp.output_text
        except Exception as e:
            last_err = e
            print(f"  [WARN] attempt {attempt}/{MAX_RETRIES}: {e}", flush=True)
            time.sleep(min(SLEEP_SECONDS * attempt, 30))
    raise RuntimeError(f"all retries failed: {last_err}")


print(f"=== PILOT: ROUND 8 + 9 prediction, {MODEL} (SELF framing) ===", flush=True)
print(f"Runs: {N_RUNS} (single cell, predict both rounds from rounds 1-7)\n", flush=True)

rows, raw_records = [], []
for run in range(1, N_RUNS + 1):
    print(f"[run {run}/{N_RUNS}]", flush=True)
    try:
        parsed, rsum, raw = get_prediction()
    except Exception as e:
        print(f"  [SKIP] run {run} failed: {e}", flush=True)
        continue
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    s8 = parsed["pred8_J"] + parsed["pred8_M"] + parsed["pred8_N"]
    s9 = parsed["pred9_J"] + parsed["pred9_M"] + parsed["pred9_N"]
    if s8 != 6 or s9 != 6:
        print(f"  [WARN] sums r8={s8} r9={s9} (not 6)", flush=True)
    rows.append({"model": MODEL, "run": run,
                 "pred8_J": parsed["pred8_J"], "pred8_M": parsed["pred8_M"], "pred8_N": parsed["pred8_N"],
                 "pred9_J": parsed["pred9_J"], "pred9_M": parsed["pred9_M"], "pred9_N": parsed["pred9_N"],
                 "sum8": s8, "sum9": s9, "reasoning": parsed["reasoning"].replace("\n", " "),
                 "reasoning_summary": rsum, "run_timestamp": ts})
    raw_records.append({"run": run, "prediction": parsed, "reasoning_summary": rsum, "raw": raw})
    time.sleep(SLEEP_SECONDS)

os.makedirs(RESULTS_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(RESULTS_DIR, f"pilot_round89_gpt54_self_{ts}.csv")
jsonl_path = os.path.join(RESULTS_DIR, f"pilot_round89_gpt54_self_{ts}.jsonl")
fields = ["model", "run", "pred8_J", "pred8_M", "pred8_N", "pred9_J", "pred9_M", "pred9_N",
          "sum8", "sum9", "reasoning", "reasoning_summary", "run_timestamp"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
with open(jsonl_path, "w") as f:
    for r in raw_records:
        f.write(json.dumps(r) + "\n")

print(f"\n=== DONE === {len(rows)} predictions", flush=True)
print(f"CSV: {csv_path}", flush=True)
if rows:
    n = len(rows)
    for rnd in ("8", "9"):
        mj = sum(r[f"pred{rnd}_J"] for r in rows) / n
        mm = sum(r[f"pred{rnd}_M"] for r in rows) / n
        mn = sum(r[f"pred{rnd}_N"] for r in rows) / n
        dist = dict(sorted(Counter(r[f"pred{rnd}_M"] for r in rows).items()))
        print(f"\nRound {rnd} (n={n}): mean J={mj:.2f} ({mj/6:.0%})  M={mm:.2f} ({mm/6:.0%})  N={mn:.2f} ({mn/6:.0%})")
        print(f"  pred_M distribution: {dist}")
