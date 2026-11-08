"""
PILOT — Name Game round-9 prediction, Claude (claude-opus-4-8), with a REWARD x
EVIDENCE manipulation. Multimodal, matching the HUMAN Qualtrics paradigm.

Design (between-subjects, 3 x 4 = 12 cells):
- REWARD (M & N point value): 15 / 20 / 50  (J stays 10 throughout)
- EVIDENCE (round-8 result):  0M / 1M / 2M / 3M
- Each simulated participant is randomly assigned ONE cell and makes a single
  prediction. Stimuli are the exact images humans saw, regenerated per reward
  level (stimuli/reward_<R>/), so the printed point values are internally
  consistent with the spoken caption.
- N_PER_CELL participants per cell.

Response (structured JSON, mirroring the survey items):
  pred_J / pred_M / pred_N : number of the 6 players (must sum to 6)
  self_choice              : "J"|"M"|"N"|"Another letter"
  reasoning                : free text

Output -> ../results/pilot_round9_claude_reward_<ts>.{csv,jsonl}
"""

import os, csv, json, base64, random, time, re
from datetime import datetime
from dotenv import load_dotenv
import anthropic

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, "..", "..", ".env"))     # repo-root .env (ANTHROPIC_API_KEY)
STIM_BASE = os.path.expanduser("~/LLM_Coordination/qualtrics_human_study/stimuli")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

client = anthropic.Anthropic()
MODEL = "claude-opus-4-8"
N_PER_CELL = 10
SLEEP_SECONDS = 2
MAX_RETRIES = 6

REWARDS = [15, 20, 50]                       # M & N point value (J stays 10)
CONDITION_ORDER = ["0M", "1M", "2M", "3M"]   # round-8 evidence

# round-8 condition -> result-image filename (within a reward folder)
COND_IMAGE = {
    "0M": "05_round8_result_0M.png",
    "1M": "06_round8_result_1M.png",
    "2M": "07_round8_result_2M.png",
    "3M": "08_round8_result_3M.png",
}
# shared image sequence shown before the condition-specific result (survey order)
SHARED_IMAGES = ["00_how_to_play.png", "01_how_points.png",
                 "03_history_rounds_1-7.png", "04_new_letters.png"]

# verbatim survey text (NEWLETTER_CAPTION is parameterized by reward)
INTRO = ("In this study, you'll watch a simple matching game played by six people who are "
         "trying to earn points by all choosing the same letter. If they all choose the same "
         "letter, they get points; if they don't, they get nothing.\n\n"
         "You'll see what everyone chose over the first several rounds, including a point where "
         "new, higher-value letters become available.\n\n"
         "Your task is to predict what the players will do next. There are no right or wrong "
         "answers, we're just interested in your predictions.")
HISTORY_CAPTION = ("The players have played for 7 rounds. In rounds 1 to 7, all 6 players have "
                   "chosen J, which is worth 10 points.")


def newletter_caption(reward):
    return ("In round 8, the game changes. Two letters, M and N, have new point values. "
            f"M and N used to be worth 0 points, but are now both worth {reward} for the "
            "remainder of the game.")


QUESTION = (
    "Above, you can see what the players chose in round 8, after the change in M and N. "
    "Now, your job is to predict what the players choose in the next round, round 9.\n\n"
    "Please predict what number of players choose J, M, and N (remember that there are 6 "
    "players total).\n\n"
    "Also answer: which letter would you pick in round 9, if you were player 1?\n\n"
    "And explain the reasoning behind your prediction.\n\n"
    "Respond with ONLY a JSON object, no other text, in exactly this form:\n"
    '{"pred_J": <int>, "pred_M": <int>, "pred_N": <int>, '
    '"self_choice": "J"|"M"|"N"|"Another letter", "reasoning": "<text>"}\n'
    "pred_J + pred_M + pred_N must equal 6."
)


def img_block(folder, filename):
    with open(os.path.join(folder, filename), "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data}}


def build_content(reward, condition):
    """Mirror the human flow: intro -> stimulus images (+captions) -> question."""
    folder = os.path.join(STIM_BASE, f"reward_{reward}")
    content = [{"type": "text", "text": INTRO}]
    content.append(img_block(folder, SHARED_IMAGES[0]))          # how to play
    content.append(img_block(folder, SHARED_IMAGES[1]))          # how points are scored
    content.append(img_block(folder, SHARED_IMAGES[2]))          # history (rounds 1-7)
    content.append({"type": "text", "text": HISTORY_CAPTION})
    content.append(img_block(folder, SHARED_IMAGES[3]))          # new letters announcement
    # Everything up to here is identical for all conditions of a given reward (the
    # images live in reward_{reward}/ and newletter_caption depends only on reward).
    # Cache it so the ~40 calls sharing this reward's prefix read it at ~0.1x.
    # 1h TTL: assignments are shuffled, so same-reward calls are scattered across the
    # run — the longer TTL keeps each of the 3 reward prefixes warm despite the gaps.
    content.append({"type": "text", "text": newletter_caption(reward),
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}})
    content.append(img_block(folder, COND_IMAGE[condition]))     # round-8 result (condition)
    content.append({"type": "text", "text": QUESTION})
    return content


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else json.loads(text)


def get_prediction(reward, condition):
    messages = [{"role": "user", "content": build_content(reward, condition)}]
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # no `thinking` param -> Opus 4.8 answers directly, like a human;
            # no temperature/top_p (removed on Opus 4.8).
            resp = client.messages.create(model=MODEL, max_tokens=1024, messages=messages)
            text = "".join(b.text for b in resp.content if b.type == "text")
            return parse_json(text), text
        except Exception as e:
            last_err = e
            print(f"  [WARN] attempt {attempt}/{MAX_RETRIES}: {e}", flush=True)
            time.sleep(min(SLEEP_SECONDS * attempt, 30))
    raise RuntimeError(f"all retries failed: {last_err}")


# balanced random assignment: each participant gets ONE (reward, condition) cell
assignments = [(r, c) for r in REWARDS for c in CONDITION_ORDER] * N_PER_CELL
random.shuffle(assignments)

print(f"=== PILOT: round-9 prediction, {MODEL} (reward x evidence, multimodal) ===", flush=True)
print(f"Cells: {len(REWARDS)}x{len(CONDITION_ORDER)}  Participants: {len(assignments)} "
      f"({N_PER_CELL}/cell)\n", flush=True)

rows, raw_records = [], []
for pid, (reward, condition) in enumerate(assignments, start=1):
    print(f"[participant {pid}/{len(assignments)}] reward={reward} evidence={condition}", flush=True)
    try:
        parsed, raw = get_prediction(reward, condition)
    except Exception as e:
        print(f"  [SKIP] participant {pid} failed after retries: {e}", flush=True)
        continue
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    s = parsed["pred_J"] + parsed["pred_M"] + parsed["pred_N"]
    if s != 6:
        print(f"  [WARN] counts sum to {s} (not 6)", flush=True)
    rows.append({
        "model": MODEL, "participant": pid, "reward": reward, "condition": condition, "round": 9,
        "pred_J": parsed["pred_J"], "pred_M": parsed["pred_M"], "pred_N": parsed["pred_N"],
        "pred_sum": s, "self_choice": parsed.get("self_choice", ""),
        "reasoning": str(parsed.get("reasoning", "")).replace("\n", " "),
        "run_timestamp": ts,
    })
    raw_records.append({"participant": pid, "reward": reward, "condition": condition,
                        "prediction": parsed, "raw": raw})
    time.sleep(SLEEP_SECONDS)

os.makedirs(RESULTS_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(RESULTS_DIR, f"pilot_round9_claude_reward_{ts}.csv")
jsonl_path = os.path.join(RESULTS_DIR, f"pilot_round9_claude_reward_{ts}.jsonl")
fields = ["model", "participant", "reward", "condition", "round", "pred_J", "pred_M", "pred_N",
          "pred_sum", "self_choice", "reasoning", "run_timestamp"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
with open(jsonl_path, "w") as f:
    for r in raw_records:
        f.write(json.dumps(r) + "\n")

print(f"\n=== DONE === {len(rows)} predictions", flush=True)
print(f"CSV: {csv_path}", flush=True)
print("\nMean predicted #players by reward x evidence (round 9):", flush=True)
for reward in REWARDS:
    for c in CONDITION_ORDER:
        cr = [r for r in rows if r["reward"] == reward and r["condition"] == c]
        if cr:
            mj = sum(r["pred_J"] for r in cr) / len(cr)
            mm = sum(r["pred_M"] for r in cr) / len(cr)
            mn = sum(r["pred_N"] for r in cr) / len(cr)
            selfM = sum(1 for r in cr if r["self_choice"] == "M") / len(cr)
            print(f"  reward={reward:>2} {c}: J={mj:.1f} M={mm:.1f} N={mn:.1f} "
                  f"| self-M={selfM:.0%} (n={len(cr)})", flush=True)
