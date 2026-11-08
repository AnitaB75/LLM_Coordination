"""
PILOT — Name Game round-9 prediction, Claude (claude-opus-4-8), matching the
HUMAN Qualtrics paradigm as closely as possible.

Design parity with the human study:
- BETWEEN-SUBJECTS: each simulated participant is randomly assigned ONE round-8
  condition (0M/1M/2M/3M) and makes a single prediction.
- SAME VISUALS: the model is shown the exact stimulus images the humans saw, in
  the same order (how-to-play -> how-points -> history -> new-letters ->
  round-8 result), as multimodal image blocks.
- SAME TEXT: the intro, image captions, and the prediction / self-choice /
  reasoning questions are quoted verbatim from the survey.
- Humans answer directly (no scratchpad), so the model runs WITHOUT extended
  thinking; the written 'reasoning' field is the analog of the survey's
  free-text reasoning box.

Response (structured JSON, mirroring the survey items):
  pred_J / pred_M / pred_N : number of the 6 players (must sum to 6)   [Q2]
  self_choice              : "J"|"M"|"N"|"Another letter"             [Q24]
  reasoning                : free text                                 [Q21]

Output -> ../results/pilot_round9_claude_<ts>.{csv,jsonl}
"""

import os, csv, json, base64, random, time, re
from datetime import datetime
from dotenv import load_dotenv
import anthropic

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, "..", "..", ".env"))     # repo-root .env (ANTHROPIC_API_KEY)
STIM = os.path.expanduser("~/LLM_Coordination/qualtrics_human_study/stimuli")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

client = anthropic.Anthropic()
MODEL = "claude-opus-4-8"
N_PER_CONDITION = 10
SLEEP_SECONDS = 2
MAX_RETRIES = 3

# round-8 condition -> result-image filename (the only thing that differs)
COND_IMAGE = {
    "0M": "05_round8_result_0M.png",
    "1M": "06_round8_result_1M.png",
    "2M": "07_round8_result_2M.png",
    "3M": "08_round8_result_3M.png",
}
CONDITION_ORDER = ["0M", "1M", "2M", "3M"]

# shared image sequence shown before the condition-specific result (survey order)
SHARED_IMAGES = ["00_how_to_play.png", "01_how_points.png",
                 "03_history_rounds_1-7.png", "04_new_letters.png"]

# verbatim survey text
INTRO = ("In this study, you'll watch a simple matching game played by six people who are "
         "trying to earn points by all choosing the same letter. If they all choose the same "
         "letter, they get points; if they don't, they get nothing.\n\n"
         "You'll see what everyone chose over the first several rounds, including a point where "
         "new, higher-value letters become available.\n\n"
         "Your task is to predict what the players will do next. There are no right or wrong "
         "answers, we're just interested in your predictions.")
HISTORY_CAPTION = ("The players have played for 7 rounds. In rounds 1 to 7, all 6 players have "
                   "chosen J, which is worth 10 points.")
NEWLETTER_CAPTION = ("In round 8, the game changes. Two letters, M and N, have new point values. "
                     "M and N used to be worth 0 points, but are now both worth 15 for the "
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


def img_block(filename):
    with open(os.path.join(STIM, filename), "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data}}


def build_content(condition):
    """Mirror the human flow: intro -> stimulus images (+captions) -> question."""
    content = [{"type": "text", "text": INTRO}]
    content.append(img_block(SHARED_IMAGES[0]))          # how to play
    content.append(img_block(SHARED_IMAGES[1]))          # how points are scored
    content.append(img_block(SHARED_IMAGES[2]))          # history (rounds 1-7)
    content.append({"type": "text", "text": HISTORY_CAPTION})
    content.append(img_block(SHARED_IMAGES[3]))          # new letters announcement
    # Everything up to here (INTRO + 4 shared images + 2 captions) is byte-identical
    # for every participant, so cache it: the first call writes it, the rest read it
    # at ~0.1x. The condition-specific image + question below vary and stay uncached.
    content.append({"type": "text", "text": NEWLETTER_CAPTION,
                    "cache_control": {"type": "ephemeral"}})
    content.append(img_block(COND_IMAGE[condition]))     # round-8 result (condition)
    content.append({"type": "text", "text": QUESTION})
    return content


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else json.loads(text)


def get_prediction(condition):
    messages = [{"role": "user", "content": build_content(condition)}]
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # no `thinking` param -> Opus 4.8 answers directly, like a human;
            # no temperature/top_p (removed on Opus 4.8).
            resp = client.messages.create(model=MODEL, max_tokens=1024, messages=messages)
            text = "".join(b.text for b in resp.content if b.type == "text")
            parsed = parse_json(text)
            return parsed, text
        except Exception as e:
            last_err = e
            print(f"  [WARN] attempt {attempt}/{MAX_RETRIES}: {e}", flush=True)
            time.sleep(SLEEP_SECONDS * attempt)
    raise RuntimeError(f"all retries failed: {last_err}")


# balanced random assignment: each participant gets ONE condition
assignments = CONDITION_ORDER * N_PER_CONDITION
random.shuffle(assignments)

print(f"=== PILOT: round-9 prediction, {MODEL} (between-subjects, multimodal) ===", flush=True)
print(f"Participants: {len(assignments)}  ({N_PER_CONDITION} per condition)\n", flush=True)

rows, raw_records = [], []
for pid, condition in enumerate(assignments, start=1):
    print(f"[participant {pid}/{len(assignments)}] assigned={condition}", flush=True)
    parsed, raw = get_prediction(condition)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    s = parsed["pred_J"] + parsed["pred_M"] + parsed["pred_N"]
    if s != 6:
        print(f"  [WARN] counts sum to {s} (not 6)", flush=True)
    rows.append({
        "model": MODEL, "participant": pid, "condition": condition, "round": 9,
        "pred_J": parsed["pred_J"], "pred_M": parsed["pred_M"], "pred_N": parsed["pred_N"],
        "pred_sum": s, "self_choice": parsed.get("self_choice", ""),
        "reasoning": str(parsed.get("reasoning", "")).replace("\n", " "),
        "run_timestamp": ts,
    })
    raw_records.append({"participant": pid, "condition": condition,
                        "prediction": parsed, "raw": raw})
    time.sleep(SLEEP_SECONDS)

os.makedirs(RESULTS_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(RESULTS_DIR, f"pilot_round9_claude_{ts}.csv")
jsonl_path = os.path.join(RESULTS_DIR, f"pilot_round9_claude_{ts}.jsonl")
fields = ["model", "participant", "condition", "round", "pred_J", "pred_M", "pred_N",
          "pred_sum", "self_choice", "reasoning", "run_timestamp"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
with open(jsonl_path, "w") as f:
    for r in raw_records:
        f.write(json.dumps(r) + "\n")

print(f"\n=== DONE === {len(rows)} predictions", flush=True)
print(f"CSV: {csv_path}", flush=True)
print("\nMean predicted #players by condition (round 9):", flush=True)
for c in CONDITION_ORDER:
    cr = [r for r in rows if r["condition"] == c]
    if cr:
        mj = sum(r["pred_J"] for r in cr) / len(cr)
        mm = sum(r["pred_M"] for r in cr) / len(cr)
        mn = sum(r["pred_N"] for r in cr) / len(cr)
        selfM = sum(1 for r in cr if r["self_choice"] == "M") / len(cr)
        print(f"  {c}: J={mj:.1f} M={mm:.1f} N={mn:.1f} | self-M={selfM:.0%} (n={len(cr)})", flush=True)
