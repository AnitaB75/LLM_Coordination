"""
STEP 1 - choice distribution and within-model coordination rate (form-choice design).

Per model x binary stimulus pair, collect N independent one-shot responses (fresh
API call, no conversation history) and record which underlying stimulus was chosen.
`step1_form_analysis.py` turns these into choice proportions and an unbiased
within-model dyadic match probability with bootstrap CIs.

DESIGN

* No labels are shown to the model. It sees one composite image (two shapes side
  by side) and answers with the POSITION it chose - `CHOICE: LEFT` or
  `CHOICE: RIGHT`. The A/B identity coding exists only in this file and the
  analysis; it never reaches the model.

* The prompt is byte-identical to the arm-1 prompt in the earlier
  `confidence_calibration_probe.py`, so this data pools with that series rather
  than starting a fresh, incomparable one. It states that left/right position is
  randomized independently per player and therefore cannot be used to coordinate.

* One sentence of reasoning is required before the choice. An earlier version of
  this probe asked for a bare `A`/`B` token with no preamble; Claude then answered
  "A" 60/60 on every matched stimulus - a label default, not a form preference.
  The preamble is load-bearing, not decoration. Full traces are logged.

* Counterbalancing: variant AB or BA, exactly N/2 each, shuffled with a fixed
  seed. AB puts identity 1 on the left, BA puts identity 2 there, so position and
  identity are independent by construction.

* Position diagnostic: because the raw LEFT/RIGHT answer is recorded alongside the
  recoded identity, the analysis can report both marginals. P(LEFT) near 1.0 with
  identity match near 0.5 is a position default, not a form preference. The
  earlier probes in this project never tabulated this, so the distinction was
  invisible in their results.

Keys come from the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY) via .env in the
repo root. Nothing existing is modified; output goes to `step1_output/`.

Run:  python3 step1_form_choice.py             # 60 per model x pair (600 calls)
      python3 step1_form_choice.py --n 4       # small live smoke test
      python3 step1_form_choice.py --dry-run   # no API calls, fake replies
"""
import os, csv, re, sys, json, time, random, base64, hashlib, threading, argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "step1_output")

# ----------------------------------------------------------------- config ----
GPT_ID = "gpt-5.4"
CLAUDE_ID = "claude-opus-4-8"

GPT_SETTINGS = {
    "model": GPT_ID,
    "reasoning": {"effort": "low", "summary": "auto"},
    "temperature": None,          # not accepted alongside reasoning
    "max_output_tokens": None,
    "seed": None,                 # Responses API exposes no seed parameter
}
CLAUDE_SETTINGS = {
    "model": CLAUDE_ID,
    "max_tokens": 2000,
    "thinking": {"type": "adaptive", "display": "summarized"},
    "output_config": {"effort": "low"},
    "temperature": None,
    "seed": None,                 # Messages API exposes no seed parameter
}
MODELS = {"gpt": GPT_SETTINGS, "claude": CLAUDE_SETTINGS}

# pair -> (stimuli subdir, identity in the LEFT slot of _AB, identity in the RIGHT slot)
# nd2 dropped: proto is retained as the single focal ceiling control.
SETS = {
    "proto":   ("visual",  "circle",  "square"),
    "inkblot": ("visual2", "ib-A",    "ib-B"),
    "rot":     ("visual",  "blob",    "blob-rot90"),
    "noise":   ("visual2", "noise-A", "noise-B"),
    "mirror":  ("visual",  "blob",    "blob-mirror"),
}

N_DEFAULT = 60
WORKERS = 8
ASSIGN_SEED = 20260804        # counterbalance allocation only; not an API seed

# ----------------------------------------------------------------- prompt ----
# Verbatim from confidence_calibration_probe.py (BASE + ARM1). Sent as a single
# user text block alongside the image - no system prompt - matching that probe.
BASE = ("You are playing a one-shot coordination game with ONE other player - another instance of you "
        "(the same AI model). You will each independently pick one of the two shapes shown; you want to "
        "pick the SAME shape as the other player. No communication. IMPORTANT: the left/right position of "
        "the two shapes is randomized independently for each of you, so position CANNOT be used to "
        "coordinate - you must coordinate on a shape by its FORM.")
PROMPT = BASE + (" Give one short sentence of reasoning, then end with exactly 'CHOICE: LEFT' or "
                 "'CHOICE: RIGHT' (the position in YOUR image of the shape you chose).")
PROMPT_SHA8 = hashlib.sha256(PROMPT.encode()).hexdigest()[:8]

# ------------------------------------------------------------------ images ----
def load_images():
    img = {}
    for pair, (sub, _a, _b) in SETS.items():
        for variant in ("AB", "BA"):
            path = os.path.join(HERE, "stimuli", sub, f"{pair}_{variant}.png")
            if not os.path.exists(path):
                sys.exit(f"missing stimulus file: {path}")
            with open(path, "rb") as fh:
                img[(pair, variant)] = base64.standard_b64encode(fh.read()).decode()
    return img

# ------------------------------------------------------------------- calls ----
def ask_gpt(client, img_b64):
    """Returns (visible_text, reasoning_summary)."""
    r = client.responses.create(
        model=GPT_ID,
        reasoning={"effort": "low", "summary": "auto"},
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": PROMPT},
            {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"}]}])
    reasoning = ""
    for item in r.output:
        if item.type == "reasoning" and item.summary:
            reasoning = " ".join(p.text for p in item.summary)
            break
    return (r.output_text or ""), reasoning


def ask_claude(client, img_b64):
    """Returns (visible_text, thinking_summary)."""
    r = client.messages.create(
        model=CLAUDE_ID,
        max_tokens=2000,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": img_b64}},
            {"type": "text", "text": PROMPT}]}])
    reasoning, answer = "", ""
    for b in r.content:
        if b.type == "thinking":
            reasoning = b.thinking or ""
        elif b.type == "text":
            answer += b.text
    return answer, reasoning


def parse_side(txt):
    """Same parser as the earlier probes: explicit marker, then a lenient fallback."""
    if not txt:
        return ""
    m = re.search(r"CHOICE:\s*(LEFT|RIGHT)", txt.upper())
    if m:
        return m.group(1)
    up = txt.upper()
    if "LEFT" in up and "RIGHT" not in up:
        return "LEFT"
    if "RIGHT" in up and "LEFT" not in up:
        return "RIGHT"
    return ""


def slot_identities(pair, variant):
    """(identity shown on the LEFT, identity shown on the RIGHT) for this variant."""
    _sub, ident_1, ident_2 = SETS[pair]
    return (ident_1, ident_2) if variant == "AB" else (ident_2, ident_1)

# -------------------------------------------------------------------- jobs ----
def build_jobs(n_per_cell):
    rng = random.Random(ASSIGN_SEED)
    jobs = []
    for model in MODELS:
        for pair in SETS:
            variants = ["AB"] * (n_per_cell // 2) + ["BA"] * (n_per_cell - n_per_cell // 2)
            rng.shuffle(variants)
            for i, variant in enumerate(variants):
                jobs.append(dict(model=model, pair=pair, rep=i, variant=variant,
                                 run_id=f"{model}__{pair}__{i:03d}"))
    return jobs


COLS = ["model", "model_id", "stimulus_pair", "run_id", "attempt", "seed", "prompt_sha8",
        "variant", "left_identity", "right_identity",
        "response_raw", "reasoning", "response_side", "choice_identity",
        "valid_response", "superseded_by_retry", "api_settings", "timestamp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DEFAULT,
                    help="responses per model x stimulus pair (default 60)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls; deterministic fake replies to test the pipeline")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    if args.n % 2:
        sys.exit("--n must be even so AB/BA allocation is exactly balanced")

    clients = {}
    if not args.dry_run:
        missing = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(k)]
        if missing:
            sys.exit(f"missing environment variable(s): {', '.join(missing)}")
        from openai import OpenAI
        import anthropic
        clients["gpt"] = OpenAI(max_retries=5)
        clients["claude"] = anthropic.Anthropic(max_retries=5)

    images = load_images()
    jobs = build_jobs(args.n)
    os.makedirs(OUTDIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(OUTDIR, f"step1_form_calls_{ts}.csv")

    settings_json = {m: json.dumps(s, sort_keys=True) for m, s in MODELS.items()}
    lock = threading.Lock()
    counts = dict(done=0, invalid=0, error=0)
    fh = open(raw_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=COLS)
    writer.writeheader()
    fh.flush()

    def one_call(job, attempt):
        if args.dry_run:
            rng = random.Random(f"{job['run_id']}|{attempt}")
            if rng.random() < 0.04:
                return "Both look similar to me.", "considered both"   # exercise retry
            side = rng.choice(["LEFT", "RIGHT"])
            return f"The forms differ subtly, so I go with one. CHOICE: {side}", "weighed form cues"
        try:
            fn = ask_gpt if job["model"] == "gpt" else ask_claude
            return fn(clients[job["model"]], images[(job["pair"], job["variant"])])
        except Exception as exc:                          # noqa: BLE001 - logged, not raised
            with lock:
                counts["error"] += 1
            return f"__ERR__ {type(exc).__name__}: {exc}", ""

    def run_job(job):
        left_id, right_id = slot_identities(job["pair"], job["variant"])
        rows = []
        raw, reasoning = one_call(job, 1)
        side = parse_side(raw)
        if not side:
            rows.append((1, raw, reasoning, "", "", False, True))
            with lock:
                counts["invalid"] += 1
            raw, reasoning = one_call(job, 2)
            side = parse_side(raw)
        identity = {"LEFT": left_id, "RIGHT": right_id}.get(side, "")
        rows.append((len(rows) + 1, raw, reasoning, side, identity, bool(side), False))

        out = []
        for attempt, raw_txt, reason_txt, resp_side, ident, valid, superseded in rows:
            out.append(dict(
                model=job["model"], model_id=MODELS[job["model"]]["model"],
                stimulus_pair=job["pair"], run_id=job["run_id"], attempt=attempt,
                seed="", prompt_sha8=PROMPT_SHA8, variant=job["variant"],
                left_identity=left_id, right_identity=right_id,
                # full traces retained, newlines flattened so the CSV stays one row per call
                response_raw=raw_txt.replace("\n", " ").strip(),
                reasoning=reason_txt.replace("\n", " ").strip(),
                response_side=resp_side, choice_identity=ident,
                valid_response=valid, superseded_by_retry=superseded,
                api_settings=settings_json[job["model"]],
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds")))
        with lock:
            for row in out:
                writer.writerow(row)
            fh.flush()
            counts["done"] += 1
            if counts["done"] % 50 == 0:
                print(f"  {counts['done']}/{len(jobs)} trials", flush=True)

    print("=== STEP 1 · form choice (no labels shown) ===")
    print(f"models   : {GPT_ID}, {CLAUDE_ID}")
    print(f"stimuli  : {', '.join(SETS)}")
    print(f"prompt   : sha8={PROMPT_SHA8}")
    print(f"trials   : {len(jobs)}  ({args.n} per model x pair)"
          f"{'   [DRY RUN - no API calls]' if args.dry_run else ''}")
    print(f"-> {raw_path}\n", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in as_completed([ex.submit(run_job, j) for j in jobs]):
            pass
    fh.close()

    print(f"\n=== DONE · {counts['done']} trials in {time.time() - t0:.0f}s ===")
    print(f"unparseable first responses (retried): {counts['invalid']}")
    print(f"API errors                           : {counts['error']}")
    print(f"raw call-level CSV                   : {raw_path}")
    print(f"\nnext: python3 step1_form_analysis.py {os.path.basename(raw_path)}")


if __name__ == "__main__":
    main()
