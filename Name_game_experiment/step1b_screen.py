"""
STEP 1b - screen the 30-pair battery for model-differentiating stimuli.

Same procedure as Step 1, driven by `stimuli/screen/manifest.csv` instead of a
hardcoded set. The prompt, API settings, parsing, and AB/BA counterbalance are
imported directly from `step1_form_choice.py`, so they are byte-identical and the
screen is comparable to the Step 1 results.

Selection criteria (applied in `step1b_screen_analysis.py`):
  1. modest position bias   - P(LEFT) in [0.35, 0.65] for BOTH models
  2. model dissociation     - |match_gpt - match_claude| > 0.15
  3. form-based choosing    - conditional-position diagnostic: the modal identity
                              is chosen at a similar rate under AB and BA
  4. crossovers             - pairs where GPT > Claude AND pairs where Claude > GPT

Run:  python3 step1b_screen.py --n 30        # 30 pairs x 2 models x 30 = 1800 calls
      python3 step1b_screen.py --n 4         # live smoke test
      python3 step1b_screen.py --dry-run
"""
import os, csv, sys, json, time, random, base64, threading, argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

# Prompt, settings and parser come from Step 1 — do not redefine them here.
from step1_form_choice import (PROMPT, PROMPT_SHA8, MODELS, GPT_ID, CLAUDE_ID,
                               ask_gpt, ask_claude, parse_side)

HERE = os.path.dirname(os.path.abspath(__file__))
STIMDIR = os.path.join(HERE, "stimuli", "screen")
OUTDIR = os.path.join(HERE, "step1_output")
N_DEFAULT = 30
WORKERS = 8
ASSIGN_SEED = 20260804


def load_manifest():
    with open(os.path.join(STIMDIR, "manifest.csv")) as fh:
        return list(csv.DictReader(fh))


def load_images(manifest):
    img = {}
    for m in manifest:
        for variant in ("AB", "BA"):
            path = os.path.join(STIMDIR, f"{m['pair']}_{variant}.png")
            if not os.path.exists(path):
                sys.exit(f"missing stimulus: {path}")
            with open(path, "rb") as fh:
                img[(m["pair"], variant)] = base64.standard_b64encode(fh.read()).decode()
    return img


COLS = ["model", "model_id", "stimulus_pair", "family", "cue_1", "cue_2",
        "run_id", "attempt_number", "seed", "prompt_sha8", "variant",
        "left_identity", "right_identity", "response_raw", "reasoning",
        "response_side", "choice_identity", "valid_response",
        "superseded_by_retry", "api_settings", "timestamp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    if args.n % 2:
        sys.exit("--n must be even")

    manifest = load_manifest()
    meta = {m["pair"]: m for m in manifest}

    clients = {}
    if not args.dry_run:
        missing = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(k)]
        if missing:
            sys.exit(f"missing env var(s): {', '.join(missing)}")
        from openai import OpenAI
        import anthropic
        clients["gpt"] = OpenAI(max_retries=5)
        clients["claude"] = anthropic.Anthropic(max_retries=5)

    images = load_images(manifest)

    rng = random.Random(ASSIGN_SEED)
    jobs = []
    for model in MODELS:
        for m in manifest:
            variants = ["AB"] * (args.n // 2) + ["BA"] * (args.n - args.n // 2)
            rng.shuffle(variants)
            for i, v in enumerate(variants):
                jobs.append(dict(model=model, pair=m["pair"], rep=i, variant=v,
                                 run_id=f"{model}__{m['pair']}__{i:03d}"))

    os.makedirs(OUTDIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(OUTDIR, f"step1b_screen_calls_{ts}.csv")
    settings_json = {k: json.dumps(v, sort_keys=True) for k, v in MODELS.items()}

    lock = threading.Lock()
    counts = dict(done=0, invalid=0, error=0)
    fh = open(raw_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=COLS)
    writer.writeheader(); fh.flush()

    def one_call(job, attempt):
        if args.dry_run:
            r = random.Random(f"{job['run_id']}|{attempt}")
            if r.random() < 0.03:
                return "Hard to say.", "unsure"
            return f"Reasoning here. CHOICE: {r.choice(['LEFT', 'RIGHT'])}", "considered form"
        try:
            fn = ask_gpt if job["model"] == "gpt" else ask_claude
            return fn(clients[job["model"]], images[(job["pair"], job["variant"])])
        except Exception as exc:                      # noqa: BLE001
            with lock:
                counts["error"] += 1
            return f"__ERR__ {type(exc).__name__}: {exc}", ""

    def run_job(job):
        mm = meta[job["pair"]]
        i1, i2 = mm["identity_1"], mm["identity_2"]
        left_id, right_id = (i1, i2) if job["variant"] == "AB" else (i2, i1)
        rows = []
        raw, reason = one_call(job, 1)
        side = parse_side(raw)
        if not side:
            rows.append((1, raw, reason, "", "", False, True))
            with lock:
                counts["invalid"] += 1
            raw, reason = one_call(job, 2)
            side = parse_side(raw)
        ident = {"LEFT": left_id, "RIGHT": right_id}.get(side, "")
        rows.append((len(rows) + 1, raw, reason, side, ident, bool(side), False))

        out = []
        for attempt, raw_txt, reason_txt, s, ident_v, valid, superseded in rows:
            out.append(dict(
                model=job["model"], model_id=MODELS[job["model"]]["model"],
                stimulus_pair=job["pair"], family=mm["family"],
                cue_1=mm["cue_1"], cue_2=mm["cue_2"],
                run_id=job["run_id"], attempt_number=attempt, seed="",
                prompt_sha8=PROMPT_SHA8, variant=job["variant"],
                left_identity=left_id, right_identity=right_id,
                response_raw=raw_txt.replace("\n", " ").strip(),
                reasoning=reason_txt.replace("\n", " ").strip(),
                response_side=s, choice_identity=ident_v,
                valid_response=valid, superseded_by_retry=superseded,
                api_settings=settings_json[job["model"]],
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds")))
        with lock:
            for row in out:
                writer.writerow(row)
            fh.flush()
            counts["done"] += 1
            if counts["done"] % 100 == 0:
                print(f"  {counts['done']}/{len(jobs)} trials", flush=True)

    print("=== STEP 1b · screening battery ===")
    print(f"models  : {GPT_ID}, {CLAUDE_ID}")
    print(f"pairs   : {len(manifest)}  "
          f"({sum(m['family'] == 'single' for m in manifest)} single-cue, "
          f"{sum(m['family'] == 'compete' for m in manifest)} competing-cue)")
    print(f"prompt  : sha8={PROMPT_SHA8}  (imported from step1_form_choice)")
    print(f"trials  : {len(jobs)}  ({args.n} per model x pair)"
          f"{'   [DRY RUN]' if args.dry_run else ''}")
    print(f"-> {raw_path}\n", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in as_completed([ex.submit(run_job, j) for j in jobs]):
            pass
    fh.close()
    print(f"\n=== DONE · {counts['done']} trials in {time.time() - t0:.0f}s ===")
    print(f"unparseable first responses: {counts['invalid']}")
    print(f"API errors                 : {counts['error']}")
    print(f"raw CSV                    : {raw_path}")
    print(f"\nnext: python3 step1b_screen_analysis.py {os.path.basename(raw_path)}")


if __name__ == "__main__":
    main()
