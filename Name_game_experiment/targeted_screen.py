"""
STEP 1c - targeted screen, and its independent confirmation run.

Procedure is Step 1's, unchanged. PROMPT, PROMPT_SHA8, MODELS, ask_gpt, ask_claude
and parse_side are IMPORTED from `step1_form_choice.py` rather than restated, so
the wording, reasoning effort, max tokens and parser are byte-identical to Step 1
and Step 1b and the three datasets are comparable. Nothing about the model call is
redefined in this file.

Two stages, deliberately kept apart:

  --stage screen   27 targeted pairs x 2 models x n=30   (winner's-curse prone)
  --stage confirm  the selected battery x 2 models x n=60, FRESH trials

They write to different files, use different counterbalance-allocation seeds, and
are analysed separately. Confirmation data are never pooled with screen data for
the confirmatory estimates - the screen picked the stimuli, so reusing it would
bias them upward.

Run:  python3 targeted_screen.py --stage screen  --dry-run
      python3 targeted_screen.py --stage screen
      python3 targeted_screen.py --stage confirm --battery targeted_screen_selected.csv
"""
import os, csv, sys, json, time, random, base64, threading, argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
load_dotenv()

# Prompt, API settings and parser come from Step 1 - do not redefine them here.
from step1_form_choice import (PROMPT, PROMPT_SHA8, MODELS, GPT_ID, CLAUDE_ID,
                               ask_gpt, ask_claude, parse_side)
import targeted_common as tc

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "step1c_output")
WORKERS = 8

# distinct allocation seeds so the two stages are independent draws.
# These seed the AB/BA shuffle only; neither API exposes a sampling seed.
SEEDS = {"screen": 20260807, "confirm": 20260808}
N_DEFAULT = {"screen": 30, "confirm": 60}

COLS = ["stage", "model", "model_id", "stimulus_pair", "family", "manipulation",
        "source", "run_id", "attempt_number", "seed", "prompt_sha8", "variant",
        "left_identity", "right_identity", "response_raw", "reasoning",
        "response_side", "choice_identity", "valid_response",
        "superseded_by_retry", "api_settings", "timestamp"]


def battery_for(stage, args, reg):
    if stage == "screen":
        ids = [k for k, v in reg.items() if v["source"] == "targeted"]
        return sorted(ids)
    if not args.battery:
        sys.exit("--stage confirm requires --battery <selected.csv>")
    path = args.battery if os.path.isabs(args.battery) else os.path.join(OUTDIR, args.battery)
    if not os.path.exists(path):
        sys.exit(f"battery file not found: {path}")
    ids = [r["stimulus_id"] for r in csv.DictReader(open(path))
           if r.get("selection_status", "").startswith("SELECT")]
    if not ids:
        sys.exit(f"no rows with selection_status starting 'SELECT' in {path}")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("screen", "confirm"), required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--battery", default=None, help="confirm stage: selected CSV")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    n = args.n if args.n is not None else N_DEFAULT[args.stage]
    if n % 2:
        sys.exit("--n must be even so AB/BA allocation is exactly balanced")

    reg = tc.registry()
    ids = battery_for(args.stage, args, reg)
    missing = [i for i in ids if i not in reg]
    if missing:
        sys.exit(f"unknown stimulus_id(s): {missing}")

    # every image must exist before a single call is spent
    images = {}
    for sid in ids:
        e = reg[sid]
        for variant, p in zip(("AB", "BA"), tc.paths(e)):
            if not os.path.exists(p):
                sys.exit(f"missing stimulus file: {p}")
            with open(p, "rb") as fh:
                images[(sid, variant)] = base64.standard_b64encode(fh.read()).decode()

    clients = {}
    if not args.dry_run:
        miss = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(k)]
        if miss:
            sys.exit(f"missing environment variable(s): {', '.join(miss)}")
        from openai import OpenAI
        import anthropic
        clients["gpt"] = OpenAI(max_retries=5)
        clients["claude"] = anthropic.Anthropic(max_retries=5)

    rng = random.Random(SEEDS[args.stage])
    jobs = []
    for model in MODELS:
        for sid in ids:
            variants = ["AB"] * (n // 2) + ["BA"] * (n - n // 2)
            rng.shuffle(variants)
            for i, v in enumerate(variants):
                jobs.append(dict(model=model, sid=sid, variant=v,
                                 run_id=f"{args.stage}__{model}__{sid}__{i:03d}"))

    os.makedirs(OUTDIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(OUTDIR, f"targeted_{args.stage}_calls_{ts}.csv")
    settings_json = {k: json.dumps(v, sort_keys=True) for k, v in MODELS.items()}

    lock = threading.Lock()
    counts = dict(done=0, invalid=0, error=0, lost=0)
    fh = open(raw_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=COLS)
    writer.writeheader()
    fh.flush()

    def one_call(job, attempt):
        if args.dry_run:
            r = random.Random(f"{job['run_id']}|{attempt}")
            if r.random() < 0.03:
                return "Hard to say either way.", "unsure"
            return f"Reasoning here. CHOICE: {r.choice(['LEFT', 'RIGHT'])}", "weighed form"
        try:
            fn = ask_gpt if job["model"] == "gpt" else ask_claude
            return fn(clients[job["model"]], images[(job["sid"], job["variant"])])
        except Exception as exc:                       # noqa: BLE001 - logged, not raised
            with lock:
                counts["error"] += 1
            return f"__ERR__ {type(exc).__name__}: {exc}", ""

    def run_job(job):
        e = reg[job["sid"]]
        left_id, right_id = tc.slot_identities(e, job["variant"])
        # Step 1 retry policy, unchanged: one retry on an unparseable reply, and
        # if the retry also fails the trial is LOST, not resampled to hit N.
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
        if not side:
            with lock:
                counts["lost"] += 1

        out = [dict(
            stage=args.stage, model=job["model"], model_id=MODELS[job["model"]]["model"],
            stimulus_pair=job["sid"], family=e["family"], manipulation=e["manipulation"],
            source=e["source"], run_id=job["run_id"], attempt_number=att, seed="",
            prompt_sha8=PROMPT_SHA8, variant=job["variant"],
            left_identity=left_id, right_identity=right_id,
            response_raw=raw_t.replace("\n", " ").strip(),
            reasoning=rea_t.replace("\n", " ").strip(),
            response_side=s, choice_identity=iv,
            valid_response=valid, superseded_by_retry=sup,
            api_settings=settings_json[job["model"]],
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            for att, raw_t, rea_t, s, iv, valid, sup in rows]

        with lock:
            for row in out:
                writer.writerow(row)
            fh.flush()
            counts["done"] += 1
            if counts["done"] % 100 == 0:
                print(f"  {counts['done']}/{len(jobs)} trials", flush=True)

    print(f"=== STEP 1c · {args.stage.upper()} ===")
    print(f"models  : {GPT_ID}, {CLAUDE_ID}")
    print(f"prompt  : sha8={PROMPT_SHA8}  (imported from step1_form_choice)")
    print(f"stimuli : {len(ids)}  {', '.join(ids)}")
    print(f"trials  : {len(jobs)}  ({n} per model x stimulus, {n // 2} AB / {n // 2} BA)"
          f"{'   [DRY RUN - no API calls]' if args.dry_run else ''}")
    print(f"-> {raw_path}\n", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in as_completed([ex.submit(run_job, j) for j in jobs]):
            pass
    fh.close()

    print(f"\n=== DONE · {counts['done']} trials in {time.time() - t0:.0f}s ===")
    print(f"unparseable first responses (retried): {counts['invalid']}")
    print(f"trials LOST (retry also unparseable) : {counts['lost']}")
    print(f"API errors                           : {counts['error']}")
    print(f"raw CSV                              : {raw_path}")
    print(f"\nnext: python3 targeted_analysis.py --stage {args.stage} "
          f"{os.path.basename(raw_path)}")


if __name__ == "__main__":
    main()
