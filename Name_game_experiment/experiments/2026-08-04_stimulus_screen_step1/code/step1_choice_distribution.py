"""
STEP 1 - choice distribution and within-model coordination rate, per stimulus pair.

For each model x binary stimulus pair, collect N independent one-shot responses
(fresh API call, no conversation history) and record which underlying stimulus was
chosen. Downstream (`step1_analysis.py`) turns this into choice proportions and an
unbiased within-model dyadic match probability with bootstrap CIs.

DESIGN NOTES (see also the "assumptions" block printed at the end of a run)

* Presentation. The six stimulus pairs exist in this repo ONLY as composite images:
  one 1040x520 PNG showing both options side by side, in an `_AB` and a `_BA`
  variant. There is no generator for the `visual2` (inkblot, noise) or `mountains`
  (nd2) sets, and the composites cannot be split faithfully (the two matplotlib
  panels sit at different x-offsets, and nd2 uses a different layout). So the model
  is shown ONE composite and told the LEFT stimulus is A and the RIGHT is B. This
  also keeps the data comparable with every earlier probe in this project, which all
  use the same composites.

* Counterbalancing. Which underlying stimulus carries the label A is counterbalanced
  by the existing variant files: in `_AB` the left/A slot holds identity A, in `_BA`
  it holds identity B. Allocation is exactly N/2 each, shuffled with a fixed seed.

* Position/label confound. An earlier wording of the system prompt said only that the
  two instances "see the same pair of stimuli". Under it, a 48-call pilot showed both
  models coordinating on the LABEL ("we both say A") rather than on the stimulus: 7 of
  12 cells were unanimous on position while identity sat at chance, including `proto`,
  a near-perfect focal point in this project's earlier data. The current SYSTEM prompt
  states that position and label are randomized per player. `step1_analysis.py` still
  reports the match probability on BOTH codings so any residual label-following is
  visible rather than silently deflating the identity-level estimate.
  Pilot data under the old wording: `step1_output/pilot_promptv0_calls_*.csv`.

Keys come from the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY) via .env in the
repo root. Nothing existing is modified; all output goes to `step1_output/`.

Run:  python3 step1_choice_distribution.py            # 60 per model x pair (720 calls)
      python3 step1_choice_distribution.py --n 4      # small live smoke test
      python3 step1_choice_distribution.py --dry-run  # no API calls, fake responses
"""
import os, csv, re, sys, json, time, random, base64, hashlib, threading, argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "step1_output")

# ----------------------------------------------------------------- config ----
# Model ids and per-model settings are the ones already used by this project's
# probes (confidence_calibration_probe.py). Held fixed across every call.
GPT_ID = "gpt-5.4"
CLAUDE_ID = "claude-opus-4-8"

GPT_SETTINGS = {
    "model": GPT_ID,
    "reasoning": {"effort": "low"},
    # temperature/top_p are not accepted alongside reasoning on the Responses API
    "temperature": None,
    "max_output_tokens": None,
    "seed": None,            # Responses API exposes no seed parameter
}
CLAUDE_SETTINGS = {
    "model": CLAUDE_ID,
    "max_tokens": 2000,
    "thinking": {"type": "adaptive", "display": "summarized"},
    "output_config": {"effort": "low"},
    "temperature": None,     # not set; unsupported alongside adaptive thinking
    "seed": None,            # Messages API exposes no seed parameter
}
MODELS = {"gpt": GPT_SETTINGS, "claude": CLAUDE_SETTINGS}

# pair -> (stimuli subdir, identity in the A slot of _AB, identity in the B slot)
SETS = {
    "proto":   ("visual",    "circle",  "square"),
    "nd2":     ("mountains", "p2a",     "p2b"),
    "inkblot": ("visual2",   "ib-A",    "ib-B"),
    "rot":     ("visual",    "blob",    "blob-rot90"),
    "noise":   ("visual2",   "noise-A", "noise-B"),
    "mirror":  ("visual",    "blob",    "blob-mirror"),
}

N_DEFAULT = 60
WORKERS = 8
ASSIGN_SEED = 20260804        # counterbalance allocation only; not an API seed

# ----------------------------------------------------------------- prompt ----
SYSTEM = ("You and another independent instance of the same model will each see the same "
          "pair of stimuli. You cannot communicate. The left-right position and A/B label "
          "of each stimulus are randomized independently for each player, so coordinate on "
          "the stimulus's visual form, not its position or label. Choose the stimulus you "
          "think both of you are most likely to select. Respond with only A or B.")
USER_TEXT = "Choose one stimulus."
# Required addition: the stimuli arrive as one composite image, so the A/B labels
# have to be tied to the two positions in it.
LABEL_NOTE = " The image shows two stimuli side by side: the one on the LEFT is A and the one on the RIGHT is B."
# Stamped on every row so runs collected under different prompt wordings can never be
# pooled by accident.
PROMPT_SHA8 = hashlib.sha256((SYSTEM + "\x00" + USER_TEXT + LABEL_NOTE).encode()).hexdigest()[:8]

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
    r = client.responses.create(
        model=GPT_ID,
        reasoning={"effort": "low"},
        instructions=SYSTEM,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": USER_TEXT + LABEL_NOTE},
            {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"}]}])
    return r.output_text or ""


def ask_claude(client, img_b64):
    r = client.messages.create(
        model=CLAUDE_ID,
        max_tokens=2000,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "low"},
        system=SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": img_b64}},
            {"type": "text", "text": USER_TEXT + LABEL_NOTE}]}])
    return "".join(b.text for b in r.content if b.type == "text")


def parse_ab(txt):
    """Strict: the reply must reduce to exactly A or B. Anything else is invalid."""
    if not txt:
        return ""
    s = re.sub(r"[^A-Za-z]", "", txt).upper()
    return s if s in ("A", "B") else ""

# -------------------------------------------------------------------- jobs ----
def build_jobs(n_per_cell):
    """One job per trial. Variant allocation is exactly half AB / half BA."""
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


def slot_identities(pair, variant):
    """(identity shown as A, identity shown as B) for this variant."""
    _sub, ident_a, ident_b = SETS[pair]
    return (ident_a, ident_b) if variant == "AB" else (ident_b, ident_a)


COLS = ["model", "model_id", "stimulus_pair", "run_id", "attempt", "seed", "prompt_sha8",
        "variant", "option_A_identity", "option_B_identity",
        "response_raw", "response_position", "choice_identity", "valid_response",
        "superseded_by_retry", "api_settings", "timestamp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DEFAULT,
                    help="responses per model x stimulus pair (default 60)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls; emit deterministic fake replies to test the pipeline")
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
    raw_path = os.path.join(OUTDIR, f"step1_calls_{ts}.csv")

    settings_json = {m: json.dumps(s, sort_keys=True) for m, s in MODELS.items()}
    lock = threading.Lock()
    counts = dict(done=0, invalid=0, error=0)
    fh = open(raw_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=COLS)
    writer.writeheader()
    fh.flush()

    def one_call(job, attempt):
        """A single API call. Returns (raw_text, parsed) — never raises."""
        if args.dry_run:
            rng = random.Random(f"{job['run_id']}|{attempt}")
            if rng.random() < 0.04:
                return "I'd pick the round one.", ""       # exercise the retry path
            return rng.choice(["A", "B"]), None
        try:
            fn = ask_gpt if job["model"] == "gpt" else ask_claude
            return fn(clients[job["model"]], images[(job["pair"], job["variant"])]), None
        except Exception as exc:                           # noqa: BLE001 - logged, not raised
            with lock:
                counts["error"] += 1
            return f"__ERR__ {type(exc).__name__}: {exc}", None

    def run_job(job):
        ident_a, ident_b = slot_identities(job["pair"], job["variant"])
        rows = []
        raw, _ = one_call(job, 1)
        pos = parse_ab(raw)
        if not pos:
            # invalid: keep the original response as its own row, then retry once
            rows.append((1, raw, "", "", False, True))
            with lock:
                counts["invalid"] += 1
            raw, _ = one_call(job, 2)
            pos = parse_ab(raw)
        attempt_no = len(rows) + 1
        ident = {"A": ident_a, "B": ident_b}.get(pos, "")
        rows.append((attempt_no, raw, pos, ident, bool(pos), False))

        out = []
        for attempt, raw_txt, position, identity, valid, superseded in rows:
            out.append(dict(
                model=job["model"], model_id=MODELS[job["model"]]["model"],
                stimulus_pair=job["pair"], run_id=job["run_id"], attempt=attempt,
                seed="", prompt_sha8=PROMPT_SHA8, variant=job["variant"],
                option_A_identity=ident_a, option_B_identity=ident_b,
                response_raw=raw_txt.replace("\n", " ").strip()[:500],
                response_position=position, choice_identity=identity,
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

    print("=== STEP 1 · choice distribution ===")
    print(f"models   : {GPT_ID}, {CLAUDE_ID}")
    print(f"stimuli  : {', '.join(SETS)}")
    print(f"trials   : {len(jobs)}  ({args.n} per model x pair)"
          f"{'   [DRY RUN - no API calls]' if args.dry_run else ''}")
    print(f"-> {raw_path}\n", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in as_completed([ex.submit(run_job, j) for j in jobs]):
            pass
    fh.close()

    print(f"\n=== DONE · {counts['done']} trials in {time.time() - t0:.0f}s ===")
    print(f"invalid first responses (retried): {counts['invalid']}")
    print(f"API errors                       : {counts['error']}")
    print(f"raw call-level CSV               : {raw_path}")
    print(f"\nnext: python3 step1_analysis.py {os.path.basename(raw_path)}")


if __name__ == "__main__":
    main()
