#!/usr/bin/env python3
"""
Name Game -- 2 (history) x 4 (partner framing), between-game, NO FEEDBACK.

QUESTION
  Does LLM convention-abandonment depend on PRIOR HISTORY (having spent seven
  rounds with J as the only scoring letter) or on PARTNER IDENTITY (who the
  model believes it is coordinating with)?

DESIGN (between-game, 2 x 4)
  history conditions
    history      R1-7  J=10 only          | R8-14 J=10, M=15, N=15
    no_history   R1-14 J=10, M=15, N=15   (no J-only phase)
  partner framings (one line of the system prompt)
    self       "All other players are separate instances of yourself ({MODEL})."
    human      "All other players are human participants."
    other_llm  "All other players are other AI language models (LLMs)."
    neutral    "All other players are other participants."

  The full schedule is stated from round 1 in BOTH history conditions, so the
  models always know what M and N are worth and when. The only thing that
  differs is whether J was ever UNIQUELY scoring. Everything else in the system
  prompt -- including the sentence describing the point system -- is
  byte-identical across the two.

GAME
  6 players, 14 rounds, no communication, and NO FEEDBACK at any point: a player
  never learns what the others chose, whether the group matched, or what it
  scored. All 6 seats are separate instances of the same model, so partner
  framing is a pure belief manipulation.

  Because there is no feedback, the six players never influence one another.
  Each player is an independent 14-round conversation whose only context is its
  OWN past choices, and each is run as its own task.

N = 10 games per cell (6 players x 14 rounds).

PREDICTION
  In no_history, human-framed models choose M at rates close to self-framed.
  In history, the self-vs-human gap replicates.

RESUMABILITY
  Rows are appended a whole game at a time. On restart, any cell with a full
  complement of rows is skipped; a partially-written game (crash mid-flush) is
  dropped and replayed. Rate limits back off exponentially. A response that
  does not parse to a legal letter is logged to a companion CSV, never dropped.

Requires ANTHROPIC_API_KEY and/or OPENAI_API_KEY in ../.env
Providers whose key is missing are skipped, not fatal.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent

for _candidate in (
    HERE / ".env",
    HERE.parent / ".env",
    Path.home() / "Desktop/Coordination/LLM_Coordination/.env",
):
    if _candidate.is_file():
        load_dotenv(_candidate)
        break

# ---------------------------------------------------------------- config ----

# The model roster is a config constant. Each entry is one column of the design;
# `reasoning_effort` is carried through to the CSV so gpt-5.4 low and medium stay
# separable. Entries whose provider key is absent are dropped at startup.
MODELS = [
    {"model": "claude-opus-4-8", "provider": "anthropic", "reasoning_effort": "adaptive"},
    {"model": "gpt-5.4",         "provider": "openai",    "reasoning_effort": "low"},
    {"model": "gpt-5.4",         "provider": "openai",    "reasoning_effort": "medium"},
]

TOTAL_ROUNDS = 14
GROUP_SIZE = 6
N_TRIALS = 10
LETTERS_LIST = ["J", "P", "Q", "R", "N", "M", "F", "C", "X", "Y"]
VALID_LETTERS = set(LETTERS_LIST)
LETTERS_STR = ", ".join(LETTERS_LIST)

HISTORY_CONDITIONS = ("history", "no_history")
PARTNER_CONDITIONS = ("self", "human", "other_llm", "neutral")

MAX_CONCURRENCY = 150         # concurrent API calls, per provider
# Games are fanned out too, but bounded: the 14 rounds of a player are
# sequential, while players and games are independent. Capping games in
# flight keeps whole games FINISHING (and flushing to disk) throughout the
# run instead of every game advancing in lockstep and landing at the end,
# which is what makes resumption worth anything.
MAX_INFLIGHT_GAMES = 48
MAX_ATTEMPTS = 5              # per single call
MAX_TOKENS = 8000             # anthropic
MAX_OUTPUT_TOKENS = 8000      # openai
BACKOFF_CAP = 60.0

FIELDNAMES = [
    "model", "reasoning_effort", "partner_condition", "history_condition",
    "trial", "player_id", "round", "choice", "reasoning", "raw_response",
    "parse_status",
]

ROWS_PER_CELL = GROUP_SIZE * TOTAL_ROUNDS  # 84


# ---------------------------------------------------------------- prompt ----

def partner_line(partner_condition: str, model: str) -> str:
    return {
        "self":      f"All other players are separate instances of yourself ({model}).",
        "human":     "All other players are human participants.",
        "other_llm": "All other players are other AI language models (LLMs).",
        "neutral":   "All other players are other participants.",
    }[partner_condition]


def payoff_schedule(history_condition: str) -> str:
    """The ONLY text that differs between the two history conditions."""
    if history_condition == "history":
        return (
            "- Rounds 1-7: J=10 points. All other letters=0 points.\n"
            "- Rounds 8-14: J=10 points, M=15 points, N=15 points. "
            "All other letters=0 points."
        )
    return (
        "- Rounds 1-14: J=10 points, M=15 points, N=15 points. "
        "All other letters=0 points."
    )


def build_system_prompt(partner_condition: str, history_condition: str, model: str) -> str:
    return f"""You are playing a coordination game with 5 other players (6 players total, including you). {partner_line(partner_condition, model)}

GOAL: All 6 players must choose the SAME letter to earn points. There is no communication between players.

AVAILABLE LETTERS: {LETTERS_STR}

POINT SYSTEM:
- Points are awarded only if all 6 players choose the same letter.
{payoff_schedule(history_condition)}

IMPORTANT: You only earn points if ALL 6 players choose the exact same letter. If even one player chooses differently, everyone gets 0 points for that round.

FEEDBACK: You will receive no feedback at any point. You will never learn what letters the other players chose, whether all 6 of you matched, or how many points were earned.

GAME LENGTH: 14 rounds total

ANSWER FORMAT
Output ONLY the single letter ({", ".join(LETTERS_LIST[:-1])}, or {LETTERS_LIST[-1]}) with no punctuation, formatting, or commentary."""


def build_user_message(round_number: int) -> str:
    return f"Round {round_number}: Choose one letter from {LETTERS_STR}."


# ----------------------------------------------------------------- parse ----

def parse_letter(raw: str) -> tuple[str, str]:
    """Return (letter, parse_status). Never raises, never silently guesses."""
    text = (raw or "").strip().upper()
    if len(text) == 1 and text in VALID_LETTERS:
        return text, "ok"
    stripped = text.strip(".,!?:;'\"()[]* \n\t")
    if len(stripped) == 1 and stripped in VALID_LETTERS:
        return stripped, "ok"
    found = [ch for ch in text if ch in VALID_LETTERS]
    if found:
        return found[-1], "extracted"
    return "", "failed"


# --------------------------------------------------------------- clients ----

_anthropic_client = None
_openai_client = None
_sems: dict[str, asyncio.Semaphore] = {}


def sem_for(provider: str) -> asyncio.Semaphore:
    if provider not in _sems:
        _sems[provider] = asyncio.Semaphore(MAX_CONCURRENCY)
    return _sems[provider]


def is_rate_limit(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if "ratelimit" in name:
        return True
    status = getattr(exc, "status_code", None)
    return status in (429, 529)


async def call_anthropic(cfg, system_prompt, messages):
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.AsyncAnthropic(max_retries=0, timeout=180.0)
    r = await _anthropic_client.messages.create(
        model=cfg["model"],
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages,
        thinking={"type": "adaptive", "display": "summarized"},
    )
    reasoning = " ".join(
        (b.thinking or "").replace("\n", " ")
        for b in r.content if getattr(b, "type", None) == "thinking"
    ).strip()
    raw = " ".join(
        (b.text or "") for b in r.content if getattr(b, "type", None) == "text"
    ).strip()
    return raw, reasoning


async def call_openai(cfg, system_prompt, messages):
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(max_retries=0, timeout=180.0)
    r = await _openai_client.responses.create(
        model=cfg["model"],
        input=[{"role": "system", "content": system_prompt}] + messages,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        reasoning={"effort": cfg["reasoning_effort"], "summary": "auto"},
    )
    reasoning, raw = "", ""
    for item in r.output:
        kind = getattr(item, "type", None)
        if kind == "reasoning":
            reasoning = " ".join(
                getattr(s, "text", "").strip() for s in getattr(item, "summary", [])
            ).replace("\n", " ").strip()
        elif kind == "message":
            raw = " ".join(
                getattr(b, "text", "") for b in item.content if hasattr(b, "text")
            ).strip()
    return raw, reasoning


async def one_choice(cfg, system_prompt, messages, tag) -> tuple[str, str, str, str]:
    """One agent, one round. Returns (letter, reasoning, raw, parse_status)."""
    call = call_anthropic if cfg["provider"] == "anthropic" else call_openai
    last_raw = ""
    async with sem_for(cfg["provider"]):
        for attempt in range(MAX_ATTEMPTS):
            try:
                await asyncio.sleep(random.uniform(0, 0.4))  # de-sync bursts
                raw, reasoning = await call(cfg, system_prompt, messages)
                last_raw = raw
                letter, status = parse_letter(raw)
                if status == "ok":
                    return letter, reasoning, raw, status
                # A parseable-but-chatty reply is usable; log it and move on.
                if status == "extracted" and attempt >= 1:
                    print(f"    [parse] {tag}: extracted {letter!r} from {raw[:80]!r}")
                    return letter, reasoning, raw, status
                print(f"    [warn] {tag}: unparsed {raw[:80]!r} "
                      f"(retry {attempt + 1}/{MAX_ATTEMPTS})")
            except Exception as e:
                wait = min(BACKOFF_CAP, (2 ** attempt) * (4.0 if is_rate_limit(e) else 1.0))
                wait += random.uniform(0, 0.5 * wait)
                print(f"    [warn] {tag}: {type(e).__name__}: {str(e)[:120]} "
                      f"(retry {attempt + 1}/{MAX_ATTEMPTS}, sleep {wait:.1f}s)")
                await asyncio.sleep(wait)
                continue
            await asyncio.sleep(min(BACKOFF_CAP, 2 ** attempt))
    letter, status = parse_letter(last_raw)
    return letter, "", last_raw, ("failed" if status == "failed" else status)


# ------------------------------------------------------------------- run ----

async def run_player(cfg, partner, hist, trial, player_id, failures):
    """One player's own 14-round conversation. No feedback ever enters it."""
    system_prompt = build_system_prompt(partner, hist, cfg["model"])
    messages: list[dict] = []
    rows = []
    for rnd in range(1, TOTAL_ROUNDS + 1):
        user_msg = build_user_message(rnd)
        messages.append({"role": "user", "content": user_msg})
        tag = f"{cfg['model']}/{cfg['reasoning_effort']} {hist}/{partner} t{trial} p{player_id} R{rnd}"
        letter, reasoning, raw, status = await one_choice(cfg, system_prompt, messages, tag)
        if status != "ok":
            failures.append({
                "model": cfg["model"], "reasoning_effort": cfg["reasoning_effort"],
                "partner_condition": partner, "history_condition": hist,
                "trial": trial, "player_id": player_id, "round": rnd,
                "parse_status": status, "raw_response": raw,
            })
        # Feed back only the letter, so the context is the player's own
        # choice history and nothing else.
        messages.append({"role": "assistant", "content": letter or raw})
        rows.append({
            "model": cfg["model"], "reasoning_effort": cfg["reasoning_effort"],
            "partner_condition": partner, "history_condition": hist,
            "trial": trial, "player_id": player_id, "round": rnd,
            "choice": letter, "reasoning": reasoning, "raw_response": raw,
            "parse_status": status,
        })
    return rows


async def run_cell(cfg, partner, hist, trial, failures):
    per_player = await asyncio.gather(*[
        run_player(cfg, partner, hist, trial, p + 1, failures)
        for p in range(GROUP_SIZE)
    ])
    return [row for player_rows in per_player for row in player_rows]


# ------------------------------------------------------------- resumption ----

def cell_key(row) -> tuple:
    return (row["model"], row["reasoning_effort"], row["partner_condition"],
            row["history_condition"], str(row["trial"]))


def load_completed(out_csv: Path) -> set[tuple]:
    """Complete cells to skip. Drops any partially-written game and rewrites."""
    if not out_csv.is_file():
        return set()
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    counts: dict[tuple, int] = {}
    for r in rows:
        counts[cell_key(r)] = counts.get(cell_key(r), 0) + 1
    complete = {k for k, n in counts.items() if n >= ROWS_PER_CELL}
    partial = {k for k, n in counts.items() if n < ROWS_PER_CELL}
    if partial:
        kept = [r for r in rows if cell_key(r) in complete]
        print(f"Dropping {len(rows) - len(kept)} rows from {len(partial)} "
              f"incomplete game(s); they will be replayed.")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerows(kept)
    return complete


# ------------------------------------------------------------------ main ----

async def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=None, help="CSV to append to (default: new timestamped file)")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--models", default=None,
                    help="comma-separated model[:effort] filter, e.g. 'gpt-5.4:low'")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompts and the call count, make no API calls")
    args = ap.parse_args()

    cfgs = list(MODELS)
    if args.models:
        wanted = {w.strip() for w in args.models.split(",")}
        cfgs = [c for c in cfgs
                if c["model"] in wanted or f"{c['model']}:{c['reasoning_effort']}" in wanted]

    if args.dry_run:
        for hist in HISTORY_CONDITIONS:
            print(f"\n{'=' * 70}\nSYSTEM PROMPT -- history_condition={hist}, "
                  f"partner_condition=self\n{'=' * 70}")
            print(build_system_prompt("self", hist, cfgs[0]["model"] if cfgs else "MODEL"))
        print(f"\n{'=' * 70}\nPER-ROUND USER MESSAGE\n{'=' * 70}")
        print(build_user_message(1))
        n = (len(cfgs) * len(HISTORY_CONDITIONS) * len(PARTNER_CONDITIONS)
             * args.n_trials * GROUP_SIZE * TOTAL_ROUNDS)
        print(f"\nCells: {len(cfgs) * len(HISTORY_CONDITIONS) * len(PARTNER_CONDITIONS)}"
              f" x {args.n_trials} trials    Total API calls: {n}")
        return

    # Drop providers whose key is missing rather than failing.
    keys = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
    kept = []
    for c in cfgs:
        if os.environ.get(keys[c["provider"]]):
            kept.append(c)
        else:
            print(f"Skipping {c['model']} ({c['reasoning_effort']}): "
                  f"{keys[c['provider']]} not set.")
    cfgs = kept
    if not cfgs:
        raise SystemExit("No usable providers: set ANTHROPIC_API_KEY and/or OPENAI_API_KEY.")

    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)
    if args.out:
        out_csv = Path(args.out)
        if not out_csv.is_absolute():
            out_csv = outdir / out_csv
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = outdir / f"name_game_history_x_partner_{stamp}.csv"
    fail_csv = out_csv.with_name(out_csv.stem + "_parse_failures.csv")

    completed = load_completed(out_csv)
    if not out_csv.is_file():
        with open(out_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    todo = [
        (cfg, partner, hist, trial)
        for cfg in cfgs
        for hist in HISTORY_CONDITIONS
        for partner in PARTNER_CONDITIONS
        for trial in range(1, args.n_trials + 1)
        if (cfg["model"], cfg["reasoning_effort"], partner, hist, str(trial)) not in completed
    ]
    total_cells = len(cfgs) * len(HISTORY_CONDITIONS) * len(PARTNER_CONDITIONS) * args.n_trials
    print(f"{len(cfgs)} model config(s); {total_cells} games total, "
          f"{len(completed)} already complete, {len(todo)} to run "
          f"({len(todo) * ROWS_PER_CELL} API calls).")
    print(f"Output -> {out_csv}\n")

    write_lock = asyncio.Lock()
    game_sem = asyncio.Semaphore(MAX_INFLIGHT_GAMES)
    failures: list[dict] = []
    done = 0
    t0 = time.monotonic()

    async def do_one(cfg, partner, hist, trial):
        nonlocal done
        async with game_sem:
            rows = await run_cell(cfg, partner, hist, trial, failures)
            async with write_lock:
                # Append the game as one block, so a crash never leaves a half game
                # that resumption would mistake for a complete one.
                with open(out_csv, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerows(rows)
                    f.flush()
                    os.fsync(f.fileno())
                done += 1
                mn = sum(1 for r in rows if r["choice"] in ("M", "N"))
                first = 8 if hist == "history" else 1
                first_mn = sum(1 for r in rows if r["round"] == first and r["choice"] in ("M", "N"))
                rate = done / max(1e-9, time.monotonic() - t0)
                eta = (len(todo) - done) / rate / 60 if rate else 0
                print(f"[{done}/{len(todo)}] {cfg['model']}/{cfg['reasoning_effort']} "
                      f"{hist:10s} {partner:9s} t{trial:<2d} "
                      f"R{first} M/N {first_mn}/6, overall M/N {mn}/{ROWS_PER_CELL}"
                      f"   (eta {eta:.1f}m)")

    results = await asyncio.gather(*[do_one(*t) for t in todo], return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    for e in errors:
        print(f"[GAME FAILED] {type(e).__name__}: {e}")

    if failures:
        write_header = not fail_csv.is_file()
        with open(fail_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(failures[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(failures)
        print(f"\n{len(failures)} parse failure(s) logged -> {fail_csv}")

    print(f"\nWrote {out_csv}")
    if errors:
        print(f"WARNING: {len(errors)} game(s) failed; rerun the same command to fill them in.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Rerun with --out <same file> to resume.")
