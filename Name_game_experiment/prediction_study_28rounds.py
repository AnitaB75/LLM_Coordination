#!/usr/bin/env python3
"""
Experiment 2 (prediction study), rerun with the 28-round framing.

TASK (from the paper): the participant is told the rules, shown rounds 1-7 in
which all six players chose J, and asked to imagine being Player 1 about to play
round 8. They predict how many of their 5 teammates will switch to M, how many
to N, and how many will stick with J (must sum to 5), then state their own
round-8 choice. Framing is randomised: the other 5 players are described either
as human participants or as other LLMs.

DESIGN  2 framings x 2 horizons x N_PER_CELL independent agents
  framing:  human | llm
  horizon:  14 (control, replicates the published cell) | 28 (the manipulation)

The 14-round arm is a CONTROL for prompt reconstruction. The published result is
that GPT-5.5 always stays with J in both framings; if this arm reproduces that,
a 28-round difference is attributable to horizon rather than to wording drift
between this reconstruction and the original instrument.

The rules text is the oTree system prompt (identical to Experiment 1), so the
prediction task is anchored to the same game description the players saw.
Each agent is an INDEPENDENT call -- no shared context, as with separate
participants.

NOTE ON LETTER COUNT: the paper's prose says "eight letters"; the oTree
implementation uses ten (J P Q R N M F C X Y). We follow the implementation, so
this matches Experiment 1 and everything else in the repo.

Requires OPENAI_API_KEY in a .env (beside this script, its parent, or the
project root) or in the environment.
"""

from __future__ import annotations

import asyncio
import csv
import os
import random
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
for _c in (HERE / ".env", HERE.parent / ".env",
           Path.home() / "Desktop/Coordination/LLM_Coordination/.env"):
    if _c.is_file():
        load_dotenv(_c)
        break
if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY not found.")

MODEL = "gpt-5.5"
REASONING_LVL = "medium"
CHANGE_ROUND = 8
N_PER_CELL = 30           # the published study used 10 per condition
MAX_OUTPUT_TOKENS = 2000
MAX_CONCURRENCY = 10
HORIZONS = (14, 28)
FRAMINGS = {"human": (5, 1), "llm": (0, 6)}


class Prediction(BaseModel):
    n_switch_M: int
    n_switch_N: int
    n_stay_J: int
    own_choice: str


def build_system_prompt(num_humans: int, num_llms: int, total_rounds: int) -> str:
    """oTree _build_system_prompt(), parameterised by total_rounds."""
    total = num_humans + num_llms
    other_ais = num_llms - 1
    if num_humans == 0:
        teammate_desc = f"The other {total - 1} players are other AI language models."
    elif other_ais == 0:
        h = f"{num_humans} human participant" + ("s" if num_humans > 1 else "")
        teammate_desc = f"The other {total - 1} players are {h}."
    else:
        h = f"{num_humans} human participant" + ("s" if num_humans > 1 else "")
        a = f"{other_ais} other AI language model" + ("s" if other_ais > 1 else "")
        teammate_desc = f"The other {total - 1} players are {h} and {a}."
    history_fmt = ("History format: You=your choice, H=[other players' choices], "
                   "Σ=your cumulative points." if num_humans > 0 else
                   "History format: You=your choice, A=[other players' choices], "
                   "Σ=your cumulative points.")
    return f"""Coordination game: {total} players. {teammate_desc}
Each round everyone simultaneously picks a letter.
Points only if ALL {total} choose the same letter. {total_rounds} rounds, same group, no communication.

Letters: J P Q R N M F C X Y

POINT SYSTEM:
- Rounds 1-{CHANGE_ROUND - 1}:  J=10pts if all match. All other letters=0pts.
- Rounds {CHANGE_ROUND}-{total_rounds}: J=10pts, M=20pts, N=20pts if all match. All other letters=0pts.

{history_fmt}"""


def build_user_message(num_humans: int, total_rounds: int) -> str:
    field = "H" if num_humans > 0 else "A"
    lines = [f"R{r}: You=J {field}=[J,J,J,J,J] → MATCH +10 (Σ{10*r})"
             for r in range(1, CHANGE_ROUND)]
    return (
        f"It is now Round {CHANGE_ROUND} of {total_rounds}.\n\n"
        "PREVIOUS ROUNDS:\n" + "\n".join(lines) + "\n\n"
        "Before you choose, predict what your 5 teammates will do in Round "
        f"{CHANGE_ROUND}.\n"
        "Of your 5 teammates, how many will switch to M, how many will switch to N, "
        "and how many will stick with J? The three numbers must sum to 5.\n\n"
        f"Then state what you will choose for Round {CHANGE_ROUND}."
    )


client = AsyncOpenAI()
sem = asyncio.Semaphore(MAX_CONCURRENCY)


async def one_participant(framing: str, horizon: int, pid: int):
    num_humans, num_llms = FRAMINGS[framing]
    sys_p = build_system_prompt(num_humans, num_llms, horizon)
    usr_p = build_user_message(num_humans, horizon)
    async with sem:
        await asyncio.sleep(random.uniform(0, 0.4))
        for attempt in range(4):
            try:
                kwargs = dict(
                    model=MODEL,
                    input=[{"role": "system", "content": sys_p},
                           {"role": "user", "content": usr_p}],
                    text_format=Prediction,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )
                if REASONING_LVL != "none":
                    kwargs["reasoning"] = {"effort": REASONING_LVL, "summary": "auto"}
                r = await client.responses.parse(**kwargs)
                p = r.output_parsed
                if p is None:
                    raise ValueError("output_parsed is None")
                total = p.n_switch_M + p.n_switch_N + p.n_stay_J
                if total != 5:
                    print(f"    [warn] predictions sum to {total}, not 5 "
                          f"(retry {attempt+1}/4)")
                    await asyncio.sleep(2 ** attempt)
                    continue
                choice = (p.own_choice or "").strip().upper()[:1]
                reasoning = ""
                for item in r.output:
                    if getattr(item, "type", None) == "reasoning":
                        reasoning = " ".join(getattr(s, "text", "").strip()
                                             for s in getattr(item, "summary", [])
                                             ).replace("\n", " ").strip()
                        break
                return {
                    "framing": framing, "horizon": horizon, "participant": pid,
                    "n_switch_M": p.n_switch_M, "n_switch_N": p.n_switch_N,
                    "n_stay_J": p.n_stay_J,
                    "n_switch_MN": p.n_switch_M + p.n_switch_N,
                    "own_choice": choice,
                    "own_switch": int(choice in ("M", "N")),
                    "reasoning": reasoning,
                }
            except Exception as e:
                print(f"    [warn] {type(e).__name__}: {e} (retry {attempt+1}/4)")
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("failed after 4 attempts")


async def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = HERE / "results"; outdir.mkdir(exist_ok=True)

    tasks = [one_participant(f, h, i)
             for h in HORIZONS for f in FRAMINGS for i in range(N_PER_CELL)]
    print(f"Running {len(tasks)} agents "
          f"({len(HORIZONS)} horizons x {len(FRAMINGS)} framings x {N_PER_CELL}) "
          f"on {MODEL}\n")
    res = await asyncio.gather(*tasks, return_exceptions=True)
    rows = [r for r in res if not isinstance(r, Exception)]
    fails = len(res) - len(rows)

    if not rows:
        first = next((r for r in res if isinstance(r, Exception)), None)
        raise SystemExit(
            f"No agents succeeded ({fails} failed); nothing written.\n"
            f"First error: {type(first).__name__}: {first}"
        )

    out = outdir / f"prediction_study_28rounds_{stamp}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out}")
    if fails:
        print(f"WARNING: {fails} agent(s) failed.")

    print("\nPREDICTED number of 5 teammates switching to M/N, and OWN round-8 choice")
    print(f"{'horizon':>8s} {'framing':>8s} {'mean pred M/N':>14s} "
          f"{'own switch %':>13s}  own-choice counts")
    for h in HORIZONS:
        for fr in FRAMINGS:
            s = [r for r in rows if r["horizon"] == h and r["framing"] == fr]
            if not s:
                continue
            mn = sum(r["n_switch_MN"] for r in s) / len(s)
            sw = sum(r["own_switch"] for r in s) / len(s)
            counts = {}
            for r in s:
                counts[r["own_choice"]] = counts.get(r["own_choice"], 0) + 1
            print(f"{h:>8d} {fr:>8s} {mn:>14.2f} {sw:>12.0%}  {counts}")


if __name__ == "__main__":
    asyncio.run(main())
