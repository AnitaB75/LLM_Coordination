#!/usr/bin/env python3
"""
Name Game -- 28 ROUNDS, gpt-5.5, two framings x 6 groups.

TESTS: is LLM entrenchment a payback-horizon calculation rather than a
preference for the incumbent convention? With only 14 rounds, a group that
transitions at round 8 has at most 7 rounds to recoup the miscoordination cost,
so staying on J can be individually rational. Doubling the game to 28 rounds
leaves 21 post-change rounds to recoup. If entrenchment is horizon-driven,
switching should rise sharply; if it is convention-lock, it should not move.

THE ONLY DIFFERENCE from the oTree prompt is the round count:
    "14 rounds"       -> "28 rounds"
    "Rounds 8-14:"    -> "Rounds 8-28:"
    "Round X of 14"   -> "Round X of 28"
The payoff change stays at ROUND 8, so the entrenchment period is unchanged at
7 rounds and only the post-change horizon grows (7 -> 21). Moving the switch
point would confound horizon with entrenchment depth.

CONDITIONS (6 groups each, 6 agents, 28 rounds)
  human      "The other 5 players are 5 human participants."   [H=... history]
  other_llm  "The other 5 players are other AI language models." [A=... history]

Both are pure BELIEF manipulations: all 6 seats are gpt-5.5 in both conditions.
Nobody is actually human. This mirrors the oTree renderings of
_build_system_prompt(5, 1) and _build_system_prompt(0, 6) respectively.

API config is identical to oTree: Responses API, reasoning effort "medium" with
auto summary, structured output, max_output_tokens=1500.

PROMPT FIDELITY NOTE
  oTree's _build_llm_messages() emits BOTH "H=[...]" and "A=[...]" on every
  history line regardless of composition, so the homogeneous renderings would
  carry a contradictory empty field ("A=[]" with 0 other AIs, "H=[]" with 0
  humans). The system prompt's declared format names only one field in each
  homogeneous case, so we follow the declared format and omit the empty one.

Requires OPENAI_API_KEY in a .env (searched beside this script, its parent, and
the project root) or in the environment.
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
    raise SystemExit("OPENAI_API_KEY not found. Put it in a .env beside this "
                     "script, or export it in your shell.")

MODEL = "gpt-5.5"
REASONING_LVL = "medium"
TOTAL_ROUNDS = 28          # <-- the manipulation (oTree was 14)
CHANGE_ROUND = 8           # payoff change stays at round 8
GROUP_SIZE = 6
N_GROUPS = 6
MAX_OUTPUT_TOKENS = 1500
MAX_CONCURRENCY = 12
VALID_LETTERS = {"J", "P", "Q", "R", "N", "M", "F", "C", "X", "Y"}

# (num_humans, num_llms) as oTree would be configured for a 6-player group
CONDITIONS = {
    "human":     (5, 1),
    "other_llm": (0, 6),
}


class LLMLetterOutput(BaseModel):
    letter: str


def build_system_prompt(num_humans: int, num_llms: int,
                        total_rounds: int = TOTAL_ROUNDS) -> str:
    """oTree _build_system_prompt(), parameterised by total_rounds."""
    total = num_humans + num_llms
    other_ais = num_llms - 1

    if num_humans == 0:
        teammate_desc = f"The other {total - 1} players are other AI language models."
    elif other_ais == 0:
        h_str = f"{num_humans} human participant" + ("s" if num_humans > 1 else "")
        teammate_desc = f"The other {total - 1} players are {h_str}."
    else:
        h_str = f"{num_humans} human participant" + ("s" if num_humans > 1 else "")
        ai_str = f"{other_ais} other AI language model" + ("s" if other_ais > 1 else "")
        teammate_desc = f"The other {total - 1} players are {h_str} and {ai_str}."

    if num_humans > 0 and other_ais > 0:
        history_fmt = ("History format: You=your choice, H=[human choices], "
                       "A=[other AI choices], Σ=your cumulative points.")
    elif num_humans > 0:
        history_fmt = ("History format: You=your choice, H=[other players' choices], "
                       "Σ=your cumulative points.")
    else:
        history_fmt = ("History format: You=your choice, A=[other players' choices], "
                       "Σ=your cumulative points.")

    return f"""Coordination game: {total} players. {teammate_desc}
Each round everyone simultaneously picks a letter.
Points only if ALL {total} choose the same letter. {total_rounds} rounds, same group, no communication.

Letters: J P Q R N M F C X Y

POINT SYSTEM:
- Rounds 1-{CHANGE_ROUND - 1}:  J=10pts if all match. All other letters=0pts.
- Rounds {CHANGE_ROUND}-{total_rounds}: J=10pts, M=20pts, N=20pts if all match. All other letters=0pts.

{history_fmt}"""


def round_points(choices: list[str], rnd: int) -> int:
    """oTree _fetch_round_history scoring."""
    if len(set(choices)) == 1 and len(choices) == GROUP_SIZE and "?" not in choices:
        letter = choices[0]
        if letter == "J":
            return 10
        if letter in ("M", "N") and rnd >= CHANGE_ROUND:
            return 20
    return 0


def build_user_message(history, agent_idx, round_number, num_humans,
                       total_rounds: int = TOTAL_ROUNDS) -> str:
    """oTree _build_llm_messages(); homogeneous groups use the declared field only."""
    field = "H" if num_humans > 0 else "A"
    lines, cumulative = [], 0
    for rd in history:
        mine = rd["choices"][agent_idx]
        others = [c for i, c in enumerate(rd["choices"]) if i != agent_idx]
        cumulative += rd["pts"]
        match_str = f"MATCH +{rd['pts']}" if rd["pts"] > 0 else "no match"
        lines.append(
            f"R{rd['r']}: You={mine} {field}=[{','.join(others)}] → {match_str} (Σ{cumulative})"
        )
    history_text = ("PREVIOUS ROUNDS:\n" + "\n".join(lines) if lines
                    else "This is the first round — no history yet.")
    return (f"It is now Round {round_number} of {total_rounds}.\n\n"
            f"{history_text}\n\n"
            f"Please choose your letter for Round {round_number}.")


client = AsyncOpenAI()
sem = asyncio.Semaphore(MAX_CONCURRENCY)


async def one_choice(system_prompt: str, user_message: str):
    """One agent, one round. Returns (letter, reasoning_summary)."""
    async with sem:
        await asyncio.sleep(random.uniform(0, 0.4))
        for attempt in range(4):
            try:
                kwargs = dict(
                    model=MODEL,
                    input=[{"role": "system", "content": system_prompt},
                           {"role": "user", "content": user_message}],
                    text_format=LLMLetterOutput,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )
                if REASONING_LVL != "none":
                    kwargs["reasoning"] = {"effort": REASONING_LVL, "summary": "auto"}
                r = await client.responses.parse(**kwargs)
                parsed = r.output_parsed
                reasoning = ""
                for item in r.output:
                    if getattr(item, "type", None) == "reasoning":
                        reasoning = " ".join(
                            getattr(s, "text", "").strip()
                            for s in getattr(item, "summary", [])
                        ).replace("\n", " ").strip()
                        break
                letter = (parsed.letter or "").strip().upper()[:1] if parsed else ""
                if letter in VALID_LETTERS:
                    return letter, reasoning
                print(f"    [warn] invalid letter {letter!r} (retry {attempt+1}/4)")
            except Exception as e:
                print(f"    [warn] {type(e).__name__}: {e} (retry {attempt+1}/4)")
            await asyncio.sleep(2 ** attempt)
        raise RuntimeError("failed to obtain a valid letter after 4 attempts")


async def run_group(cond_name: str, group_idx: int):
    num_humans, num_llms = CONDITIONS[cond_name]
    system_prompt = build_system_prompt(num_humans, num_llms)
    group_key = f"gpt55_28r_{cond_name}_G{group_idx + 1}"
    history, rows, reasoning_rows = [], [], []

    for rnd in range(1, TOTAL_ROUNDS + 1):
        results = await asyncio.gather(*[
            one_choice(system_prompt,
                       build_user_message(history, i, rnd, num_humans))
            for i in range(GROUP_SIZE)
        ])
        choices = [r[0] for r in results]
        pts = round_points(choices, rnd)
        history.append({"r": rnd, "choices": choices, "pts": pts})

        for i, (letter, reasoning) in enumerate(results):
            rows.append({
                "dataset_source": "llm_gpt55_28r",
                "condition": cond_name,
                "session": f"gpt55_28r_{cond_name}",
                "group_id_raw": group_idx + 1,
                "group_key": group_key,
                "actor_type": "LLM",
                "actor_id": f"{group_key}_L{i+1}",
                "actor_slot": i + 1,
                "actor_id_stable": True,
                "round": rnd,
                "letter": letter,
                "choice_category": letter if letter in ("J", "M", "N") else "Other",
                "switch_mn": int(letter in ("M", "N")),
                "post_change": int(rnd >= CHANGE_ROUND),
                "num_humans": num_humans,
                "num_llms": num_llms,
                "total_rounds": TOTAL_ROUNDS,
            })
            reasoning_rows.append({
                "condition": cond_name, "group_key": group_key,
                "actor_slot": i + 1, "round": rnd,
                "letter": letter, "reasoning": reasoning,
            })
        print(f"  [{cond_name}] G{group_idx+1} R{rnd:2d}: {' '.join(choices)}  (+{pts})")
    return rows, reasoning_rows


async def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)

    tasks = [run_group(name, g) for name in CONDITIONS for g in range(N_GROUPS)]
    print(f"Launching {len(tasks)} groups x {TOTAL_ROUNDS} rounds x {GROUP_SIZE} agents "
          f"= {len(tasks)*TOTAL_ROUNDS*GROUP_SIZE} calls on {MODEL}\n")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_rows, all_reasoning, failures = [], [], 0
    for res in results:
        if isinstance(res, Exception):
            failures += 1
            print(f"[GROUP FAILED] {type(res).__name__}: {res}")
            continue
        all_rows.extend(res[0])
        all_reasoning.extend(res[1])

    if all_rows:
        out_csv = outdir / f"name_game_gpt55_28rounds_{stamp}.csv"
        out_reas = outdir / f"name_game_gpt55_28rounds_reasoning_{stamp}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
        with open(out_reas, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_reasoning[0].keys()))
            w.writeheader(); w.writerows(all_reasoning)
        print(f"\nWrote {len(all_rows)} rows -> {out_csv}")
        print(f"Wrote {len(all_reasoning)} reasoning rows -> {out_reas}")

    if failures:
        print(f"WARNING: {failures} group(s) failed and are absent from the CSV.")

    print("\nM/N adoption by condition:")
    for name in CONDITIONS:
        post = [r for r in all_rows if r["condition"] == name and r["post_change"] == 1]
        early = [r for r in post if r["round"] < 15]
        late = [r for r in post if r["round"] >= 15]
        if post:
            f = lambda rs: sum(r["switch_mn"] for r in rs) / len(rs) if rs else float("nan")
            print(f"  {name:10s} R8-14 {f(early):.3f} | R15-28 {f(late):.3f} | "
                  f"all post {f(post):.3f}  (n={len(post)})")


if __name__ == "__main__":
    asyncio.run(main())
