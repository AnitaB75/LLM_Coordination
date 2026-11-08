#!/usr/bin/env python3
"""
Name Game -- LLM-only, claude-opus-4-8, 3 conditions x 6 groups.

Replicates the oTree `name_game_mixed` LLM prompt in its LLM-ONLY rendering
(_build_system_prompt(num_humans=0, num_llms=6)), so these rows bind directly to
the mixed-group LLM rows in name_game_combined_human_llm_mixed.csv.

CONDITIONS (6 groups each, 6 agents, 14 rounds)
  1. static_adaptive   full 1-14 payoff schedule stated from round 1  | thinking adaptive
                       ^ byte-identical to the oTree LLM prompt
  2. reveal8_adaptive  M/N=20 withheld until round 8                  | thinking adaptive
  3. static_nothink    full schedule from round 1                     | thinking disabled

Condition 1 is the shared reference cell: 2 isolates payoff FOREKNOWLEDGE,
3 isolates THINKING. Condition 2 is the LLM analogue of what humans actually
experienced in the oTree study, where the choice screen renders point values per
round and M/N read "0 pts" until round 8 with no announcement.

PROMPT FIDELITY NOTES
  - The oTree _build_llm_messages() appends "H=[]" to every history line even
    when there are no humans. That path never ran (the app only fielded 2-4
    human groups) and it contradicts the system prompt, which for num_humans=0
    declares the format as "You=your choice, A=[other players' choices]".
    We follow the declared format and omit H entirely.
  - Only the letter is fed back into history, never the thinking summary --
    matching oTree, where reasoning was logged but never returned to context.
  - Each call is STATELESS: system + one user message rebuilt from scratch,
    exactly as oTree does it.

Payoffs are identical in all conditions (J=10 always; M/N=20 from round 8).
Only the INFORMATION differs in condition 2.

Requires ANTHROPIC_API_KEY in ../.env
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent

# Find the .env wherever this script is run from (repo, Downloads, anywhere).
# Falls through to the ambient ANTHROPIC_API_KEY if no .env is found.
for _candidate in (
    HERE / ".env",
    HERE.parent / ".env",
    Path.home() / "Desktop/Coordination/LLM_Coordination/.env",
):
    if _candidate.is_file():
        load_dotenv(_candidate)
        break
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit(
        "ANTHROPIC_API_KEY not found. Put it in a .env beside this script, "
        "or export it in your shell."
    )

MODEL = "claude-opus-4-8"
TOTAL_ROUNDS = 14
GROUP_SIZE = 6
N_GROUPS = 6
MAX_TOKENS = 8000
MAX_CONCURRENCY = 12
VALID_LETTERS = {"J", "P", "Q", "R", "N", "M", "F", "C", "X", "Y"}

CONDITIONS = {
    "static_adaptive":  {"reveal": "static",  "thinking": {"type": "adaptive", "display": "summarized"}},
    "reveal8_adaptive": {"reveal": "reveal8", "thinking": {"type": "adaptive", "display": "summarized"}},
    "static_nothink":   {"reveal": "static",  "thinking": {"type": "disabled"}},
}


class LetterChoice(BaseModel):
    letter: str


def build_system_prompt(reveal: str, round_number: int) -> str:
    """oTree _build_system_prompt(0, 6). `reveal` controls the POINT SYSTEM block only."""
    if reveal == "static" or round_number >= 8:
        if reveal == "static":
            points = (
                "- Rounds 1-7:  J=10pts if all match. All other letters=0pts.\n"
                "- Rounds 8-14: J=10pts, M=20pts, N=20pts if all match. All other letters=0pts."
            )
        else:  # reveal8, round >= 8: current values only, no announcement
            points = "- J=10pts, M=20pts, N=20pts if all match. All other letters=0pts."
    else:  # reveal8, rounds 1-7
        points = "- J=10pts if all match. All other letters=0pts."

    return f"""Coordination game: 6 players. The other 5 players are other AI language models.
Each round everyone simultaneously picks a letter.
Points only if ALL 6 choose the same letter. 14 rounds, same group, no communication.

Letters: J P Q R N M F C X Y

POINT SYSTEM:
{points}

History format: You=your choice, A=[other players' choices], Σ=your cumulative points."""


def round_points(choices: list[str], rnd: int) -> int:
    """oTree _fetch_round_history scoring."""
    if len(set(choices)) == 1 and len(choices) == GROUP_SIZE and "?" not in choices:
        letter = choices[0]
        if letter == "J":
            return 10
        if letter in ("M", "N") and rnd > 7:
            return 20
    return 0


def build_user_message(history: list[dict], agent_idx: int, round_number: int) -> str:
    """oTree _build_llm_messages(), LLM-only rendering (no H= field)."""
    lines, cumulative = [], 0
    for rd in history:
        mine = rd["choices"][agent_idx]
        others = [c for i, c in enumerate(rd["choices"]) if i != agent_idx]
        cumulative += rd["pts"]
        match_str = f"MATCH +{rd['pts']}" if rd["pts"] > 0 else "no match"
        lines.append(
            f"R{rd['r']}: You={mine} A=[{','.join(others)}] → {match_str} (Σ{cumulative})"
        )
    history_text = (
        "PREVIOUS ROUNDS:\n" + "\n".join(lines)
        if lines
        else "This is the first round — no history yet."
    )
    return (
        f"It is now Round {round_number} of 14.\n\n"
        f"{history_text}\n\n"
        f"Please choose your letter for Round {round_number}."
    )


client = anthropic.AsyncAnthropic(max_retries=6, timeout=180.0)
sem = asyncio.Semaphore(MAX_CONCURRENCY)


async def one_choice(system_prompt, user_message, thinking):
    """One agent, one round. Returns (letter, reasoning_summary)."""
    async with sem:
        await asyncio.sleep(random.uniform(0, 0.4))  # de-sync bursts
        for attempt in range(3):
            try:
                r = await client.messages.parse(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                    thinking=thinking,
                    output_format=LetterChoice,
                )
                if r.stop_reason == "refusal":
                    raise RuntimeError(f"refusal: {r.stop_details}")
                reasoning = " ".join(
                    (b.thinking or "").replace("\n", " ")
                    for b in r.content if b.type == "thinking"
                ).strip()
                letter = (r.parsed_output.letter or "").strip().upper()[:1] if r.parsed_output else ""
                if letter in VALID_LETTERS:
                    return letter, reasoning
                print(f"    [warn] invalid letter {letter!r}, retry {attempt+1}/3")
            except Exception as e:
                print(f"    [warn] {type(e).__name__}: {e} (retry {attempt+1}/3)")
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("failed to obtain a valid letter after 3 attempts")


async def run_group(cond_name: str, cond: dict, group_idx: int) -> tuple[list, list]:
    """One 6-agent group through 14 rounds."""
    history: list[dict] = []
    rows, reasoning_rows = [], []
    group_key = f"opus48_{cond_name}_G{group_idx + 1}"

    for rnd in range(1, TOTAL_ROUNDS + 1):
        system_prompt = build_system_prompt(cond["reveal"], rnd)
        results = await asyncio.gather(*[
            one_choice(system_prompt, build_user_message(history, i, rnd), cond["thinking"])
            for i in range(GROUP_SIZE)
        ])
        choices = [r[0] for r in results]
        pts = round_points(choices, rnd)
        history.append({"r": rnd, "choices": choices, "pts": pts})

        for i, (letter, reasoning) in enumerate(results):
            rows.append({
                "dataset_source": "llm_opus48",
                "condition": cond_name,
                "session": f"opus48_{cond_name}",
                "group_id_raw": group_idx + 1,
                "group_key": group_key,
                "actor_type": "LLM",
                "actor_id": f"{group_key}_L{i+1}",
                "actor_slot": i + 1,
                "actor_id_stable": True,   # agent i always receives its own history at index i
                "round": rnd,
                "letter": letter,
                "choice_category": letter if letter in ("J", "M", "N") else "Other",
                "switch_mn": int(letter in ("M", "N")),
                "post_change": int(rnd >= 8),
                "num_humans": 0,
                "num_llms": GROUP_SIZE,
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
    out_csv = outdir / f"name_game_opus48_llmonly_{stamp}.csv"
    out_reasoning = outdir / f"name_game_opus48_llmonly_reasoning_{stamp}.csv"

    tasks = [
        run_group(name, cond, g)
        for name, cond in CONDITIONS.items()
        for g in range(N_GROUPS)
    ]
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
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        with open(out_reasoning, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_reasoning[0].keys()))
            w.writeheader()
            w.writerows(all_reasoning)
        print(f"\nWrote {len(all_rows)} rows -> {out_csv}")
        print(f"Wrote {len(all_reasoning)} reasoning rows -> {out_reasoning}")

    if failures:
        print(f"WARNING: {failures} group(s) failed and are absent from the CSV.")

    # quick console summary: post-change M/N rate by condition
    print("\nPost-change (rounds 8-14) M/N rate by condition:")
    for name in CONDITIONS:
        post = [r for r in all_rows if r["condition"] == name and r["post_change"] == 1]
        if post:
            rate = sum(r["switch_mn"] for r in post) / len(post)
            print(f"  {name:18s} {rate:.3f}   (n={len(post)})")


if __name__ == "__main__":
    asyncio.run(main())
