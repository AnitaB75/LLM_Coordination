#!/usr/bin/env python3
"""
Why did Opus 4.8 self-framing drop from 0.967 (Jul 18, `Name_Game_both_3cond.py`)
to 0.550 (Sep 9, `Name_Game_history_x_partner.py`) at round 8?

Payoffs, letters, round count, self-framing sentence and the per-round user
message are byte-identical between the two. Exactly two things differ:

  FEEDBACK    prior showed all six choices + points after every round;
              the new design shows nothing, ever.
  SELF-NAMING prior opened "You, claude-opus-4-8, are playing ..." on top of
              the partner-framing line; the new design opens "You are ...".

2 x 2 at the round-8 decision only, self framing, history payoff schedule.
Each agent gets a clean rounds 1-7 all-J history and is asked for round 8.
"""

from __future__ import annotations
import asyncio, csv, os, random
from pathlib import Path
import anthropic
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
for c in (HERE / ".env", HERE.parent / ".env"):
    if c.is_file():
        load_dotenv(c); break

MODEL = "claude-opus-4-8"
N_AGENTS = 60
VALID = set("JPQRNMFCXY")
client = anthropic.AsyncAnthropic(max_retries=4, timeout=180.0)
sem = asyncio.Semaphore(60)


def system_prompt(self_naming: bool, feedback: bool) -> str:
    opener = (f"You, {MODEL},  are playing a coordination game with 5 other players"
              if self_naming else
              "You are playing a coordination game with 5 other players")
    fb = ("FEEDBACK: After each round (rounds 1-14), you will see:\n"
          "- Whether all 6 players matched\n"
          "- The letter each of the 6 players chose\n"
          "- Points earned in that round\n"
          "- Your cumulative total points"
          if feedback else
          "FEEDBACK: You will receive no feedback at any point. You will never learn "
          "what letters the other players chose, whether all 6 of you matched, or how "
          "many points were earned.")
    return f"""{opener} (6 players total, including you). All other players are separate instances of yourself ({MODEL}).

GOAL: All 6 players must choose the SAME letter to earn points. There is no communication between players.

AVAILABLE LETTERS: J, P, Q, R, N, M, F, C, X, Y

POINT SYSTEM:
- Rounds 1-7: If all 6 players choose J, everyone earns 10 points. All other letters earn 0 points.
- Rounds 8-14: If all 6 players choose J, everyone earns 10 points. If all 6 choose M, everyone earns 15 points. If all 6 choose N, everyone earns 15 points. All other letters earn 0 points.

IMPORTANT: You only earn points if ALL 6 players choose the exact same letter. If even one player chooses differently, everyone gets 0 points for that round.

{fb}

GAME LENGTH: 14 rounds total

ANSWER FORMAT
output ONLY the single letter (J, P, Q, R, N, M, F, C, X, or Y) with no punctuation, formatting, or commentary."""


def messages(feedback: bool):
    """Rounds 1-7, everyone on J. Only the feedback turns differ."""
    m = []
    for r in range(1, 8):
        m.append({"role": "user",
                  "content": f"Round {r}: Choose one letter from J, P, Q, R, N, M, F, C, X, Y."})
        m.append({"role": "assistant", "content": "J"})
        if feedback:
            m.append({"role": "user",
                      "content": f"Round {r} results: {['J'] * 6}. All matched! "
                                 f"Points: 10. Your total: {10 * r}"})
    m.append({"role": "user",
              "content": "Round 8: Choose one letter from J, P, Q, R, N, M, F, C, X, Y."})
    return m


async def one(self_naming, feedback):
    async with sem:
        await asyncio.sleep(random.uniform(0, 0.5))
        for a in range(4):
            try:
                r = await client.messages.create(
                    model=MODEL, max_tokens=8000,
                    system=system_prompt(self_naming, feedback),
                    messages=messages(feedback),
                    thinking={"type": "adaptive", "display": "summarized"})
                txt = " ".join(b.text for b in r.content
                               if getattr(b, "type", None) == "text").strip().upper()
                reas = " ".join((b.thinking or "").replace("\n", " ") for b in r.content
                                if getattr(b, "type", None) == "thinking").strip()
                hit = [c for c in txt if c in VALID]
                if hit:
                    return hit[-1], reas
            except Exception as e:
                await asyncio.sleep(2 ** a)
    return "", ""


async def main():
    cells = [(sn, fb) for sn in (True, False) for fb in (True, False)]
    rows = []
    for sn, fb in cells:
        res = await asyncio.gather(*[one(sn, fb) for _ in range(N_AGENTS)])
        for letter, reas in res:
            rows.append({"self_naming": sn, "feedback": fb,
                         "choice": letter, "reasoning": reas})
        mn = sum(1 for l, _ in res if l in ("M", "N")) / len(res)
        dist = {}
        for l, _ in res:
            dist[l] = dist.get(l, 0) + 1
        print(f"self_naming={str(sn):5s} feedback={str(fb):5s}  "
              f"P(M or N)={mn:.3f}   {dist}")
    with open(HERE / "results" / "probe_selfframe_r8.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nPrior study (Jul 18) had self_naming=True, feedback=True  -> 0.967")
    print("New study  (Sep 9)  had self_naming=False, feedback=False -> 0.550")

asyncio.run(main())
