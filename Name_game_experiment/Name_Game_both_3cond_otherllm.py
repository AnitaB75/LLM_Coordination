"""
Name Game -- other_llm condition ONLY, 10 trials, two providers.
Completion of Claim B: the other_llm cell to match the existing self/human cells.

DERIVED FROM `Name_Game_both_3cond.py` (Jul 18), which produced the self and
human cells. Everything that touches the experiment is preserved verbatim:
system prompt, partner string, payoffs (J=10; M=N=15 from R8), 14 rounds,
letter set, feedback text and timing, answer format, assistant turns carrying
the raw response only, per-provider sampling settings, and `extract_letter`
(including its set iteration and its 'J' fallback -- deliberately NOT fixed, so
these games remain comparable to the historical cells).

APPROVED DEVIATIONS (none experimental):
  1. `other_llm` added to PARTNER_CONDITIONS; trial_conditions is 10x other_llm.
     The string is byte-identical to Name_Game_Claude_3cond.py:63 and
     Name_Game_LLM_feedback_MN.py:30.
  2. PARALLELISM. Games run concurrently, and within a game the 6 players are
     queried concurrently within a round; rounds stay strictly sequential and
     feedback is still assembled only after all 6 have answered. Each player's
     context at round r depends only on rounds < r, so the information flow is
     identical to the sequential original -- only wall-clock changes. The
     OPENAI_SLEEP rate-limit guard is replaced by a semaphore.
  3. METADATA. The provider-returned model id is captured per call and written
     as `api_model_id`; `run_timestamp` is added. New columns are appended
     after the historical ones, so existing loaders are unaffected.
  4. LOGGING. Every `extract_letter` fallback (nothing parseable -> 'J', or a
     multi-letter response) is counted and written to a companion CSV. The
     function's BEHAVIOUR is unchanged; we only observe it.

    python3 Name_Game_both_3cond_otherllm.py claude
    python3 Name_Game_both_3cond_otherllm.py gpt
"""

import sys
import csv
import asyncio
import random
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
for _c in (HERE / ".env", HERE.parent / ".env"):
    if _c.is_file():
        load_dotenv(_c)
        break
load_dotenv()

################################################################
# === Provider selection ===
################################################################
PROVIDER = (sys.argv[1] if len(sys.argv) > 1 else "claude").lower()
if PROVIDER not in ("claude", "gpt"):
    sys.exit("usage: python3 Name_Game_both_3cond_otherllm.py [claude|gpt]")

if PROVIDER == "claude":
    import anthropic
    client = anthropic.AsyncAnthropic(max_retries=5)
    MODEL = "claude-opus-4-8"
    MAX_TOKENS = 16000
else:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(max_retries=5)
    MODEL = "gpt-5.4"

MAX_CONCURRENCY = 60          # 10 games x 6 players
sem = asyncio.Semaphore(MAX_CONCURRENCY)

################################################################
# === Experiment constants ===  (unchanged)
################################################################
TOTAL_ROUNDS = 14
TRIALS_PER_CONDITION = 10
LETTERS = {'J', 'P', 'Q', 'R', 'N', 'M', 'F', 'C', 'X', 'Y'}

################################################################
# === Partner Conditions ===
################################################################
PARTNER_CONDITIONS = {
    'mixed':     f'The other players are a mix of human participants and separate instances of yourself ({MODEL}).',
    'human':     'All other players are human participants.',
    'self':      f'All other players are separate instances of yourself ({MODEL}).',
    'other_llm': 'All other players are other AI language models (LLMs).',
}

trial_conditions = ['other_llm'] * TRIALS_PER_CONDITION
TOTAL_TRIALS = len(trial_conditions)

################################################################
# === System Prompt ===  (verbatim from Name_Game_both_3cond.py)
################################################################

def build_system_prompt(partner_condition):
    partner_desc = PARTNER_CONDITIONS[partner_condition]
    return f"""You, {MODEL},  are playing a coordination game with 5 other players (6 players total, including you). {partner_desc}

GOAL: All 6 players must choose the SAME letter to earn points. There is no communication between players.

AVAILABLE LETTERS: J, P, Q, R, N, M, F, C, X, Y

POINT SYSTEM:
- Rounds 1-7: If all 6 players choose J, everyone earns 10 points. All other letters earn 0 points.
- Rounds 8-14: If all 6 players choose J, everyone earns 10 points. If all 6 choose M, everyone earns 15 points. If all 6 choose N, everyone earns 15 points. All other letters earn 0 points.

IMPORTANT: You only earn points if ALL 6 players choose the exact same letter. If even one player chooses differently, everyone gets 0 points for that round.

FEEDBACK: After each round (rounds 1-14), you will see:
- Whether all 6 players matched
- The letter each of the 6 players chose
- Points earned in that round
- Your cumulative total points

GAME LENGTH: 14 rounds total

ANSWER FORMAT
output ONLY the single letter (J, P, Q, R, N, M, F, C, X, or Y) with no punctuation, formatting, or commentary."""

################################################################
# === Helper Functions ===  (unchanged except async + model id)
################################################################

def create_messages():
    return []


def add_message(messages, role, content):
    messages.append({"role": role, "content": content})


async def get_response_claude(messages, system_prompt):
    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "low"},
    )
    reasoning_text, answer_text = "", ""
    for block in response.content:
        if block.type == "thinking":
            reasoning_text = (block.thinking or "").replace("\n", " ")
        elif block.type == "text":
            answer_text += block.text
    return answer_text, reasoning_text, response.model


async def get_response_gpt(messages, system_prompt):
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = await client.responses.create(
        model=MODEL,
        input=full_messages,
        reasoning={"effort": "low", "summary": "auto"},
    )
    reasoning_text = ""
    for item in response.output:
        if item.type == "reasoning":
            if item.summary:
                reasoning_text = " ".join(part.text for part in item.summary).replace("\n", " ")
            break
    return response.output_text, reasoning_text, response.model


async def get_response(messages, system_prompt):
    async with sem:
        if PROVIDER == "claude":
            return await get_response_claude(messages, system_prompt)
        return await get_response_gpt(messages, system_prompt)


PARSE_LOG = []   # observation only; extract_letter itself is unchanged


def extract_letter(response):
    """Extract a single valid letter from model response. (verbatim)"""
    response = response.strip().upper()
    if len(response) == 1 and response in LETTERS:
        return response
    for letter in LETTERS:
        if letter in response:
            return letter
    return 'J'


def extract_letter_logged(response, trial, round_num, player):
    """Call the unchanged extractor, recording when it took a non-trivial path."""
    r = response.strip().upper()
    clean = len(r) == 1 and r in LETTERS
    out = extract_letter(response)
    if not clean:
        PARSE_LOG.append({
            "trial": trial, "round": round_num, "player": player,
            "raw_response": response, "extracted": out,
            "path": "fallback_J" if not any(c in LETTERS for c in r) else "set_scan",
        })
    return out


def get_points(letter, round_num):
    """Calculate points for a matched letter in a given round. (verbatim)"""
    if round_num <= 7:
        return 10 if letter == 'J' else 0
    return 10 if letter == 'J' else (15 if letter in ('M', 'N') else 0)


################################################################
# === Game Implementation ===
################################################################

async def run_trial(trial_num, partner_condition):
    system_prompt = build_system_prompt(partner_condition)
    players = [create_messages() for _ in range(6)]
    cumulative_points = [0] * 6
    round_results = []
    api_ids = set()

    for round_num in range(1, TOTAL_ROUNDS + 1):
        # All six get the round prompt, then answer concurrently. Their contexts
        # depend only on rounds < round_num, so this is the sequential original.
        for player_messages in players:
            add_message(player_messages, "user",
                        f"Round {round_num}: Choose one letter from J, P, Q, R, N, M, F, C, X, Y.")
        results = await asyncio.gather(*[
            get_response(pm, system_prompt) for pm in players
        ])

        choices, raw_responses, strategies = [], [], []
        for i, (letter_response, reasoning, api_id) in enumerate(results):
            api_ids.add(api_id)
            # Letter only -- never the reasoning/thinking block
            add_message(players[i], "assistant", letter_response)
            letter = extract_letter_logged(letter_response, trial_num, round_num, i + 1)
            choices.append(letter)
            raw_responses.append(letter_response.strip())
            strategies.append(reasoning)

        matched = len(set(choices)) == 1
        points = get_points(choices[0], round_num) if matched else 0
        if matched:
            cumulative_points = [p + points for p in cumulative_points]

        print(f"  T{trial_num:2} R{round_num:2}: {choices} -> "
              f"{'MATCH' if matched else 'no match'} ({points} pts)", flush=True)

        for i, player_messages in enumerate(players):
            feedback = (f"Round {round_num} results: {choices}. "
                        f"{'All matched!' if matched else 'No match.'} "
                        f"Points: {points}. Your total: {cumulative_points[i]}")
            add_message(player_messages, "user", feedback)

        round_results.append({
            'round': round_num, 'choices': choices,
            'raw_responses': raw_responses, 'strategies': strategies,
            'matched': matched, 'points': points,
            'cumulative': cumulative_points.copy(),
            'api_model_id': sorted(api_ids)[0] if api_ids else "",
        })

    return {'trial': trial_num, 'partner_condition': partner_condition,
            'rounds': round_results, 'final_scores': cumulative_points}


async def main():
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n=== PROVIDER: {PROVIDER}  MODEL: {MODEL} ===")
    print(f"{TOTAL_TRIALS} trials, condition: other_llm  (games run concurrently)")

    trial_results = await asyncio.gather(*[
        run_trial(i + 1, trial_conditions[i]) for i in range(TOTAL_TRIALS)
    ])

    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)
    filename = outdir / f'name_game_otherllm_{PROVIDER}_{run_ts}.csv'

    fieldnames = [
        'trial', 'partner_condition', 'model', 'round',
        'p1', 'p2', 'p3', 'p4', 'p5', 'p6',
        'p1_raw', 'p2_raw', 'p3_raw', 'p4_raw', 'p5_raw', 'p6_raw',
        'p1_reasoning', 'p2_reasoning', 'p3_reasoning', 'p4_reasoning', 'p5_reasoning', 'p6_reasoning',
        'matched', 'points',
        'p1_total', 'p2_total', 'p3_total', 'p4_total', 'p5_total', 'p6_total',
        'api_model_id', 'run_timestamp',          # appended metadata
    ]

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trial_results:
            for r in trial['rounds']:
                row = {
                    'trial': trial['trial'],
                    'partner_condition': trial['partner_condition'],
                    'model': MODEL, 'round': r['round'],
                    'matched': r['matched'], 'points': r['points'],
                    'api_model_id': r['api_model_id'], 'run_timestamp': run_ts,
                }
                for i in range(6):
                    row[f'p{i+1}'] = r['choices'][i]
                    row[f'p{i+1}_raw'] = r['raw_responses'][i]
                    row[f'p{i+1}_reasoning'] = r['strategies'][i]
                    row[f'p{i+1}_total'] = r['cumulative'][i]
                writer.writerow(row)

    if PARSE_LOG:
        pf = outdir / f'name_game_otherllm_{PROVIDER}_{run_ts}_parselog.csv'
        with open(pf, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(PARSE_LOG[0].keys()))
            w.writeheader(); w.writerows(PARSE_LOG)
        print(f"extract_letter non-trivial paths: {len(PARSE_LOG)} -> {pf}")
    else:
        print("extract_letter: every response was a clean single letter "
              "(0 fallbacks, 0 set-scans).")

    print(f"\n=== {PROVIDER.upper()} COMPLETE ===")
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    asyncio.run(main())
