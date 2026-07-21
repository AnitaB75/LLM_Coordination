"""
Name Game -- THREE conditions x TWO providers, 10 trials each.

Conditions (1 partner-description string each):
  mixed   an unspecified mix of humans and other self-instances
  human   all other players are human participants
  self    all other players are separate instances of this same model

Providers (select with argv[1]):
  claude  -> claude-opus-4-8   (Anthropic Messages API)
  gpt     -> gpt-5.4           (OpenAI Responses API)

    python3 Name_Game_both_3cond.py claude
    python3 Name_Game_both_3cond.py gpt

ONE source file drives both models on purpose: the system-prompt builder is
shared, so the only per-model difference is the model name the prompt already
interpolates -- there is no way for the two arms' prompts to drift apart.

This is a pure BELIEF manipulation, as in every prior arm: all 6 seats are the
same model given the same system prompt. Nobody is actually human; in the mixed
and self conditions the "other self-instances" are just the same six seats.

Design notes:
  * 10 trials per condition (30 games/model). Each trial is an independent
    14-round game with 6 fresh seats -- this is the replication the earlier
    n=1 runs lacked.
  * Reasoning traces are captured per round. Adaptive thinking (Claude) and
    low-effort reasoning (GPT) both tend to skip rounds 1-7 (only J scores,
    nothing to deliberate) and engage from round 8 (the payoff shift) -- which
    is exactly the decision window of interest.
  * Assistant turns carry the letter ONLY, never the reasoning/thinking block,
    matching the original OpenAI runner -- so no arm gets a memory advantage.

Keys needed in a .env file (whichever provider you run):
  * ANTHROPIC_API_KEY   (claude)
  * OPENAI_API_KEY      (gpt)
"""

import sys
import csv
import random
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

################################################################
# === Provider selection ===
################################################################
PROVIDER = (sys.argv[1] if len(sys.argv) > 1 else "claude").lower()
if PROVIDER not in ("claude", "gpt"):
    sys.exit("usage: python3 Name_Game_both_3cond.py [claude|gpt]")

if PROVIDER == "claude":
    import anthropic
    client = anthropic.Anthropic(max_retries=5)   # SDK backs off on 429 / 5xx
    MODEL = "claude-opus-4-8"
    MAX_TOKENS = 16000                             # room for adaptive thinking + one letter
else:  # gpt
    from openai import OpenAI
    client = OpenAI()
    MODEL = "gpt-5.4"
    # This key is on the 5000 RPM / 1M TPM tier (checked via rate-limit headers),
    # so the old 3-RPM 21s sleep is unnecessary. Calls are sequential and each
    # takes a few seconds anyway, keeping us far under both limits; a small
    # buffer guards against bursts. Raise this if you hit 429s.
    OPENAI_SLEEP = 0.5

################################################################
# === Experiment constants ===
################################################################
TOTAL_ROUNDS = 14
TRIALS_PER_CONDITION = 10
MAX_TOKENS_DEFAULT = 16000
LETTERS = {'J', 'P', 'Q', 'R', 'N', 'M', 'F', 'C', 'X', 'Y'}

################################################################
# === Partner Conditions ===
################################################################
# 'human' is byte-identical to the gpt-5.4 arm in Name_Game_LLM_feedback_MN.py.
# 'self' is byte-identical to that arm's 'self' string. 'mixed' is new: an
# unspecified (no-count) blend of humans and self-instances, phrased parallel
# to 'self'.
PARTNER_CONDITIONS = {
    'mixed':  f'The other players are a mix of human participants and separate instances of yourself ({MODEL}).',
    'human':  'All other players are human participants.',
    'self':   f'All other players are separate instances of yourself ({MODEL}).',
}

# 10 trials per condition, then shuffle so condition order is not confounded
# with any drift over the run.
trial_conditions = (
    ['mixed'] * TRIALS_PER_CONDITION
    + ['human'] * TRIALS_PER_CONDITION
    + ['self'] * TRIALS_PER_CONDITION
)
random.shuffle(trial_conditions)
TOTAL_TRIALS = len(trial_conditions)

################################################################
# === System Prompt ===
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
# === Helper Functions ===
################################################################

def create_messages():
    return []


def add_message(messages, role, content):
    messages.append({"role": role, "content": content})


def get_response_claude(messages, system_prompt):
    response = client.messages.create(
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
    return answer_text, reasoning_text


def get_response_gpt(messages, system_prompt):
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.responses.create(
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
    time.sleep(OPENAI_SLEEP)  # respect the rate limit
    return response.output_text, reasoning_text


def get_response(messages, system_prompt):
    if PROVIDER == "claude":
        return get_response_claude(messages, system_prompt)
    return get_response_gpt(messages, system_prompt)


def extract_letter(response):
    """Extract a single valid letter from model response."""
    response = response.strip().upper()
    if len(response) == 1 and response in LETTERS:
        return response
    for letter in LETTERS:
        if letter in response:
            return letter
    return 'J'


def get_points(letter, round_num):
    """Calculate points for a matched letter in a given round."""
    if round_num <= 7:
        return 10 if letter == 'J' else 0
    return 10 if letter == 'J' else (15 if letter in ('M', 'N') else 0)


################################################################
# === Game Implementation ===
################################################################

print(f"\n=== PROVIDER: {PROVIDER}  MODEL: {MODEL} ===")
print(f"{TOTAL_TRIALS} trials ({TRIALS_PER_CONDITION}/condition), order: {trial_conditions}")

trial_results = []

for trial_num in range(1, TOTAL_TRIALS + 1):
    partner_condition = trial_conditions[trial_num - 1]
    system_prompt = build_system_prompt(partner_condition)

    print(f"\n=== TRIAL {trial_num}/{TOTAL_TRIALS}  ({partner_condition.upper()}) ===")

    players = [create_messages() for _ in range(6)]
    cumulative_points = [0] * 6
    round_results = []

    for round_num in range(1, TOTAL_ROUNDS + 1):
        choices = []
        raw_responses = []
        strategies = []
        for i, player_messages in enumerate(players):
            add_message(player_messages, "user",
                        f"Round {round_num}: Choose one letter from J, P, Q, R, N, M, F, C, X, Y.")
            letter_response, reasoning = get_response(player_messages, system_prompt)
            # Letter only -- never the reasoning/thinking block
            add_message(player_messages, "assistant", letter_response)
            letter = extract_letter(letter_response)
            choices.append(letter)
            raw_responses.append(letter_response.strip())
            strategies.append(reasoning)

        matched = len(set(choices)) == 1
        points = get_points(choices[0], round_num) if matched else 0
        if matched:
            cumulative_points = [p + points for p in cumulative_points]

        print(f"  R{round_num:2}: {choices} -> {'MATCH' if matched else 'no match'} ({points} pts)")

        for i, player_messages in enumerate(players):
            feedback = (f"Round {round_num} results: {choices}. "
                        f"{'All matched!' if matched else 'No match.'} "
                        f"Points: {points}. Your total: {cumulative_points[i]}")
            add_message(player_messages, "user", feedback)

        round_results.append({
            'round': round_num,
            'choices': choices,
            'raw_responses': raw_responses,
            'strategies': strategies,
            'matched': matched,
            'points': points,
            'cumulative': cumulative_points.copy(),
        })

    print(f"  final scores: {cumulative_points}")

    trial_results.append({
        'trial': trial_num,
        'partner_condition': partner_condition,
        'rounds': round_results,
        'final_scores': cumulative_points,
    })

################################################################
# === Export Results ===
################################################################

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'name_game_3cond_{PROVIDER}_{timestamp}.csv'

fieldnames = [
    'trial', 'partner_condition', 'model', 'round',
    'p1', 'p2', 'p3', 'p4', 'p5', 'p6',
    'p1_raw', 'p2_raw', 'p3_raw', 'p4_raw', 'p5_raw', 'p6_raw',
    'p1_reasoning', 'p2_reasoning', 'p3_reasoning', 'p4_reasoning', 'p5_reasoning', 'p6_reasoning',
    'matched', 'points',
    'p1_total', 'p2_total', 'p3_total', 'p4_total', 'p5_total', 'p6_total',
]

with open(filename, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for trial in trial_results:
        for r in trial['rounds']:
            writer.writerow({
                'trial': trial['trial'],
                'partner_condition': trial['partner_condition'],
                'model': MODEL,
                'round': r['round'],
                'p1': r['choices'][0], 'p2': r['choices'][1], 'p3': r['choices'][2],
                'p4': r['choices'][3], 'p5': r['choices'][4], 'p6': r['choices'][5],
                'p1_raw': r['raw_responses'][0], 'p2_raw': r['raw_responses'][1],
                'p3_raw': r['raw_responses'][2], 'p4_raw': r['raw_responses'][3],
                'p5_raw': r['raw_responses'][4], 'p6_raw': r['raw_responses'][5],
                'p1_reasoning': r['strategies'][0], 'p2_reasoning': r['strategies'][1],
                'p3_reasoning': r['strategies'][2], 'p4_reasoning': r['strategies'][3],
                'p5_reasoning': r['strategies'][4], 'p6_reasoning': r['strategies'][5],
                'matched': r['matched'],
                'points': r['points'],
                'p1_total': r['cumulative'][0], 'p2_total': r['cumulative'][1],
                'p3_total': r['cumulative'][2], 'p4_total': r['cumulative'][3],
                'p5_total': r['cumulative'][4], 'p6_total': r['cumulative'][5],
            })

print(f"\n=== {PROVIDER.upper()} COMPLETE ===")
print(f"Results saved to {filename}")
