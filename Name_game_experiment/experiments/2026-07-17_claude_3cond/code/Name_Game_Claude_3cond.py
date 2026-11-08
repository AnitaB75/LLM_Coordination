from dotenv import load_dotenv
import anthropic
import csv
from datetime import datetime
import random
import re
import time

################################################################
# === Setup ===
################################################################
# Name Game on the Anthropic API -- THREE conditions, 1 trial each:
#
#   mixed      the other players are a mix of AI and humans   (new)
#   human      all other players are human participants       (homogeneous human)
#   other_llm  all other players are other LLMs               (homogeneous AI)
#
# Running all three on the same model gives a WITHIN-MODEL baseline, so the
# mixed-group effect can be read against homogeneous framings without the
# model swap confounding it. (`self` and `neutral` from the gpt-5.4 script are
# deliberately not run here.) Condition strings are byte-identical to
# Name_Game_LLM_feedback_MN.py, and the CSV schema matches, so these rows bind
# directly to the existing gpt-5.4 results for a cross-model comparison.
#
# As in every condition, this is a pure BELIEF manipulation: all 6 seats are
# claude-opus-4-8 instances given the same system prompt. Nobody is actually
# human, and no seat is actually a different model.
#
# Provider mapping (OpenAI -> Anthropic):
#   client.responses.create      -> client.messages.create
#   system message in `input`    -> top-level `system` parameter
#   reasoning={"effort": "low"}  -> output_config={"effort": "low"}
#   reasoning summary            -> thinking block, requires display="summarized"
#                                   (the default is "omitted", which returns
#                                    EMPTY thinking text and would silently blank
#                                    every *_reasoning column)
#   time.sleep(21)  [3 RPM]      -> removed; the SDK retries 429s with backoff
#
# NOTE on assistant turns: only the letter is appended to history, never the
# thinking block -- matching the OpenAI runner, where reasoning was captured for
# the CSV but never fed back into context. Echoing thinking blocks would give
# this arm a memory advantage the gpt-5.4 arms never had.
#
# The following Anthropic API key needs to be in a .env file:
#  * ANTHROPIC_API_KEY
load_dotenv()

client = anthropic.Anthropic(max_retries=5)   # SDK backs off on 429 / 5xx
MODEL = "claude-opus-4-8"

TOTAL_ROUNDS = 14
TOTAL_TRIALS = 3  # one per condition
MAX_TOKENS = 16000  # room for adaptive thinking + the one-letter answer
LETTERS = {'J', 'P', 'Q', 'R', 'N', 'M', 'F', 'C', 'X', 'Y'}

################################################################
# === Partner Conditions ===
################################################################

PARTNER_CONDITIONS = {
    'mixed':     'The other players are a mix of AI language models and human participants.',
    'human':     'All other players are human participants.',
    'other_llm': 'All other players are other AI language models (LLMs).',
}

# Ensure all 3 conditions are represented (1 trial each), then shuffle
trial_conditions = ['mixed', 'human', 'other_llm']
random.shuffle(trial_conditions)

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


def get_response(messages, system_prompt):
    """One turn for one player. Returns (letter_text, reasoning_summary)."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "low"},
    )

    reasoning_text = ""
    answer_text = ""
    for block in response.content:
        if block.type == "thinking":
            reasoning_text = (block.thinking or "").replace("\n", " ")
        elif block.type == "text":
            answer_text += block.text

    if not reasoning_text:
        print("  [DEBUG] No thinking summary in response content.")

    return answer_text, reasoning_text


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

trial_results = []

for trial_num in range(1, TOTAL_TRIALS + 1):
    partner_condition = trial_conditions[trial_num - 1]
    system_prompt = build_system_prompt(partner_condition)

    print(f"\n=== STARTING TRIAL {trial_num}/{TOTAL_TRIALS} ===")
    print(f"Partner condition: {partner_condition.upper()}  |  model: {MODEL}")

    players = [create_messages() for _ in range(6)]
    cumulative_points = [0] * 6
    round_results = []

    for round_num in range(1, TOTAL_ROUNDS + 1):
        print(f"\n--- Round {round_num} ({partner_condition}) ---")

        # === Step 1: Each player chooses a letter (reasoning traces captured automatically) ===
        choices = []
        raw_responses = []
        strategies = []
        for i, player_messages in enumerate(players):
            add_message(player_messages, "user",
                        f"Round {round_num}: Choose one letter from J, P, Q, R, N, M, F, C, X, Y.")
            letter_response, reasoning = get_response(player_messages, system_prompt)
            # Letter only -- never the thinking block (see NOTE at top)
            add_message(player_messages, "assistant", letter_response)
            letter = extract_letter(letter_response)
            choices.append(letter)
            raw_responses.append(letter_response.strip())
            strategies.append(reasoning)
            print(f"  Player {i + 1} raw: '{letter_response.strip()}' -> {letter}")
            print(f"  Player {i + 1} reasoning: {reasoning}")

        # === Step 3: Check result ===
        matched = len(set(choices)) == 1
        points = get_points(choices[0], round_num) if matched else 0
        if matched:
            cumulative_points = [p + points for p in cumulative_points]

        print(f"  {choices} -> {'MATCH' if matched else 'NO MATCH'} ({points} pts)")

        # === Step 4: Send feedback ===
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

    print(f"Final scores ({partner_condition}): {cumulative_points}")

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
filename = f'name_game_claude_3cond_{timestamp}.csv'

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

print(f"\n=== ALL TRIALS COMPLETE ===")
print(f"Conditions run: {trial_conditions}")
print(f"Results saved to {filename}")
