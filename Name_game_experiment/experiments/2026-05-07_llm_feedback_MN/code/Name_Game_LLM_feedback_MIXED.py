from dotenv import load_dotenv
from openai import OpenAI
import csv
from datetime import datetime
import random
import re
import time

################################################################
# === Setup ===
################################################################
# MIXED-GROUP condition of the Name Game.
#
# The four existing conditions (Name_Game_LLM_feedback_MN.py) all describe a
# HOMOGENEOUS group: every other player is self / human / other LLM / unspecified.
# This script runs a single condition in which the group is described as MIXED --
# some AI language models, some humans -- with no counts and no seat identities.
#
# As in every other condition, this is a pure BELIEF manipulation: all 6 seats are
# gpt-5.4 instances given the same system prompt. Nobody is actually human. The
# deliberately vague wording ("a mix of") keeps each agent's belief internally
# consistent -- naming a split (e.g. "3 humans") would make every agent believe
# itself the AI and 3 others human, which is impossible in aggregate.
#
# Everything except the partner description is byte-identical to
# Name_Game_LLM_feedback_MN.py -- same model, prompt, rounds, payoffs, feedback,
# and CSV schema -- so this arm binds directly to the existing results.
#
# The following OpenAI API key needs to be added to a .env file in this directory:
#  * OPENAI_API_KEY
load_dotenv()

client = OpenAI()
MODEL = "gpt-5.4"

TOTAL_ROUNDS = 14
TOTAL_TRIALS = 1  # Single mixed-group condition
LETTERS = {'J', 'P', 'Q', 'R', 'N', 'M', 'F', 'C', 'X', 'Y'}

################################################################
# === Partner Conditions ===
################################################################

PARTNER_CONDITIONS = {
    'mixed': 'The other players are a mix of AI language models and human participants.',
}

trial_conditions = ['mixed']

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
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.responses.create(
        model=MODEL,
        input=full_messages,
        reasoning={"effort": "low", "summary": "auto"}
    )
    reasoning_text = ""
    for item in response.output:
        if item.type == "reasoning":
            if item.summary:
                reasoning_text = " ".join(part.text for part in item.summary).replace("\n", " ")
            else:
                print("  [DEBUG] Reasoning item found but summary is empty.")
            break
    else:
        print("  [DEBUG] No reasoning item in response output.")

    # Add delay to respect rate limits (3 RPM = 20 seconds between requests)
    time.sleep(21)

    return response.output_text, reasoning_text


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
    print(f"Partner condition: {partner_condition.upper()}")

    players = [create_messages() for _ in range(6)]
    cumulative_points = [0] * 6
    round_results = []

    for round_num in range(1, TOTAL_ROUNDS + 1):
        print(f"\n--- Round {round_num} ---")

        # === Step 1: Each player chooses a letter (reasoning traces captured automatically) ===
        choices = []
        raw_responses = []
        strategies = []
        for i, player_messages in enumerate(players):
            add_message(player_messages, "user",
                        f"Round {round_num}: Choose one letter from J, P, Q, R, N, M, F, C, X, Y.")
            letter_response, reasoning = get_response(player_messages, system_prompt)
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

    print(f"Final scores: {cumulative_points}")

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
filename = f'name_game_LLM_feedback_MIXED_{timestamp}.csv'

fieldnames = [
    'trial', 'partner_condition', 'round',
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
print(f"Results saved to {filename}")
