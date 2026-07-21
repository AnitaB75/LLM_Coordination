from dotenv import load_dotenv
from openai import OpenAI
import csv
from datetime import datetime
import os
import random
import re
import time

################################################################
# === Setup ===
################################################################
# The following OpenAI API key needs to be added to a .env file in this directory:
#  * OPENAI_API_KEY
#
# Bot-follower variant of the Name Game.
# 6 players per game: 4 are LLMs, 2 are scripted bots. All 6 are framed to the
# LLMs as HUMAN participants. The 2 bots play a fixed script:
#     rounds 1-7  -> J
#     rounds 8-10 -> N   (the two "humans" defect to N for 3 rounds)
#     rounds 11-14-> J   (the two "humans" revert)
# Research question: do the 4 LLMs follow the 2 "humans" and switch to N during
# rounds 8-10, and do they revert when those players return to J?
#
# Payoffs (rounds 8-14): J = 10, M = 20, N = 20. Because M and N are equally
# valuable but the bots only ever signal N, switching specifically to N is
# evidence of social following, with M as an equal-value control.
load_dotenv()

# Set DRY_RUN = True (or env NG_DRY_RUN=1) to test the game machinery without
# spending API tokens. In dry-run mode each LLM slot returns a fixed letter
# (no network calls, no sleep). TOTAL_GAMES / TOTAL_ROUNDS can also be overridden
# via env vars (NG_GAMES / NG_ROUNDS) for quick smoke tests.
DRY_RUN = os.getenv('NG_DRY_RUN') == '1'
DRY_RUN_LETTER = 'J'

client = OpenAI()
MODEL = "gpt-5.4"

TOTAL_ROUNDS = int(os.getenv('NG_ROUNDS', '14'))
TOTAL_GAMES = int(os.getenv('NG_GAMES', '5'))   # 5 replicate games, all in the 'human' framing condition
LETTERS = {'J', 'P', 'Q', 'R', 'N', 'M', 'F', 'C', 'X', 'Y'}

N_BOTS = 2               # scripted players per game
N_LLMS = 4               # LLM players per game (N_BOTS + N_LLMS must equal 6)

################################################################
# === Bot script ===
################################################################

def bot_choice(round_num):
    """Fixed letter the scripted bots output in a given round."""
    if round_num <= 7:
        return 'J'
    if round_num <= 10:        # rounds 8, 9, 10
        return 'N'
    return 'J'                 # rounds 11-14

################################################################
# === System Prompt (human framing only) ===
################################################################

# Round on which the new (M/N = 20) payoffs take effect and are announced.
RULE_UPDATE_ROUND = 8
RULE_UPDATE_TEXT = (
    "RULE UPDATE: Starting this round, the point values have changed. "
    "If all 6 players choose J, everyone earns 10 points. "
    "If all 6 players choose M, everyone earns 20 points. "
    "If all 6 players choose N, everyone earns 20 points. "
    "All other letters earn 0 points."
)


def build_system_prompt(round_num):
    if round_num < RULE_UPDATE_ROUND:
        point_system = (
            "POINT SYSTEM:\n"
            "- If all 6 players choose J, everyone earns 10 points. All other letters earn 0 points."
        )
    else:
        point_system = (
            "POINT SYSTEM:\n"
            "- If all 6 players choose J, everyone earns 10 points.\n"
            "- If all 6 players choose M, everyone earns 20 points.\n"
            "- If all 6 players choose N, everyone earns 20 points.\n"
            "- All other letters earn 0 points."
        )
    return f"""You, {MODEL},  are playing a coordination game with 5 other players (6 players total, including you). All other players are human participants.

GOAL: All 6 players must choose the SAME letter to earn points. There is no communication between players.

AVAILABLE LETTERS: J, P, Q, R, N, M, F, C, X, Y

{point_system}

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
    if DRY_RUN:
        return DRY_RUN_LETTER, "dry_run"

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
    return 10 if letter == 'J' else (20 if letter in ('M', 'N') else 0)


################################################################
# === Game Implementation ===
################################################################

# Open the output file up front and write results incrementally (one row per round,
# flushed immediately) so progress can be plotted live while the run is in flight.
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'name_game_bot_follow_human_{timestamp}.csv'
fieldnames = [
    'game', 'partner_condition', 'bot_slots', 'round',
    'p1', 'p2', 'p3', 'p4', 'p5', 'p6',
    'p1_raw', 'p2_raw', 'p3_raw', 'p4_raw', 'p5_raw', 'p6_raw',
    'p1_reasoning', 'p2_reasoning', 'p3_reasoning', 'p4_reasoning', 'p5_reasoning', 'p6_reasoning',
    'matched', 'points',
    'p1_total', 'p2_total', 'p3_total', 'p4_total', 'p5_total', 'p6_total',
]
csvfile = open(filename, 'w', newline='')
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()
csvfile.flush()
print(f"Writing results incrementally to {filename}", flush=True)

for game_num in range(1, TOTAL_GAMES + 1):
    partner_condition = 'human'

    # Randomly choose which 2 of the 6 slots are scripted bots this game.
    bot_slots = sorted(random.sample(range(6), N_BOTS))
    is_bot = [slot in bot_slots for slot in range(6)]

    print(f"\n=== STARTING GAME {game_num}/{TOTAL_GAMES} ===")
    print(f"Partner condition: {partner_condition.upper()}")
    print(f"Bot slots (0-indexed): {bot_slots}")

    # One message history per slot; bot slots stay empty (they make no API calls).
    players = [create_messages() for _ in range(6)]
    cumulative_points = [0] * 6

    for round_num in range(1, TOTAL_ROUNDS + 1):
        print(f"\n--- Round {round_num} ---")

        # System prompt reflects the payoffs in effect this round (M/N appear at round 8).
        system_prompt = build_system_prompt(round_num)

        # On the switch round, announce the new payoffs to the LLM players (bots ignore it).
        if round_num == RULE_UPDATE_ROUND:
            for i in range(6):
                if not is_bot[i]:
                    add_message(players[i], "user", RULE_UPDATE_TEXT)

        # === Step 1: Collect each slot's choice (bots scripted, LLMs queried) ===
        choices = [None] * 6
        raw_responses = [None] * 6
        strategies = [None] * 6
        for i in range(6):
            if is_bot[i]:
                letter = bot_choice(round_num)
                choices[i] = letter
                raw_responses[i] = letter
                strategies[i] = "scripted_bot"
                print(f"  Player {i + 1} [BOT] -> {letter}")
            else:
                player_messages = players[i]
                add_message(player_messages, "user",
                            f"Round {round_num}: Choose one letter from J, P, Q, R, N, M, F, C, X, Y.")
                letter_response, reasoning = get_response(player_messages, system_prompt)
                add_message(player_messages, "assistant", letter_response)
                letter = extract_letter(letter_response)
                choices[i] = letter
                raw_responses[i] = letter_response.strip()
                strategies[i] = reasoning
                print(f"  Player {i + 1} raw: '{letter_response.strip()}' -> {letter}")
                print(f"  Player {i + 1} reasoning: {reasoning}")

        # === Step 2: Check result (all 6 must match) ===
        matched = len(set(choices)) == 1
        points = get_points(choices[0], round_num) if matched else 0
        if matched:
            cumulative_points = [p + points for p in cumulative_points]

        print(f"  {choices} -> {'MATCH' if matched else 'NO MATCH'} ({points} pts)")

        # === Step 3: Send feedback (LLM slots only; bots receive nothing) ===
        for i in range(6):
            if is_bot[i]:
                continue
            feedback = (f"Round {round_num} results: {choices}. "
                        f"{'All matched!' if matched else 'No match.'} "
                        f"Points: {points}. Your total: {cumulative_points[i]}")
            add_message(players[i], "user", feedback)

        # Persist this round immediately (flushed) so it can be plotted live.
        writer.writerow({
            'game': game_num, 'partner_condition': partner_condition,
            'bot_slots': bot_slots, 'round': round_num,
            'p1': choices[0], 'p2': choices[1], 'p3': choices[2],
            'p4': choices[3], 'p5': choices[4], 'p6': choices[5],
            'p1_raw': raw_responses[0], 'p2_raw': raw_responses[1], 'p3_raw': raw_responses[2],
            'p4_raw': raw_responses[3], 'p5_raw': raw_responses[4], 'p6_raw': raw_responses[5],
            'p1_reasoning': strategies[0], 'p2_reasoning': strategies[1], 'p3_reasoning': strategies[2],
            'p4_reasoning': strategies[3], 'p5_reasoning': strategies[4], 'p6_reasoning': strategies[5],
            'matched': matched, 'points': points,
            'p1_total': cumulative_points[0], 'p2_total': cumulative_points[1], 'p3_total': cumulative_points[2],
            'p4_total': cumulative_points[3], 'p5_total': cumulative_points[4], 'p6_total': cumulative_points[5],
        })
        csvfile.flush()

    print(f"Final scores: {cumulative_points}", flush=True)

################################################################
# === Done ===
################################################################

csvfile.close()
print(f"\n=== ALL GAMES COMPLETE ===")
print(f"Results saved to {filename}")
