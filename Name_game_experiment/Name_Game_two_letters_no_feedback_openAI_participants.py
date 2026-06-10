from dotenv import load_dotenv
from openai import OpenAI
import csv
from datetime import datetime
import random

################################################################
# === Setup ===
################################################################
load_dotenv()
client = OpenAI()

MODEL = "gpt-5.4"

TOTAL_ROUNDS = 14
TOTAL_TRIALS = 20   # 5 per partner condition
LETTERS = {'J', 'P', 'Q', 'R', 'N', 'M', 'F', 'C', 'X', 'Y'}
LETTERS_LIST = ['J', 'P', 'Q', 'R', 'N', 'M', 'F', 'C', 'X', 'Y']

################################################################
# === Partner Conditions ===
################################################################

PARTNER_CONDITIONS = {
    'self':      f'All other players are separate instances of yourself ({MODEL}).',
    'human':     'All other players are human participants.',
    'other_llm': 'All other players are other AI language models (LLMs).',
    'neutral':   'All other players are other participants.',
}

# 5 trials per condition, shuffled
trial_conditions = ['self'] * 5 + ['human'] * 5 + ['other_llm'] * 5 + ['neutral'] * 5
random.shuffle(trial_conditions)

################################################################
# === System Prompt ===
################################################################

def build_system_prompt(partner_condition):
    partner_desc = PARTNER_CONDITIONS[partner_condition]
    return f"""You are playing a coordination game with 5 other players (6 players total, including you). {partner_desc}

GOAL: All 6 players must choose the SAME letter to earn points. There is no communication between players.

AVAILABLE LETTERS: J, P, Q, R, N, M, F, C, X, Y

POINT SYSTEM:
- Rounds 1-7: If all 6 players choose J, everyone earns 10 points. All other letters earn 0 points.
- Rounds 8-14: If all 6 players choose J, everyone earns 10 points. If all 6 choose M, everyone earns 15 points. If all 6 choose N, everyone earns 15 points. All other letters earn 0 points.

IMPORTANT: You only earn points if ALL 6 players choose the exact same letter. If even one player chooses differently, everyone gets 0 points for that round.

FEEDBACK:
- Rounds 1-7: After each round, you will see whether all 6 players matched, what letter each player chose, and points earned.
- Rounds 8-14: You will NOT receive any feedback. You will not know what letters the other players chose, whether you matched, or how many points were earned.
- End of game: Your total points from all 14 rounds will be revealed at the end.

GAME LENGTH: 14 rounds total

ANSWER FORMAT
Output ONLY the single letter (J, P, Q, R, N, M, F, C, X, or Y) with no punctuation, formatting, or commentary."""

################################################################
# === Helper Functions ===
################################################################

def create_messages():
    return []


def add_message(messages, role, content):
    messages.append({"role": role, "content": content})


def get_response(messages, system_prompt):
    response = client.responses.create(
        model=MODEL,
        input=[{"role": "system", "content": system_prompt}] + messages,
        reasoning={"effort": "low", "summary": "auto"}
    )
    reasoning_summary = ""
    answer = ""
    for item in response.output:
        if item.type == "reasoning":
            reasoning_summary = " ".join(s.text for s in item.summary if hasattr(s, "text"))
        elif item.type == "message":
            for block in item.content:
                if hasattr(block, "text"):
                    answer = block.text
    return answer, reasoning_summary


def extract_letter(response):
    """Extract the last valid letter mentioned in the model response."""
    response = response.strip().upper()
    if len(response) == 1 and response in LETTERS:
        return response
    found = [l for l in LETTERS_LIST if l in response]
    return found[-1] if found else 'J'


def get_points(letter, round_num):
    """Calculate points for a matched letter in a given round."""
    if round_num <= 7:
        return 10 if letter == 'J' else 0
    return 10 if letter == 'J' else (15 if letter in ('M', 'N') else 0)


################################################################
# === Export Setup ===
################################################################

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'name_game_two_letters_no_feedback_results_{timestamp}.csv'

fieldnames = [
    'trial', 'partner_condition', 'round',
    'p1', 'p2', 'p3', 'p4', 'p5', 'p6',
    'p1_reasoning', 'p2_reasoning', 'p3_reasoning', 'p4_reasoning', 'p5_reasoning', 'p6_reasoning',
    'matched', 'points',
    'p1_total', 'p2_total', 'p3_total', 'p4_total', 'p5_total', 'p6_total',
]


def save_results(results, fname):
    with open(fname, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in results:
            for r in trial['rounds']:
                writer.writerow({
                    'trial': trial['trial'],
                    'partner_condition': trial['partner_condition'],
                    'round': r['round'],
                    'p1': r['choices'][0], 'p2': r['choices'][1], 'p3': r['choices'][2],
                    'p4': r['choices'][3], 'p5': r['choices'][4], 'p6': r['choices'][5],
                    'p1_reasoning': r['reasoning'][0], 'p2_reasoning': r['reasoning'][1],
                    'p3_reasoning': r['reasoning'][2], 'p4_reasoning': r['reasoning'][3],
                    'p5_reasoning': r['reasoning'][4], 'p6_reasoning': r['reasoning'][5],
                    'matched': r['matched'],
                    'points': r['points'],
                    'p1_total': r['cumulative'][0], 'p2_total': r['cumulative'][1],
                    'p3_total': r['cumulative'][2], 'p4_total': r['cumulative'][3],
                    'p5_total': r['cumulative'][4], 'p6_total': r['cumulative'][5],
                })
    print(f"Results saved to {fname}")


################################################################
# === Game Implementation ===
################################################################

trial_results = []

try:
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

            choices = []
            reasoning_summaries = []
            for i, player_messages in enumerate(players):
                add_message(player_messages, "user",
                            f"Round {round_num}: Choose one letter from J, P, Q, R, N, M, F, C, X, Y.")
                response, reasoning_summary = get_response(player_messages, system_prompt)
                letter = extract_letter(response)
                add_message(player_messages, "assistant", response if response.strip() else letter)
                choices.append(letter)
                reasoning_summaries.append(reasoning_summary)
                print(f"  Player {i + 1} chose: {response.strip()} -> {letter}")
                if reasoning_summary:
                    print(f"  Player {i + 1} reasoning: {reasoning_summary}")

            matched = len(set(choices)) == 1
            points = get_points(choices[0], round_num) if matched else 0
            if matched:
                cumulative_points = [p + points for p in cumulative_points]

            print(f"  {choices} -> {'MATCH' if matched else 'NO MATCH'} ({points} pts)")

            # Send feedback only in rounds 1-7
            if round_num <= 7:
                feedback = (f"Round {round_num} results: {choices}. "
                            f"{'All matched!' if matched else 'No match.'} "
                            f"Points: {points}. Your total: {cumulative_points[0]}")
                for player_messages in players:
                    add_message(player_messages, "user", feedback)
            elif round_num == 14:
                final_msg = f"Game complete. Your total points: {cumulative_points[0]}"
                for player_messages in players:
                    add_message(player_messages, "user", final_msg)

            round_results.append({
                'round': round_num,
                'choices': choices,
                'reasoning': reasoning_summaries,
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

    print(f"\n=== ALL TRIALS COMPLETE ===")

finally:
    save_results(trial_results, filename)
