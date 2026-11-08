"""
Name Game -- CONVENTION ENTRENCHMENT x TWO providers, human framing only.

Question: does the amount of J-play accumulated BEFORE the M/N payoff shift
determine whether players abandon the J convention for the higher-paying one?

Conditions (the ONLY thing that varies is how many J-only rounds come first):
  j7   7 J-only rounds, shift announced at round 8  -> 14 rounds total (original)
  j4   4 J-only rounds, shift announced at round 5  -> 11 rounds total
  j1   1 J-only round,  shift announced at round 2  ->  8 rounds total

The post-shift phase is held at 7 rounds in every condition, so each condition
gives the players the same amount of time to switch; only the entrenchment of
the old convention differs.

THE SHIFT IS NOT ANNOUNCED IN ADVANCE. Before the shift the system prompt
states ONLY the J payoff; M and N sit in the available-letter list but are
worth nothing and are never mentioned. On the shift round -- immediately after
the condition's J-only phase ends -- the players receive a RULE UPDATE message
and the system prompt switches to the M/N payoff table. Players therefore
cannot anticipate the shift or pre-position for it: the convention they build
in the J-only phase is built in good faith.

Providers (select with argv[1]):
  claude  -> claude-opus-4-8   (Anthropic Messages API)
  gpt     -> gpt-5.4           (OpenAI Responses API)

    python3 Name_Game_entrenchment_human.py claude
    python3 Name_Game_entrenchment_human.py gpt
    python3 Name_Game_entrenchment_human.py gpt --print-prompts   # no API calls

WORDING: the prompt scaffold is byte-identical to the existing runners. The
pre-shift / post-shift POINT SYSTEM blocks and the RULE UPDATE text are taken
verbatim from Name_Game_bot_follow_human.py (which introduces M/N mid-game the
same way), with the M/N value set to 15 to match the M/N payoff design in
Name_Game_LLM_feedback_MN.py. Only the round NUMBERS vary by condition.

This is a pure BELIEF manipulation: all 6 seats are the same model given the
same system prompt. Nobody is actually human.

Keys needed in the repo-root .env (whichever provider you run):
  * ANTHROPIC_API_KEY   (claude)
  * OPENAI_API_KEY      (gpt)
"""

import os
import sys
import csv
import time
from datetime import datetime
from dotenv import load_dotenv

# .env lives at the repo root, one level above this script.
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(os.path.dirname(_HERE), ".env"))

################################################################
# === Provider selection ===
################################################################
_POSITIONAL = [a for a in sys.argv[1:] if not a.startswith("-")]
_FLAGS = {a for a in sys.argv[1:] if a.startswith("-")}

# --print-prompts renders every prompt the models will see and exits WITHOUT
# any API call, so the wording can be checked before spending anything.
PRINT_PROMPTS_ONLY = "--print-prompts" in _FLAGS

# --single-high makes M the ONLY high-value option; N reverts to a dead letter
# worth 0, like P/Q/R/etc. The available-letter list is unchanged, so the only
# thing that differs from the default arm is whether the switch target is
# unique. This isolates the tie-breaking cost: with M and N both at 15,
# switching requires a SECOND act of coordination (which one?), and the
# post-shift traces suggested that, not entrenchment, may be what protects J.
SINGLE_HIGH = "--single-high" in _FLAGS
VARIANT = "singleM" if SINGLE_HIGH else "MN"

PROVIDER = (_POSITIONAL[0] if _POSITIONAL else "claude").lower()
if PROVIDER not in ("claude", "gpt"):
    sys.exit("usage: python3 Name_Game_entrenchment_human.py [claude|gpt] [--print-prompts]")

OPENAI_SLEEP = 0.5
client = None

if PROVIDER == "claude":
    MODEL = "claude-opus-4-8"
    MAX_TOKENS = 16000                             # room for adaptive thinking + one letter
else:  # gpt
    MODEL = "gpt-5.4"

# --model NAME overrides the provider default. NOTE: the model name is
# interpolated into the system prompt ("You, <MODEL>, are playing..."), as in
# every runner in this project, so changing it changes what the agent is told
# it is -- that is intended, not a leak.
_MODEL_OVERRIDE = next((a.split("=", 1)[1] for a in _FLAGS if a.startswith("--model=")), None)
if _MODEL_OVERRIDE is None and "--model" in sys.argv:
    _MODEL_OVERRIDE = sys.argv[sys.argv.index("--model") + 1]
if _MODEL_OVERRIDE:
    MODEL = _MODEL_OVERRIDE
    # Key is on the 5000 RPM / 2M TPM tier (confirmed via rate-limit headers),
    # so the old 3-RPM 21s sleep is unnecessary. Calls are sequential and each
    # takes a few seconds anyway. Raise this if you start seeing 429s.


def make_client():
    """Instantiate the provider SDK client. Not called under --print-prompts."""
    if PROVIDER == "claude":
        import anthropic
        return anthropic.Anthropic(max_retries=5)   # SDK backs off on 429 / 5xx
    from openai import OpenAI
    return OpenAI(max_retries=5)

################################################################
# === Experiment constants ===
################################################################
# condition name -> number of J-only rounds before the M/N shift.
# The shift is announced on round (j_only_rounds + 1): round 8 for j7 (as in
# the original design), round 5 for j4, round 2 for j1.
CONDITIONS = {
    'j7': 7,   # the usual design
    'j4': 4,
    'j1': 1,
}
MN_ROUNDS = 7          # rounds AFTER the shift, held constant across conditions
J_VALUE = 10           # the old convention's payoff, never varies

# Post-shift payoff for the high-value letter(s). One trial is run per value,
# crossed with every entrenchment condition, so the payoff GAP can be varied
# while everything else is held fixed:
#   --mn-values 15,20,40   ->  gaps of 1.5x, 2x and 4x over J
# Default 15 matches Name_Game_LLM_feedback_MN.py.
_MNV = next((a.split("=", 1)[1] for a in _FLAGS if a.startswith("--mn-values=")), None)
if _MNV is None and "--mn-values" in sys.argv:
    _MNV = sys.argv[sys.argv.index("--mn-values") + 1]
MN_VALUES = [int(v) for v in _MNV.split(",")] if _MNV else [15]

# Which letters become high-value at the shift. Default is the original design
# (M and N tied at 15); --single-high leaves M alone as the switch target.
HIGH_LETTERS = ('M',) if SINGLE_HIGH else ('M', 'N')
LETTERS = {'J', 'P', 'Q', 'R', 'N', 'M', 'F', 'C', 'X', 'Y'}


def _flag_value(name, default=None):
    """Read --name=value, or --name value."""
    v = next((a.split("=", 1)[1] for a in _FLAGS if a.startswith(f"--{name}=")), None)
    if v is None and f"--{name}" in sys.argv:
        v = sys.argv[sys.argv.index(f"--{name}") + 1]
    return v if v is not None else default


# --replicates=N  independent games per cell (default 1). Each replicate is a
# fresh game with 6 new seats; nothing carries over between them.
TRIALS_PER_CONDITION = int(_flag_value("replicates", 1))

# --conditions=j7,j4  restricts which entrenchment conditions run, so the three
# can be launched as separate concurrent processes (they are independent, and
# the rate limits are nowhere near binding).
_CONDS = _flag_value("conditions")
RUN_CONDITIONS = tuple(_CONDS.split(",")) if _CONDS else ('j7', 'j4', 'j1')
for _c in RUN_CONDITIONS:
    if _c not in CONDITIONS:
        sys.exit(f"unknown condition {_c!r}; choose from {list(CONDITIONS)}")

# Fixed order, not shuffled: with one trial per cell there is nothing to
# counterbalance, and a fixed order keeps the run reproducible. Each trial is an
# independent game with 6 fresh seats and no shared conversation state.
# Payoff level varies slowest so each level's three conditions run together.
trial_cells = [(c, v) for v in MN_VALUES
                      for c in RUN_CONDITIONS
                      for _ in range(TRIALS_PER_CONDITION)]
TOTAL_TRIALS = len(trial_cells)

# Byte-identical to the 'human' string in Name_Game_both_3cond.py.
PARTNER_DESC = 'All other players are human participants.'


def rule_update_text(mn_value):
    """Announced once, on the shift round. Verbatim from
    Name_Game_bot_follow_human.py with the high-value payoff interpolated."""
    return (
        "RULE UPDATE: Starting this round, the point values have changed. "
        f"If all 6 players choose J, everyone earns {J_VALUE} points. "
        + "".join(f"If all 6 players choose {L}, everyone earns {mn_value} points. "
                  for L in HIGH_LETTERS)
        + "All other letters earn 0 points."
    )

################################################################
# === System Prompt ===
################################################################

def build_system_prompt(round_num, j_only_rounds, mn_value):
    """
    Payoffs in effect THIS round. Before the shift the M/N payoffs are absent
    entirely -- nothing hints that they are coming.
    """
    total = j_only_rounds + MN_ROUNDS
    if round_num <= j_only_rounds:
        point_system = (
            "POINT SYSTEM:\n"
            f"- If all 6 players choose J, everyone earns {J_VALUE} points. All other letters earn 0 points."
        )
    else:
        point_system = (
            "POINT SYSTEM:\n"
            f"- If all 6 players choose J, everyone earns {J_VALUE} points.\n"
            + "".join(f"- If all 6 players choose {L}, everyone earns {mn_value} points.\n"
                      for L in HIGH_LETTERS)
            + "- All other letters earn 0 points."
        )
    return f"""You, {MODEL},  are playing a coordination game with 5 other players (6 players total, including you). {PARTNER_DESC}

GOAL: All 6 players must choose the SAME letter to earn points. There is no communication between players.

AVAILABLE LETTERS: J, P, Q, R, N, M, F, C, X, Y

{point_system}

IMPORTANT: You only earn points if ALL 6 players choose the exact same letter. If even one player chooses differently, everyone gets 0 points for that round.

FEEDBACK: After each round (rounds 1-{total}), you will see:
- Whether all 6 players matched
- The letter each of the 6 players chose
- Points earned in that round
- Your cumulative total points

GAME LENGTH: {total} rounds total

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


def get_response(messages, system_prompt, _attempts=6):
    """
    Provider call with backoff on TRANSIENT failures.

    The SDKs already retry (max_retries=5), but a network drop that outlives
    their window raises APIConnectionError and kills the run -- which cost a
    half-finished 9-cell sweep once. Retry on top of them, with long waits, so
    a blip costs minutes rather than the whole run. Auth/billing/bad-request
    errors are NOT transient and are re-raised immediately.
    """
    call = get_response_claude if PROVIDER == "claude" else get_response_gpt
    for attempt in range(_attempts):
        try:
            return call(messages, system_prompt)
        except Exception as e:
            name = type(e).__name__
            fatal = ("Authentication", "PermissionDenied", "NotFound", "BadRequest")
            if any(f in name for f in fatal) or attempt == _attempts - 1:
                raise
            wait = min(60, 5 * 2 ** attempt)
            print(f"    [{name}] retry {attempt + 1}/{_attempts - 1} in {wait}s: {str(e)[:80]}",
                  flush=True)
            time.sleep(wait)


def extract_letter(response):
    """Extract a single valid letter from model response."""
    response = response.strip().upper()
    if len(response) == 1 and response in LETTERS:
        return response
    for letter in LETTERS:
        if letter in response:
            return letter
    return 'J'


def get_points(letter, round_num, j_only_rounds, mn_value):
    """Calculate points for a matched letter in a given round."""
    if round_num <= j_only_rounds:
        return J_VALUE if letter == 'J' else 0
    return J_VALUE if letter == 'J' else (mn_value if letter in HIGH_LETTERS else 0)


################################################################
# === Game Implementation ===
################################################################

def main():
    global client
    client = make_client()

    print(f"\n=== PROVIDER: {PROVIDER}  MODEL: {MODEL}  VARIANT: {VARIANT} ===")
    print(f"High-value letter(s) after the shift: {'/'.join(HIGH_LETTERS)} "
          f"at {MN_VALUES} pts vs J={J_VALUE} "
          f"(gap {', '.join(f'{v/J_VALUE:g}x' for v in MN_VALUES)})")
    print("Framing: human (all conditions).  Shift is UNANNOUNCED until it happens.")
    print(f"{TOTAL_TRIALS} trials (condition x payoff): "
          + ", ".join(f"{c}@{v}" for c, v in trial_cells))

    trial_results = []

    for trial_num in range(1, TOTAL_TRIALS + 1):
        condition, mn_value = trial_cells[trial_num - 1]
        j_only_rounds = CONDITIONS[condition]
        total_rounds = j_only_rounds + MN_ROUNDS
        shift_round = j_only_rounds + 1
        RULE_UPDATE_TEXT = rule_update_text(mn_value)

        print(f"\n=== TRIAL {trial_num}/{TOTAL_TRIALS}  ({condition}: "
              f"{j_only_rounds} J-only + {MN_ROUNDS} post = {total_rounds} rounds, "
              f"rule update at round {shift_round}, "
              f"{'/'.join(HIGH_LETTERS)}={mn_value} vs J={J_VALUE}) ===")

        players = [create_messages() for _ in range(6)]
        cumulative_points = [0] * 6
        round_results = []

        for round_num in range(1, total_rounds + 1):
            # System prompt reflects the payoffs in effect THIS round.
            system_prompt = build_system_prompt(round_num, j_only_rounds, mn_value)

            # On the shift round, announce the new payoffs.
            if round_num == shift_round:
                for player_messages in players:
                    add_message(player_messages, "user", RULE_UPDATE_TEXT)

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
            points = get_points(choices[0], round_num, j_only_rounds, mn_value) if matched else 0
            if matched:
                cumulative_points = [p + points for p in cumulative_points]

            phase = 'pre_shift' if round_num <= j_only_rounds else 'post_shift'
            print(f"  R{round_num:2} [{phase:10}]: {choices} -> "
                  f"{'MATCH' if matched else 'no match'} ({points} pts)")

            for i, player_messages in enumerate(players):
                feedback = (f"Round {round_num} results: {choices}. "
                            f"{'All matched!' if matched else 'No match.'} "
                            f"Points: {points}. Your total: {cumulative_points[i]}")
                add_message(player_messages, "user", feedback)

            round_results.append({
                'round': round_num,
                'phase': phase,
                'round_rel_shift': round_num - j_only_rounds,   # 1 = first post-shift round
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
            'condition': condition,
            'mn_value': mn_value,
            'j_only_rounds': j_only_rounds,
            'total_rounds': total_rounds,
            'rounds': round_results,
            'final_scores': cumulative_points,
        })

        # Persist after every trial so a crash costs at most one trial.
        write_results(trial_results)
        print(f"  saved {trial_num}/{TOTAL_TRIALS} trials -> {os.path.basename(RESULT_PATH)}")

    print(f"\n=== {PROVIDER.upper()} COMPLETE ===")
    print(f"Results saved to {RESULT_PATH}")


################################################################
# === Export Results ===
################################################################
# Written after EVERY completed trial, not just at the end. A run that dies
# partway (rate limit, credit exhaustion, network) then still leaves every
# finished trial on disk with its reasoning traces intact -- only the
# in-flight trial is lost.

RESULTS_DIR = os.path.join(_HERE, 'results')
_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
_MODEL_TAG = MODEL.replace('.', '').replace('-', '')      # gpt-5.5 -> gpt55
# Tag the conditions when a run is restricted, so concurrent per-condition
# processes cannot collide on one filename.
_COND_TAG = ("_" + "".join(RUN_CONDITIONS)) if len(RUN_CONDITIONS) < len(CONDITIONS) else ""
# Payoff must be in the name too: two runs differing only by payoff, launched in
# the same second, would otherwise share a filename and clobber each other.
_PAY_TAG = "_v" + "-".join(str(v) for v in MN_VALUES)
RESULT_PATH = os.path.join(
    RESULTS_DIR,
    f'name_game_entrenchment_{VARIANT}_{_MODEL_TAG}{_COND_TAG}{_PAY_TAG}_{_STAMP}.csv')

FIELDNAMES = [
    'trial', 'variant', 'high_letters', 'mn_value', 'j_value', 'payoff_gap',
    'condition', 'j_only_rounds',
    'total_rounds', 'shift_round',
    'partner_condition', 'model', 'round', 'phase', 'round_rel_shift',
    'p1', 'p2', 'p3', 'p4', 'p5', 'p6',
    'p1_raw', 'p2_raw', 'p3_raw', 'p4_raw', 'p5_raw', 'p6_raw',
    'p1_reasoning', 'p2_reasoning', 'p3_reasoning', 'p4_reasoning', 'p5_reasoning', 'p6_reasoning',
    'matched', 'points',
    'p1_total', 'p2_total', 'p3_total', 'p4_total', 'p5_total', 'p6_total',
]


def write_results(trial_results):
    """(Re)write the CSV from every trial completed so far."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for trial in trial_results:
            for r in trial['rounds']:
                writer.writerow({
                    'trial': trial['trial'],
                    'variant': VARIANT,
                    'high_letters': "/".join(HIGH_LETTERS),
                    'mn_value': trial['mn_value'],
                    'j_value': J_VALUE,
                    'payoff_gap': round(trial['mn_value'] / J_VALUE, 2),
                    'condition': trial['condition'],
                    'j_only_rounds': trial['j_only_rounds'],
                    'total_rounds': trial['total_rounds'],
                    'shift_round': trial['j_only_rounds'] + 1,
                    'partner_condition': 'human',
                    'model': MODEL,
                    'round': r['round'],
                    'phase': r['phase'],
                    'round_rel_shift': r['round_rel_shift'],
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


if __name__ == "__main__":
    if PRINT_PROMPTS_ONLY:
        print(f"{TOTAL_TRIALS} trials = {len(CONDITIONS)} conditions x "
              f"{len(MN_VALUES)} payoff level(s) {MN_VALUES}, "
              f"high letter(s) {'/'.join(HIGH_LETTERS)} vs J={J_VALUE}")
        for _v in MN_VALUES:
            for _name, _k in CONDITIONS.items():
                _total = _k + MN_ROUNDS
                print(f"\n{'='*72}\n=== {_name} x {'/'.join(HIGH_LETTERS)}={_v}: "
                      f"{_k} J-only round(s), rule update at round {_k + 1}, "
                      f"{_total} rounds total\n{'='*72}")
                print(f"\n--- SYSTEM PROMPT, rounds 1-{_k} (pre-shift) ---\n")
                print(build_system_prompt(1, _k, _v))
                print(f"\n--- USER MESSAGE injected at the start of round {_k + 1} ---\n")
                print(rule_update_text(_v))
                print(f"\n--- SYSTEM PROMPT, rounds {_k + 1}-{_total} (post-shift) ---\n")
                print(build_system_prompt(_k + 1, _k, _v))
        sys.exit(0)
    main()
