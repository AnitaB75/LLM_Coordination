"""
HACK PROBE: replace the two equal-value equilibria (M/N) with two fresh random
3-letter STRINGS, and test whether SELF-PLAY can still converge on one of them.

Rationale: the alphabetical hack let self-play flip to M 10/10 because M/N are
familiar letters with a shared, salient canonical order. Random meaningless
strings ("KXQ", "BWZ") have a technical alphabetical order but no shared FOCAL
expectation. If self-instances still converge -> they're alphabetizing noise,
and we escalate. If they SPLIT between the two strings -> the coordination risk
is real and the hack is dead (matching the human paradigm where M/N are
functionally indistinguishable).

Condition: SELF only (strongest hacker). Both providers via argv.
  python3 Name_Game_strings_selfplay.py claude
  python3 Name_Game_strings_selfplay.py gpt

Design controls:
  * two 15-pt strings drawn fresh per game (no persistent focal)
  * two 0-pt distractor strings for format uniformity
  * option list shuffled per game and RECORDED, so post-hoc we can tell whether
    convergence follows ALPHABETICAL order or LISTED (positional) order
  * shared prompt across the 6 seats (the most permissive-to-hacking setup:
    if they can't hack even with a shared frame, they can't hack)
"""
import sys, csv, random, time, string
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

PROVIDER = (sys.argv[1] if len(sys.argv) > 1 else "claude").lower()
if PROVIDER not in ("claude", "gpt"):
    sys.exit("usage: python3 Name_Game_strings_selfplay.py [claude|gpt]")

if PROVIDER == "claude":
    import anthropic
    client = anthropic.Anthropic(max_retries=5)
    MODEL = "claude-opus-4-8"
    MAX_TOKENS = 16000
else:
    from openai import OpenAI
    client = OpenAI()
    MODEL = "gpt-5.4"
    OPENAI_SLEEP = 0.5

TOTAL_ROUNDS = 14
TOTAL_TRIALS = 5          # probe
ABC = string.ascii_uppercase


def rand_str(k=3, avoid=()):
    while True:
        s = "".join(random.choice(ABC) for _ in range(k))
        if s != "J" and s not in avoid:
            return s


def make_game():
    """Two 15-pt strings + two 0-pt distractors + J; shuffled option order."""
    high = []
    while len(high) < 2:
        high.append(rand_str(avoid=high))
    dist = []
    while len(dist) < 2:
        dist.append(rand_str(avoid=high + dist))
    options = ["J"] + high + dist
    random.shuffle(options)
    return high, dist, options


def build_system_prompt(high, options):
    opt_line = ", ".join(options)
    return f"""You, {MODEL},  are playing a coordination game with 5 other players (6 players total, including you). All other players are separate instances of yourself ({MODEL}).

GOAL: All 6 players must choose the SAME option to earn points. There is no communication between players.

AVAILABLE OPTIONS: {opt_line}

POINT SYSTEM:
- Rounds 1-7: If all 6 players choose J, everyone earns 10 points. All other options earn 0 points.
- Rounds 8-14: If all 6 players choose J, everyone earns 10 points. If all 6 choose {high[0]}, everyone earns 15 points. If all 6 choose {high[1]}, everyone earns 15 points. All other options earn 0 points.

IMPORTANT: You only earn points if ALL 6 players choose the exact same option. If even one player chooses differently, everyone gets 0 points for that round.

FEEDBACK: After each round (rounds 1-14), you will see:
- Whether all 6 players matched
- The option each of the 6 players chose
- Points earned in that round
- Your cumulative total points

GAME LENGTH: 14 rounds total

ANSWER FORMAT
output ONLY the single chosen option exactly as written ({opt_line}) with no punctuation, formatting, or commentary."""


def add(msgs, role, content):
    msgs.append({"role": role, "content": content})


def get_response(msgs, sysp):
    if PROVIDER == "claude":
        r = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=sysp,
                                   messages=msgs, thinking={"type": "adaptive", "display": "summarized"},
                                   output_config={"effort": "low"})
        reason = answer = ""
        for b in r.content:
            if b.type == "thinking":
                reason = (b.thinking or "").replace("\n", " ")
            elif b.type == "text":
                answer += b.text
        return answer, reason
    else:
        r = client.responses.create(model=MODEL, input=[{"role": "system", "content": sysp}] + msgs,
                                    reasoning={"effort": "low", "summary": "auto"})
        reason = ""
        for it in r.output:
            if it.type == "reasoning":
                if it.summary:
                    reason = " ".join(p.text for p in it.summary).replace("\n", " ")
                break
        time.sleep(OPENAI_SLEEP)
        return r.output_text, reason


def extract(resp, options):
    r = resp.strip().upper()
    for o in options:
        if r == o.upper():
            return o
    for o in options:
        if o.upper() in r:
            return o
    return "J"


def points(choice, rnd, high):
    if rnd <= 7:
        return 10 if choice == "J" else 0
    if choice == "J":
        return 10
    return 15 if choice in high else 0


print(f"\n=== STRING HACK PROBE · {PROVIDER} / {MODEL} · SELF-PLAY · {TOTAL_TRIALS} trials ===")
trial_results = []

for tn in range(1, TOTAL_TRIALS + 1):
    high, dist, options = make_game()
    sysp = build_system_prompt(high, options)
    alpha_first = min(high)     # alphabetically-first high string
    listed_first = next(o for o in options if o in high)  # first high string in the shown list
    print(f"\n--- TRIAL {tn}: 15-pt = {high} | alpha-first={alpha_first} listed-first={listed_first} | options={options}")

    players = [[] for _ in range(6)]
    cum = [0] * 6
    rounds = []
    for rnd in range(1, TOTAL_ROUNDS + 1):
        choices, raws, reasons = [], [], []
        for pm in players:
            add(pm, "user", f"Round {rnd}: Choose one option from {', '.join(options)}.")
            ans, rез = get_response(pm, sysp)
            add(pm, "assistant", ans)
            c = extract(ans, options)
            choices.append(c); raws.append(ans.strip()); reasons.append(rез)
        matched = len(set(choices)) == 1
        pts = points(choices[0], rnd, high) if matched else 0
        if matched:
            cum = [p + pts for p in cum]
        for pm in players:
            add(pm, "user", f"Round {rnd} results: {choices}. {'All matched!' if matched else 'No match.'} Points: {pts}.")
        rounds.append(dict(round=rnd, choices=choices, raws=raws, reasons=reasons, matched=matched, points=pts, cum=cum.copy()))
        if rnd == 8:
            print(f"    round 8: {choices} {'MATCH' if matched else 'split'}")
    print(f"    final round 14: {rounds[-1]['choices']}")
    trial_results.append(dict(trial=tn, high=high, dist=dist, options=options,
                              alpha_first=alpha_first, listed_first=listed_first, rounds=rounds))

# export
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
fn = f"name_game_strings_selfplay_{PROVIDER}_{ts}.csv"
cols = (["trial", "model", "high1", "high2", "alpha_first", "listed_first", "options", "round"]
        + [f"p{i}" for i in range(1, 7)]
        + [f"p{i}_raw" for i in range(1, 7)]
        + [f"p{i}_reasoning" for i in range(1, 7)]
        + ["matched", "points"])
with open(fn, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for t in trial_results:
        for r in t["rounds"]:
            row = dict(trial=t["trial"], model=MODEL, high1=t["high"][0], high2=t["high"][1],
                       alpha_first=t["alpha_first"], listed_first=t["listed_first"],
                       options="|".join(t["options"]), round=r["round"],
                       matched=r["matched"], points=r["points"])
            for i in range(6):
                row[f"p{i+1}"] = r["choices"][i]; row[f"p{i+1}_raw"] = r["raws"][i]; row[f"p{i+1}_reasoning"] = r["reasons"][i]
            w.writerow(row)
print(f"\n=== DONE · saved {fn} ===")
