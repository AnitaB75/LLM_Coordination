"""
ROUNDABOUT paradigm — un-hackable-by-construction coordination, self-play test.

Six underlying exits arranged in a hexagon (indices 0..5, clockwise):
  CONVENTION = exit 0  -> 10 pts (all rounds)
  HIGH       = exits 2 and 4 -> 15 pts each (rounds 8-14 only)
  ZERO       = exits 1, 3, 5 -> 0 pts
The three scoring exits (0, 2, 4) form an equilateral triangle, so each 15-pt
exit is exactly 120 deg from the convention AND from the other 15-pt exit.

KEY MECHANIC — each seat gets a FIXED private frame for the whole game:
  rotation k_p in 0..5  and  mirror m_p in {0,1}   (a dihedral relabelling)
  slot s in the player's view  ->  underlying exit  (s+k)%6      if not mirrored
                                                    (k-s)%6      if mirrored
So a given real exit is "NE" to one driver and "S" to another.

Why BOTH rotation and mirroring:
  * rotation alone kills "pick N"/"pick up" (absolute position), BUT
  * rotation PRESERVES handedness, so "pick the 15 that is clockwise from my 10"
    would still name the same real exit for everyone -> a leak.
  * random mirroring flips handedness for ~half the seats, so that rule splits
    them 50/50. With both, no verbal rule identifies a shared exit.

Coordinable vs not (this is the whole paradigm):
  * The CONVENTION is uniquely identified by its payoff ("my only 10"), so all
    seats can converge on it with no shared frame -> rounds 1-7 work.
  * The two 15s are symmetric and payoff-identical, so nothing a priori picks
    between them -> reaching 15 requires reading the FEEDBACK and tipping.

Feedback is rendered per-seat: each driver sees how many cars took each exit,
positioned in THEIR OWN frame. Frames are fixed across rounds, so history is
trackable within a seat even though labels differ between seats.

LEAK CHECK: round 8 should SCATTER. Instant convergence => a cue leaked.

  python3 Name_Game_roundabout_selfplay.py [claude|gpt]
"""
import sys, csv, re, random, time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

PROVIDER = (sys.argv[1] if len(sys.argv) > 1 else "claude").lower()
if PROVIDER not in ("claude", "gpt"):
    sys.exit("usage: python3 Name_Game_roundabout_selfplay.py [claude|gpt]")

if PROVIDER == "claude":
    import anthropic
    client = anthropic.Anthropic(max_retries=5)
    MODEL, MAX_TOKENS = "claude-opus-4-8", 16000
else:
    from openai import OpenAI
    client = OpenAI()
    MODEL, OPENAI_SLEEP = "gpt-5.4", 0.4

TOTAL_ROUNDS, TOTAL_TRIALS, N_PLAYERS = 14, 5, 6
LABELS = ["N", "NE", "SE", "S", "SW", "NW"]      # slots 0..5, clockwise
CONV, HIGHS = 0, {2, 4}


def slot_to_underlying(k, m, s):
    return (k - s) % 6 if m else (s + k) % 6


CONV_LATE = 3   # convention is DEVALUED in rounds 8-14 to induce exploration

def payoff(u, rnd):
    if u == CONV:
        return 10 if rnd <= 7 else CONV_LATE
    if u in HIGHS and rnd >= 8:
        return 15
    return 0


def view_lines(k, m, rnd):
    return "\n".join(f"  {LABELS[s]}: {payoff(slot_to_underlying(k,m,s), rnd)} points"
                     for s in range(6))


def feedback_lines(k, m, counts):
    return ", ".join(f"{LABELS[s]}: {counts[slot_to_underlying(k,m,s)]}" for s in range(6))


def build_system_prompt():
    return f"""You, {MODEL},  are driving into a six-exit junction with 5 other drivers (6 drivers total, including you). All other drivers are separate instances of yourself ({MODEL}).

GOAL: All 6 drivers must leave by the SAME exit to earn points. There is no communication between drivers.

CRITICAL — PRIVATE ORIENTATION: Each driver approaches the junction from a different direction, so each of you sees it from your own orientation. The compass labels (N, NE, SE, S, SW, NW) are PRIVATE TO YOUR OWN VIEW. An exit you call "NE" may be "S" to another driver. Direction labels do NOT identify the same exit across drivers, and neither does any rule based on them (for example "the exit clockwise from my 10-point exit" does not pick the same real exit for everyone).

POINT SYSTEM (you only score if ALL 6 drivers take the exact same exit):
- Rounds 1-7: one exit is worth 10 points; all others 0.
- Rounds 8-14: that same exit drops to only 3 points, and TWO other exits are worth 15 points each; all others 0.

FEEDBACK: after each round you will see how many drivers took each exit, shown in YOUR OWN view's directions. Your own orientation stays the same for the whole game.

ANSWER FORMAT
output ONLY one direction label (N, NE, SE, S, SW, or NW) with no punctuation or commentary."""


def add(msgs, role, content):
    msgs.append({"role": role, "content": content})


def get_response(msgs, sysp):
    if PROVIDER == "claude":
        r = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=sysp, messages=msgs,
                                   thinking={"type": "adaptive", "display": "summarized"},
                                   output_config={"effort": "low"})
        reason = answer = ""
        for b in r.content:
            if b.type == "thinking":
                reason = (b.thinking or "").replace("\n", " ")
            elif b.type == "text":
                answer += b.text
        return answer, reason
    g = client.responses.create(model=MODEL, input=[{"role": "system", "content": sysp}] + msgs,
                                reasoning={"effort": "low", "summary": "auto"})
    reason = ""
    for it in g.output:
        if it.type == "reasoning":
            if it.summary:
                reason = " ".join(p.text for p in it.summary).replace("\n", " ")
            break
    time.sleep(OPENAI_SLEEP)
    return g.output_text, reason


def extract_slot(resp):
    r = resp.strip().upper()
    for lab in ("NE", "NW", "SE", "SW"):          # two-char first
        if re.search(rf"\b{lab}\b", r):
            return LABELS.index(lab)
    for lab in ("N", "S"):
        if re.search(rf"\b{lab}\b", r):
            return LABELS.index(lab)
    return None


print(f"\n=== ROUNDABOUT DEVALUED self-play · {PROVIDER}/{MODEL} · {TOTAL_TRIALS} trials ===")
sysp = build_system_prompt()
trials = []

for tn in range(1, TOTAL_TRIALS + 1):
    frames = [(random.randrange(6), random.random() < 0.5) for _ in range(N_PLAYERS)]
    print(f"\n--- TRIAL {tn} · frames (rot,mirror): {frames}")
    players = [[] for _ in range(N_PLAYERS)]
    cum = [0] * N_PLAYERS
    rounds = []
    for rnd in range(1, TOTAL_ROUNDS + 1):
        unders, raws, reasons = [], [], []
        for i, msgs in enumerate(players):
            k, m = frames[i]
            add(msgs, "user", f"Round {rnd}. Your view of the six exits:\n{view_lines(k,m,rnd)}\n"
                              f"Choose one exit (N, NE, SE, S, SW, NW).")
            ans, why = get_response(msgs, sysp)
            add(msgs, "assistant", ans.strip())
            s = extract_slot(ans)
            u = slot_to_underlying(k, m, s) if s is not None else CONV
            unders.append(u); raws.append(ans.strip()); reasons.append(why)
        counts = [unders.count(u) for u in range(6)]
        matched = len(set(unders)) == 1
        pts = payoff(unders[0], rnd) if matched else 0
        if matched:
            cum = [c + pts for c in cum]
        for i, msgs in enumerate(players):
            k, m = frames[i]
            add(msgs, "user", f"Round {rnd} result — drivers per exit in your view: {feedback_lines(k,m,counts)}. "
                              f"{'All 6 matched!' if matched else 'No match.'} Points: {pts}. Your total: {cum[i]}.")
        tag = "  <-- ROUND 8" if rnd == 8 else ""
        print(f"   r{rnd:2}: underlying={unders} counts={counts} {'MATCH' if matched else 'split'} {pts}p{tag}")
        rounds.append(dict(round=rnd, unders=unders, counts=counts, matched=matched,
                           points=pts, raws=raws, reasons=reasons))
    trials.append(dict(trial=tn, frames=frames, rounds=rounds))

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
fn = f"roundabout_devalued_{PROVIDER}_{ts}.csv"
cols = (["trial", "model", "frames", "round", "matched", "points"]
        + [f"p{i}_exit" for i in range(1, 7)] + [f"p{i}_raw" for i in range(1, 7)]
        + [f"p{i}_reasoning" for i in range(1, 7)] + [f"count_exit{u}" for u in range(6)])
with open(fn, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for t in trials:
        for r in t["rounds"]:
            row = dict(trial=t["trial"], model=MODEL, frames=str(t["frames"]), round=r["round"],
                       matched=r["matched"], points=r["points"])
            for i in range(6):
                row[f"p{i+1}_exit"] = r["unders"][i]; row[f"p{i+1}_raw"] = r["raws"][i]
                row[f"p{i+1}_reasoning"] = r["reasons"][i]
            for u in range(6):
                row[f"count_exit{u}"] = r["counts"][u]
            w.writerow(row)

def classify(unders):
    """A LEAK is converging on a 15-pt exit with no shared frame. Holding the
    convention is correct behaviour (it is uniquely identified by its payoff),
    and scattering means they attempted an uncoordinated switch."""
    s = set(unders)
    if len(s) == 1:
        u = next(iter(s))
        if u in HIGHS:
            return "LEAK — converged on a 15-pt exit despite private frames"
        if u == CONV:
            return "held convention (correct; no leak, no exploration)"
        return f"converged on 0-pt exit {u}"
    return f"scatter across {len(s)} exits (uncoordinated switch attempt)"


print("\n=== ROUND-8 CHECK ===")
for t in trials:
    r8, r14 = t["rounds"][7], t["rounds"][13]
    end = r14["unders"][0] if len(set(r14["unders"])) == 1 else "split"
    endlab = ("15-pt exit" if end in HIGHS else "convention" if end == CONV else end)
    print(f"  trial {t['trial']}: r8 -> {classify(r8['unders'])} | r14 lock={endlab}")

# did any 15-pt exit ever attract a plurality (tipping seed)?
print("\n=== EXPLORATION / TIPPING ===")
for t in trials:
    post = t["rounds"][7:]
    hi_cells = sum(sum(1 for u in r["unders"] if u in HIGHS) for r in post)
    peak = max((max(r["counts"][h] for h in HIGHS) for r in post), default=0)
    print(f"  trial {t['trial']}: 15-pt choices in rounds 8-14 = {hi_cells}/42 | peak on a single 15-pt exit = {peak}/6")
print(f"\nsaved {fn}")
