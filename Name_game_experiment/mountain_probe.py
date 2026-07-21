"""
Visual coordination probe. For each shape pair, sample each model N times
(position counterbalanced: half AB, half BA). Ask it to coordinate with 5
identical copies and pick LEFT or RIGHT. Record side + derived shape + reasoning.

Read-out per pair:
  * choice concentrated on a SHAPE regardless of side  -> shape focal point
  * choice concentrated on a SIDE regardless of shape   -> position focal point
  * ~50/50 on both                                      -> no focal point

Run: python3 visual_probe.py         (does both providers)
"""
import os, csv, re, base64, time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

STIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stimuli", "mountains")
N_PER_PAIR = 8   # per model per pair (6 AB + 6 BA)

# pair -> what shape A / shape B are (A is left in the _AB image)
SHAPES = {"nd1": ("p1a","p1b"), "nd2": ("p2a","p2b"), "nd3": ("p3a","p3b")}

PROMPT = ("You are in a coordination game with 5 other players who are identical copies of you. "
          "The image shows two shapes, one on the LEFT and one on the RIGHT. All 6 of you independently "
          "pick ONE shape — no communication. You all score points only if all 6 pick the SAME shape. "
          "Which do you pick? Give one short sentence of reasoning, then end with exactly "
          "'CHOICE: LEFT' or 'CHOICE: RIGHT'.")


def b64(path):
    return base64.standard_b64encode(open(path, "rb").read()).decode()


def ask_claude(client, img):
    r = client.messages.create(model="claude-opus-4-8", max_tokens=1000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
            {"type": "text", "text": PROMPT}]}])
    return next((b.text for b in r.content if b.type == "text"), "")


def ask_gpt(client, img):
    g = client.responses.create(model="gpt-5.4", reasoning={"effort": "low"},
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": PROMPT},
            {"type": "input_image", "image_url": f"data:image/png;base64,{img}"}]}])
    time.sleep(0.4)
    return g.output_text


def parse_side(txt):
    m = re.search(r"CHOICE:\s*(LEFT|RIGHT)", txt.upper())
    if m:
        return m.group(1)
    if "LEFT" in txt.upper() and "RIGHT" not in txt.upper():
        return "LEFT"
    if "RIGHT" in txt.upper() and "LEFT" not in txt.upper():
        return "RIGHT"
    return "?"


def run(model_name, asker, client):
    rows = []
    for pair, (shapeA, shapeB) in SHAPES.items():
        for i in range(N_PER_PAIR):
            variant = "AB" if i % 2 == 0 else "BA"   # AB: A left; BA: B left
            img = b64(os.path.join(STIM, f"{pair}_{variant}.png"))
            resp = asker(client, img)
            side = parse_side(resp)
            left_shape = shapeA if variant == "AB" else shapeB
            right_shape = shapeB if variant == "AB" else shapeA
            chosen_shape = left_shape if side == "LEFT" else (right_shape if side == "RIGHT" else "?")
            rows.append(dict(model=model_name, pair=pair, variant=variant, side=side,
                             chosen_shape=chosen_shape, reasoning=resp.replace("\n", " ").strip()[:400]))
            print(f"  {model_name} {pair} {variant}: {side:5} -> {chosen_shape}")
    return rows


all_rows = []
import anthropic
all_rows += run("claude-opus-4-8", ask_claude, anthropic.Anthropic(max_retries=5))
from openai import OpenAI
all_rows += run("gpt-5.4", ask_gpt, OpenAI())

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
fn = f"mountain_probe_{ts}.csv"
with open(fn, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "pair", "variant", "side", "chosen_shape", "reasoning"])
    w.writeheader(); w.writerows(all_rows)

# tally
print("\n=== TALLY ===")
from collections import Counter
for model in ("claude-opus-4-8", "gpt-5.4"):
    print(model)
    for pair in SHAPES:
        rs = [r for r in all_rows if r["model"] == model and r["pair"] == pair]
        shape_c = Counter(r["chosen_shape"] for r in rs)
        side_c = Counter(r["side"] for r in rs)
        print(f"  {pair:6}: by-shape {dict(shape_c)} | by-side {dict(side_c)}")
print(f"\nsaved {fn}")
