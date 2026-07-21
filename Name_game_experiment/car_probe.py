"""
Five-landscape coordination probe. Each sample: the 5 scenes are composed in a
RANDOM left-right order, the model coordinates with 5 identical copies on ONE
scene (chosen by NAME, so position is controlled). Tally which scene wins.

  by-scene concentrated  -> scene focal point (they coordinate on a scene)
  by-position concentrated -> position focal point (scene would then be ~uniform)
"""
import os, io, csv, re, base64, time, random
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
load_dotenv()

STIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stimuli", "cars")
SCENES = ["sedan", "coupe", "hatch", "suv", "van", "pickup"]
N = 10  # samples per model

PROMPT = ("The image shows SIX cars in a row, numbered 1 to 6 from left to right. You are coordinating "
          "with 5 other players who are identical copies of you. Each of you independently picks ONE car — "
          "no communication. You all score points only if all 6 pick the SAME car. Which do you pick? Give "
          "one short sentence of reasoning, then end with exactly 'CHOICE: <n>' where n is 1-6.")

PANELS = {s: Image.open(os.path.join(STIM, f"{s}.png")).convert("RGB") for s in SCENES}


def compose(order):
    w = sum(PANELS[s].width for s in order); h = max(PANELS[s].height for s in order)
    comp = Image.new("RGB", (w, h), "white"); x = 0
    for s in order:
        comp.paste(PANELS[s], (x, 0)); x += PANELS[s].width
    buf = io.BytesIO(); comp.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


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


def parse(txt):
    m = re.search(r"CHOICE:\s*([1-6])", txt)
    if m:
        return int(m.group(1)) - 1
    m = re.search(r"\b([1-6])\b", txt)
    return int(m.group(1)) - 1 if m else None


def run(name, asker, client, orders):
    rows = []
    for order in orders:
        resp = asker(client, compose(order))
        pos = parse(resp)
        scene = order[pos] if pos is not None and 0 <= pos < len(order) else "?"
        rows.append(dict(model=name, order="|".join(order), chosen=scene, position=pos,
                         reasoning=resp.replace("\n", " ").strip()[:400]))
        print(f"  {name} order={order} -> {scene} (pos {pos})")
    return rows


# fixed set of random orders (same across models for fair comparison); seedless shuffle
orders = []
pool = list(SCENES)
for _ in range(N):
    random.shuffle(pool); orders.append(list(pool))

all_rows = []
import anthropic
all_rows += run("claude-opus-4-8", ask_claude, anthropic.Anthropic(max_retries=5), orders)
from openai import OpenAI
all_rows += run("gpt-5.4", ask_gpt, OpenAI(), orders)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
fn = f"car_probe_{ts}.csv"
with open(fn, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "order", "chosen", "position", "reasoning"])
    w.writeheader(); w.writerows(all_rows)

from collections import Counter
print("\n=== TALLY ===")
for m in ("claude-opus-4-8", "gpt-5.4"):
    rs = [r for r in all_rows if r["model"] == m]
    print(f"{m}")
    print(f"  by-scene:    {dict(Counter(r['chosen'] for r in rs))}")
    print(f"  by-position: {dict(Counter(r['position'] for r in rs))}")
print(f"\nsaved {fn}")
