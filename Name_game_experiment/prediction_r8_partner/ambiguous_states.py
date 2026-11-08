"""
Constructed M/N-ambiguous Name Game states. 5 designs x M<->N mirror + real R8 anchor.
Rounds 1-7 are the real all-J history (10 pts each). Rounds 8+ are constructed, rendered in
the exact relational feedback format (non-match = 0 pts). P1 = predictor seat, P2 = target.
Prediction only; fixed question; relational partner wording; Opus 4.8 adaptive/low.
"""
import asyncio, csv, json
from datetime import datetime
from pathlib import Path
import anthropic
from dotenv import load_dotenv
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for c in (ROOT / ".env", ROOT.parent / ".env"):
    if c.is_file(): load_dotenv(c); break
MODEL = "claude-opus-4-8"; MAX_TOKENS = 16000; N = 20; CAP = 4.00
QUESTION = "Predict which letter Player 2 will choose next. Answer with only J, M, or N."
src = (ROOT / "Name_Game_both_3cond_otherllm.py").read_text()
i = src.index('return f"""') + len('return f"""'); BODY = src[i:src.index('"""', i)]
PARTNER = {"self": f"All other players are separate instances of yourself ({MODEL}).",
           "other_llm": "All other players are other AI language models (LLMs)."}
DESIGNS = {1: ["MMMNNN"], 2: ["MNMNMN"], 3: ["MMJNNM"],
           4: ["MMMNNN", "NNNMMM"], 5: ["MMMNNN", "MNMNMN", "NMNMNM"]}
swap = lambda s: s.translate(str.maketrans("MN", "NM"))


def ambiguous(rounds):
    last = rounds[-1]; w = "".join(rounds[-2:])
    return (len(set(last)) > 1 and abs(last.count("M") - last.count("N")) <= 1
            and abs(w.count("M") - w.count("N")) <= 1 and last.count("J") < max(last.count("M"), last.count("N")))


def messages(post):
    m = []; tot = 0
    for r in range(1, 8):
        tot += 10
        m += [{"role": "user", "content": f"Round {r}: Choose one letter from J, P, Q, R, N, M, F, C, X, Y."},
              {"role": "assistant", "content": "J"},
              {"role": "user", "content": f"Round {r} results: {['J']*6}. All matched! Points: 10. Your total: {tot}"}]
    for k, row in enumerate(post, start=8):
        m += [{"role": "user", "content": f"Round {k}: Choose one letter from J, P, Q, R, N, M, F, C, X, Y."},
              {"role": "assistant", "content": row[0]},
              {"role": "user", "content": f"Round {k} results: {list(row)}. No match. Points: 0. Your total: {tot}"}]
    m.append({"role": "user", "content": QUESTION})
    return m


STATES = [dict(state="R8", design=0, mirror="real", post=[])]
for d, rows in DESIGNS.items():
    for mir, rr in (("a", rows), ("b", [swap(x) for x in rows])):
        assert ambiguous(rr), (d, mir, rr)
        STATES.append(dict(state=f"{d}{mir}", design=d, mirror=mir, post=rr))
for d in DESIGNS:
    a = [s for s in STATES if s["design"] == d and s["mirror"] == "a"][0]["post"]
    b = [s for s in STATES if s["design"] == d and s["mirror"] == "b"][0]["post"]
    assert b == [swap(x) for x in a] and a != b


def sysp(l): return BODY.replace("{MODEL}", MODEL).replace("{partner_desc}", PARTNER[l])


async def main():
    print("%d states built; all constructed states pass the ambiguity rule; mirrors verified" % len(STATES))
    client = anthropic.AsyncAnthropic(max_retries=4, timeout=180.0); sem = asyncio.Semaphore(20); spent = [0.0]
    async def one(st, label, k):
        async with sem:
            if spent[0] >= CAP: return None
            r = await client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=sysp(label),
                messages=messages(st["post"]), thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": "low"})
        ans = "".join(b.text for b in r.content if b.type == "text").strip()
        t = ans.upper().strip(" .*`'\""); pred = t if t in ("J", "M", "N") else ""
        c = (r.usage.input_tokens * 5 + r.usage.output_tokens * 30) / 1e6; spent[0] += c
        return dict(state=st["state"], design=st["design"], mirror=st["mirror"],
                    history_post_r7=" ".join(st["post"]), p2_last=(st["post"][-1][1] if st["post"] else "J"),
                    label=label, sample=k, prediction=pred, valid=int(bool(pred)), raw=ans,
                    reasoning=" ".join((b.thinking or "") for b in r.content if b.type == "thinking").replace("\n", " "),
                    model_returned=r.model, in_tokens=r.usage.input_tokens, out_tokens=r.usage.output_tokens,
                    cost_usd=round(c, 6))
    jobs = [(s, l, k) for k in range(N) for s in STATES for l in ("self", "other_llm")]
    rows = [x for x in await asyncio.gather(*[one(*j) for j in jobs]) if x]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S"); out = HERE / f"ambiguous_{ts}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump({"question": QUESTION, "system_prompts": {l: sysp(l) for l in PARTNER},
               "states": [dict(s, messages=messages(s["post"])) for s in STATES]},
              open(HERE / f"ambiguous_{ts}_instrument.json", "w"), indent=1)
    print(f"{len(rows)}/{len(jobs)} calls  spent ${spent[0]:.4f}  -> {out}")

if __name__ == "__main__":
    asyncio.run(main())
