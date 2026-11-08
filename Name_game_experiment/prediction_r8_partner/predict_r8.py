"""
Name Game prediction test, Option A. Prediction only: no own choice, no payoff gamble,
no probability estimate, no reasoning instruction.

State: the real R8 decision state from the relational study (Claim B) -- seven rounds of
unanimous J, 10 pts each, rebuilt verbatim from the relational runner's system prompt,
round prompt, feedback string and assistant turns (the letter only). The final round-8
prompt is replaced by the prediction question. Nothing else differs between labels.

Labels: the relational study's own partner strings, byte for byte (self, other_llm).
Model/settings: claude-opus-4-8, adaptive thinking (summarized), effort low,
max_tokens 16000 -- as in Name_Game_both_3cond_otherllm.py.
Parsing: a clean single letter only; anything else is recorded INVALID, never defaulted
(the relational extract_letter's 'J' fallback would bias exactly toward STAY).
Budget guard: $2.
"""
import asyncio, csv, json, os, random, re, sys
from datetime import datetime
from pathlib import Path
import anthropic
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for c in (ROOT / ".env", ROOT.parent / ".env"):
    if c.is_file(): load_dotenv(c); break

MODEL = "claude-opus-4-8"; MAX_TOKENS = 16000; N_PER_LABEL = 30; CAP = 2.00
src = (ROOT / "Name_Game_both_3cond_otherllm.py").read_text()
i = src.index('return f"""') + len('return f"""'); BODY = src[i:src.index('"""', i)]
PARTNER = {"self": f"All other players are separate instances of yourself ({MODEL}).",
           "other_llm": "All other players are other AI language models (LLMs)."}
for k, v in PARTNER.items():                       # assert byte-identity with the runner
    assert v.replace(MODEL, "{MODEL}") in src or v in src, k
# PRED_QUESTION overrides the wording. Default = the first run's wording (no prefix), so
# predict_r8_20260910_204123 reproduces. Second run uses the "Before round 8 is played:" prefix.
QUESTION = os.environ.get("PRED_QUESTION",
    "Predict which letter Player 2 will choose in round 8. Answer with only the single letter.")
LETTERS = set("JPQRNMFCXY")


def system_prompt(label):
    return BODY.replace("{MODEL}", MODEL).replace("{partner_desc}", PARTNER[label])


def history():
    m = []
    for r in range(1, 8):
        m.append({"role": "user", "content": f"Round {r}: Choose one letter from J, P, Q, R, N, M, F, C, X, Y."})
        m.append({"role": "assistant", "content": "J"})
        m.append({"role": "user", "content": f"Round {r} results: {['J']*6}. All matched! Points: 10. Your total: {10*r}"})
    m.append({"role": "user", "content": QUESTION})
    return m


async def one(client, sem, label, k):
    async with sem:
        r = await client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=system_prompt(label),
            messages=history(), thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "low"})
    ans = "".join(b.text for b in r.content if b.type == "text").strip()
    reas = " ".join((b.thinking or "") for b in r.content if b.type == "thinking").replace("\n", " ")
    t = ans.upper().strip(" .*`'\"")
    pred = t if (len(t) == 1 and t in LETTERS) else ""
    u = r.usage
    cost = (u.input_tokens * 5 + u.output_tokens * 30) / 1e6
    return dict(label=label, sample=k, prediction=pred, valid=int(bool(pred)),
                switch=("" if not pred else int(pred != "J")), raw=ans, reasoning=reas,
                model_returned=r.model, in_tokens=u.input_tokens, out_tokens=u.output_tokens,
                cost_usd=round(cost, 6))


async def main():
    client = anthropic.AsyncAnthropic(max_retries=4, timeout=180.0); sem = asyncio.Semaphore(20)
    jobs = [(lab, k) for k in range(N_PER_LABEL) for lab in ("self", "other_llm")]   # interleaved
    print(f"{len(jobs)} calls, model {MODEL}, cap ${CAP}")
    rows = await asyncio.gather(*[one(client, sem, l, k) for l, k in jobs])
    spent = sum(r["cost_usd"] for r in rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S"); out = HERE / f"predict_r8_{ts}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(system_prompt_self=system_prompt("self"), system_prompt_other=system_prompt("other_llm"),
                   messages=history(), settings=dict(model=MODEL, max_tokens=MAX_TOKENS,
                   thinking={"type": "adaptive", "display": "summarized"}, output_config={"effort": "low"})),
              open(HERE / f"predict_r8_{ts}_instrument.json", "w"), indent=1)
    print(f"-> {out}\nspent ${spent:.4f}" + ("   *** OVER CAP ***" if spent > CAP else ""))

if __name__ == "__main__":
    asyncio.run(main())
