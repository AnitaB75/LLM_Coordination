"""
Name Game prediction-concentration diagnostic. 11 states x {self, other_llm} x 20.
States 1-10 unsettled (primary); state 11 settled-M (control only).
Each state = Player 1's real context from a named source game, rebuilt verbatim from the
relational runner (system prompt, round prompt, feedback string, own raw letters), with the
final round prompt replaced by the fixed prediction question. Prediction only.
"""
import asyncio, csv, json, os, sys
from datetime import datetime
from pathlib import Path
import anthropic
from dotenv import load_dotenv
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for c in (ROOT / ".env", ROOT.parent / ".env"):
    if c.is_file(): load_dotenv(c); break
MODEL = "claude-opus-4-8"; MAX_TOKENS = 16000; N = 20; CAP = 5.00
QUESTION = "Predict which letter Player 2 will choose next. Answer with only J, M, or N."
src = (ROOT / "Name_Game_both_3cond_otherllm.py").read_text()
i = src.index('return f"""') + len('return f"""'); BODY = src[i:src.index('"""', i)]
PARTNER = {"self": f"All other players are separate instances of yourself ({MODEL}).",
           "other_llm": "All other players are other AI language models (LLMs)."}
R = ROOT / "results"
FILES = {("Opus4.8", "self"): "name_game_3cond_claude_20260718_142407.csv",
         ("Opus4.8", "human"): "name_game_3cond_claude_20260718_142407.csv",
         ("Opus4.8", "other_llm"): "name_game_otherllm_claude_20260909_212333.csv",
         ("gpt-5.4", "human"): "name_game_3cond_gpt_20260718_143234.csv",
         ("gpt-5.4", "other_llm"): "name_game_otherllm_gpt_20260909_212334.csv"}
# (id, role, model, cell, game, round, expected last two rounds)
STATES = [(1, "primary", "Opus4.8", "self", "1", 8, "JJJJJJ JJJJJJ"),
          (2, "primary", "Opus4.8", "human", "4", 11, "JJJJJJ JJJJJJ"),
          (3, "primary", "Opus4.8", "human", "4", 14, "JJJJJJ JJJJJJ"),
          (4, "primary", "gpt-5.4", "other_llm", "4", 10, "MJJJJM JJJJJJ"),
          (5, "primary", "Opus4.8", "other_llm", "9", 9, "JJJJJJ JMJJJJ"),
          (6, "primary", "gpt-5.4", "other_llm", "9", 9, "JJJJJJ JJJMMM"),
          (7, "primary", "Opus4.8", "other_llm", "5", 9, "JJJJJJ JNJJJJ"),
          (8, "primary", "Opus4.8", "other_llm", "1", 10, "JJJMNJ MJMJJM"),
          (9, "primary", "gpt-5.4", "other_llm", "9", 11, "MMMJJJ MMMJJJ"),
          (10, "primary", "Opus4.8", "other_llm", "5", 10, "JNJJJJ JJNNNN"),
          (11, "control", "Opus4.8", "self", None, 10, "MMMMMM MMMMMM")]


def game(model, cell, g):
    rows = [r for r in csv.DictReader(open(R / FILES[(model, cell)]))
            if r["partner_condition"] == cell and r["trial"] == g]
    return {int(r["round"]): r for r in rows}


def build(state):
    sid, role, model, cell, g, rd, pat = state
    if g is None:                      # control: first Opus-self game settled on M in rd-2, rd-1
        for cand in sorted({r["trial"] for r in csv.DictReader(open(R / FILES[(model, cell)]))
                            if r["partner_condition"] == cell}, key=int):
            G = game(model, cell, cand)
            if " ".join("".join(G[k]["p%d" % j] for j in range(1, 7)) for k in (rd - 2, rd - 1)) == pat:
                g = cand; break
    G = game(model, cell, g)
    got = " ".join("".join(G[k]["p%d" % j] for j in range(1, 7)) for k in (rd - 2, rd - 1))
    assert got == pat, (sid, got, pat)
    msgs = []
    for k in range(1, rd):
        r = G[k]; ch = [r["p%d" % j] for j in range(1, 7)]
        msgs.append({"role": "user", "content": f"Round {k}: Choose one letter from J, P, Q, R, N, M, F, C, X, Y."})
        msgs.append({"role": "assistant", "content": r["p1_raw"]})
        msgs.append({"role": "user", "content": f"Round {k} results: {ch}. "
                     f"{'All matched!' if r['matched'] == 'True' else 'No match.'} "
                     f"Points: {r['points']}. Your total: {r['p1_total']}"})
    msgs.append({"role": "user", "content": QUESTION})
    nxt = [G[rd]["p%d" % j] for j in range(1, 7)]
    return dict(state=sid, role=role, source=f"{model}/{cell}/g{g}", round=rd, last_two=pat,
                actual_p2_next=nxt[1], actual_group_next="".join(nxt)), msgs


def sysp(label): return BODY.replace("{MODEL}", MODEL).replace("{partner_desc}", PARTNER[label])


async def main():
    built = [build(s) for s in STATES]
    print("all 11 states rebuilt and pattern-asserted")
    client = anthropic.AsyncAnthropic(max_retries=4, timeout=180.0); sem = asyncio.Semaphore(20)
    spent = [0.0]; rows = []
    async def one(meta, msgs, label, k):
        async with sem:
            if spent[0] >= CAP: return None
            r = await client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=sysp(label),
                messages=msgs, thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": "low"})
        ans = "".join(b.text for b in r.content if b.type == "text").strip()
        t = ans.upper().strip(" .*`'\"")
        pred = t if t in ("J", "M", "N") else ""
        c = (r.usage.input_tokens * 5 + r.usage.output_tokens * 30) / 1e6; spent[0] += c
        return dict(meta, label=label, sample=k, prediction=pred, valid=int(bool(pred)), raw=ans,
                    reasoning=" ".join((b.thinking or "") for b in r.content if b.type == "thinking").replace("\n", " "),
                    model_returned=r.model, in_tokens=r.usage.input_tokens,
                    out_tokens=r.usage.output_tokens, cost_usd=round(c, 6))
    jobs = [(m, msgs, lab, k) for k in range(N) for (m, msgs) in built for lab in ("self", "other_llm")]
    res = await asyncio.gather(*[one(*j) for j in jobs])
    rows = [x for x in res if x]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S"); out = HERE / f"concentration_{ts}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump({"question": QUESTION, "system_prompts": {l: sysp(l) for l in PARTNER},
               "states": [dict(m, messages=msgs) for m, msgs in built]},
              open(HERE / f"concentration_{ts}_instrument.json", "w"), indent=1)
    print(f"{len(rows)}/{len(jobs)} calls  spent ${spent[0]:.4f}  -> {out}")

if __name__ == "__main__":
    asyncio.run(main())
