"""Step 1 figure: SELF framing changes round-8 choice (A) and Claude's reasoning (B).
Data: relational Name Game (Claim B) result files + validated R8 trace codes.
CIs: 95% percentile bootstrap resampling GAMES (players in a game converge; ICC .92-.99)."""
import csv, os, random, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("academic")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.expanduser("~/Downloads/self-coordination-figures"); os.makedirs(OUT, exist_ok=True)
FR = ["self", "other_llm", "human"]; FL = {"self": "SELF", "other_llm": "OTHER_LLM", "human": "HUMAN"}
SRC = {("Claude Opus 4.8", "self"): "name_game_3cond_claude_20260718_142407.csv",
       ("Claude Opus 4.8", "human"): "name_game_3cond_claude_20260718_142407.csv",
       ("Claude Opus 4.8", "other_llm"): "name_game_otherllm_claude_20260909_212333.csv",
       ("GPT-5.4", "self"): "name_game_3cond_gpt_20260718_143234.csv",
       ("GPT-5.4", "human"): "name_game_3cond_gpt_20260718_143234.csv",
       ("GPT-5.4", "other_llm"): "name_game_otherllm_gpt_20260909_212334.csv"}


def cluster_ci(games, B=10000, seed=11):
    """games: list of per-game (k, n). Returns pooled proportion and bootstrap CI over games."""
    k = sum(a for a, _ in games); n = sum(b for _, b in games); rng = random.Random(seed); bs = []
    for _ in range(B):
        s = [games[rng.randrange(len(games))] for _ in games]
        bs.append(sum(a for a, _ in s) / sum(b for _, b in s))
    bs.sort(); return k / n, bs[int(.025 * B)], bs[int(.975 * B)], k, n, len(games)


rows = []
A = {}
for (model, fr), f in SRC.items():
    G = collections.defaultdict(lambda: [0, 0, 0])
    for r in csv.DictReader(open(os.path.join(ROOT, "results", f))):
        if r["partner_condition"] == fr and r["round"] == "8":
            for j in range(1, 7):
                c = r["p%d" % j]; G[r["trial"]][0] += c == "M"; G[r["trial"]][1] += 1; G[r["trial"]][2] += c == "N"
    p, lo, hi, k, n, g = cluster_ci([(a, b) for a, b, _ in G.values()])
    nN = sum(v[2] for v in G.values()); A[(model, fr)] = (p, lo, hi)
    rows.append(dict(panel="A", model=model, framing=FL[fr], measure="P(choose M) at R8", value=round(p, 4),
                     ci_lo=round(lo, 4), ci_hi=round(hi, 4), k=k, n=n, n_games=g,
                     note=f"N chosen by {nN}/{n}" if nN else ""))
TC = list(csv.DictReader(open(os.path.join(ROOT, "prediction_r8_partner", "r8_play_trace_codes.csv"))))
CODES = [("identical", "Reasons about identical partner"), ("tiebreak", "Uses alphabetical rule"),
         ("empirical", "Empirical partner prediction")]
Bv = {}
for fr in FR:
    x = [t for t in TC if t["framing"] == FL[fr]]
    avail = [t for t in x if t["trace"].strip()]
    for key, lab in CODES:
        G = collections.defaultdict(lambda: [0, 0])
        for t in avail: G[t["game"]][0] += int(t[key]); G[t["game"]][1] += 1
        p, lo, hi, k, n, g = cluster_ci(list(G.values()))
        Bv[(fr, key)] = (p, lo, hi)
        rows.append(dict(panel="B", model="Claude Opus 4.8", framing=FL[fr], measure=lab, value=round(p, 4),
                         ci_lo=round(lo, 4), ci_hi=round(hi, 4), k=k, n=n, n_games=g,
                         note=f"traces available {len(avail)}/{len(x)}"))
# assertions against numbers reported earlier
exp = {("Claude Opus 4.8", "self"): 58 / 60, ("Claude Opus 4.8", "other_llm"): 7 / 60, ("Claude Opus 4.8", "human"): 0,
       ("GPT-5.4", "self"): 1.0, ("GPT-5.4", "other_llm"): 21 / 60, ("GPT-5.4", "human"): 7 / 60}
for k_, v in exp.items(): assert abs(A[k_][0] - v) < 1e-9, (k_, A[k_][0], v)
for (fr, key), v in {("self", "identical"): .7667, ("self", "tiebreak"): .9333, ("other_llm", "empirical"): .1833,
                     ("human", "empirical"): .3833}.items():
    assert abs(Bv[(fr, key)][0] - v) < 5e-4, (fr, key, Bv[(fr, key)][0])
with open(os.path.join(OUT, "step1_figure_data.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.0, 3.3), gridspec_kw=dict(width_ratios=[1, 1.35]))
fig.subplots_adjust(wspace=.45, bottom=.30, top=.88)
x = np.arange(3)
for off, model, fc, hatch in ((-0.19, "Claude Opus 4.8", "#3A4152", None), (0.19, "GPT-5.4", "white", "///")):
    v = np.array([A[(model, f)] for f in FR])
    axA.bar(x + off, v[:, 0], .36, color=fc, edgecolor="black", linewidth=.8, hatch=hatch, label=model)
    axA.errorbar(x + off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor="black", elinewidth=.8, capsize=2.5)
for off, model in ((-0.19, "Claude Opus 4.8"), (0.19, "GPT-5.4")):
    for xi, f_ in zip(x, FR):
        if A[(model, f_)][0] == 0: axA.text(xi + off, .012, "0", ha="center", va="bottom", fontsize=7)
axA.set_xticks(x); axA.set_xticklabels([FL[f] for f in FR]); axA.set_ylim(0, 1.05)
axA.set_ylabel("Proportion choosing M at round 8"); axA.set_xlabel("Partner framing")
axA.legend(frameon=False, loc="upper right", handlelength=1.6)
axA.set_title(r"$\bf{A}$   Identity framing shapes switching behavior", fontsize=9, pad=8, loc="left")
styles = [("#3A4152", None), ("#9AA0AB", None), ("white", "///")]
for i, ((key, lab), (fc, hatch)) in enumerate(zip(CODES, styles)):
    v = np.array([Bv[(f, key)] for f in FR]); off = (i - 1) * .26
    axB.bar(x + off, v[:, 0] * 100, .24, color=fc, edgecolor="black", linewidth=.8, hatch=hatch, label=lab)
    axB.errorbar(x + off, v[:, 0] * 100, yerr=[(v[:, 0] - v[:, 1]) * 100, (v[:, 2] - v[:, 0]) * 100], fmt="none", ecolor="black", elinewidth=.8, capsize=2)
for i, (key, _) in enumerate(CODES):
    for xi, f_ in zip(x, FR):
        if Bv[(f_, key)][0] == 0: axB.text(xi + (i - 1) * .26, 1.2, "0", ha="center", va="bottom", fontsize=7)
axB.set_xticks(x); axB.set_xticklabels([FL[f] for f in FR]); axB.set_ylim(0, 105)
axB.set_ylabel("Round-8 traces with code (%)"); axB.set_xlabel("Partner framing")
axB.legend(frameon=False, loc="upper right", fontsize=7.5, handlelength=1.4)
axB.set_title(r"$\bf{B}$   Reasoning about identity", fontsize=9, pad=8, loc="left")
fig.text(.5, .02, "Relational Name Game, round 8 (first round M and N pay 15). A: n = 60 players / 10 games per bar. "
         "B: Claude Opus 4.8 reasoning traces, 60/60 available per framing.\nError bars: 95% bootstrap CIs resampling games. "
         "Opus OTHER_LLM also chose N on 2/60 (not shown as M).", ha="center", fontsize=7, style="italic")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"step1_selfframing_choice_and_reasoning.{ext}"), dpi=300, bbox_inches="tight")
print("assertions passed; saved to", OUT)
for r in rows: print(r)
