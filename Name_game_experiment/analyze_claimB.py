#!/usr/bin/env python3
"""
Claim B: do Opus 4.8 and gpt-5.4 behaviourally distinguish self copies, other
LLMs, and humans in the history+feedback coordination game?

PRIMARY DV (descriptive), per player trajectory:
    DV_i = (# rounds 8-14 with choice in {M,N}) / 7
PRIMARY MODEL (inferential): trial-level Bayesian mixed logistic over the
rounds 8-14 binary choices, so the 7 rounds from one player are not treated as
independent.

Cells, all from the SAME byte-identical prompt body:
  self, human  -> name_game_3cond_{claude,gpt}_20260718_*   (Jul 18)
  other_llm    -> name_game_otherllm_{claude,gpt}_*         (fresh, Sep 9)
Historical n=6 other_llm cells are EXCLUDED from primary, used only in R4.
"""

from __future__ import annotations
import glob, itertools, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

HERE = Path(__file__).resolve().parent
OUT = HERE / "claimB_output"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260909)
NBOOT = 20000
FRAMINGS = ["human", "other_llm", "self"]          # theoretical order
LABEL = {"human": "Human", "other_llm": "Other LLM", "self": "Self"}
MODELS = ["Opus 4.8", "gpt-5.4"]

def _tee(path):
    class T:
        def __init__(s): s.f = open(path, "w"); s.o = sys.stdout
        def write(s, x): s.o.write(x); s.f.write(x)
        def flush(s): s.o.flush(); s.f.flush()
    return T()
sys.stdout = _tee(OUT / "claimB_report.txt")

def head(t): print(f"\n{'='*78}\n{t}\n{'='*78}")

# ------------------------------------------------------------------ load ----
def longify(path, model_label, conds=None):
    d = pd.read_csv(path)
    pc = [f"p{i}" for i in range(1, 7)]
    idv = [c for c in ["trial", "partner_condition", "round"] if c in d.columns]
    L = d.melt(id_vars=idv, value_vars=pc, var_name="player", value_name="choice")
    L["model"] = model_label
    L["src"] = Path(path).name
    if conds: L = L[L.partner_condition.isin(conds)]
    return L

SRC = {
    ("Opus 4.8", "hist"): "results/name_game_3cond_claude_20260718_142407.csv",
    ("gpt-5.4",  "hist"): "results/name_game_3cond_gpt_20260718_143234.csv",
    ("Opus 4.8", "new"):  sorted(g for g in glob.glob(str(HERE/"results/name_game_otherllm_claude_*.csv")) if "parselog" not in g)[-1],
    ("gpt-5.4",  "new"):  sorted(g for g in glob.glob(str(HERE/"results/name_game_otherllm_gpt_*.csv")) if "parselog" not in g)[-1],
}
parts = []
for m in MODELS:
    parts.append(longify(HERE / SRC[(m, "hist")], m, conds=["self", "human"]))
    parts.append(longify(SRC[(m, "new")], m, conds=["other_llm"]))
L = pd.concat(parts, ignore_index=True)
L["mn"] = L.choice.isin(["M", "N"]).astype(int)
# unique ids: trial numbers repeat across model x framing x source
L["game_uid"]   = L.model + "|" + L.partner_condition + "|T" + L.trial.astype(str)
L["player_uid"] = L.game_uid + "|" + L.player
POST = L[L["round"].between(8, 14)].copy()

head("PART 1  PROVENANCE")
for (m, c), g in L.groupby(["model", "partner_condition"], sort=False):
    print(f"  {m:9s} {c:10s} games={g.game_uid.nunique():3d} players={g.player_uid.nunique():3d} "
          f"rounds={g['round'].min()}-{g['round'].max()}  src={g.src.unique()[0]}")

# ------------------------------------------------- PART 2  DV + VERIFY ------
head("PART 2  PRIMARY DV AND VERIFICATION OF REPORTED HISTORICAL VALUES")
dv = (POST.groupby(["model", "partner_condition", "game_uid", "player_uid"])
          .mn.mean().rename("DV").reset_index())
dv.to_csv(OUT / "player_level_DV.csv", index=False)

print("\n  Reported-vs-actual. The status-table values are ROUND-8-ONLY, not the")
print("  rounds 8-14 proportion defined as the primary DV:\n")
print(f"  {'cell':22s} {'reported':>9s} {'R8-only':>9s} {'PRIMARY DV (R8-14)':>20s}")
rep = {("Opus 4.8","self"):.967, ("Opus 4.8","human"):.000,
       ("gpt-5.4","self"):1.000, ("gpt-5.4","human"):.117}
for (m, c), v in rep.items():
    r8 = L[(L.model==m)&(L.partner_condition==c)&(L["round"]==8)].mn.mean()
    d8 = dv[(dv.model==m)&(dv.partner_condition==c)].DV.mean()
    flag = "  <-- differs" if abs(d8-v) > .005 else ""
    print(f"  {m+' / '+c:22s} {v:9.3f} {r8:9.3f} {d8:20.3f}{flag}")

def boot_ci_games(sub, n=NBOOT):
    """Resample GAMES (the independent replicate), then average player DVs."""
    gl = [g.DV.values for _, g in sub.groupby("game_uid")]
    if len(gl) < 2: return (np.nan, np.nan)
    d = np.empty(n)
    for i in range(n):
        d[i] = np.concatenate([gl[j] for j in RNG.integers(0, len(gl), len(gl))]).mean()
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

head("PART 2b  DIAGNOSTIC TABLE  (player-level DV; CI clustered on GAME)")
rows = []
print(f"  {'model':10s} {'framing':10s} {'games':>5s} {'players':>7s} {'mean':>6s} "
      f"{'SD':>6s} {'median':>6s} {'95% CI (game-clustered)':>26s} {'naive CI':>16s}")
for m in MODELS:
    for c in FRAMINGS:
        s = dv[(dv.model == m) & (dv.partner_condition == c)]
        lo, hi = boot_ci_games(s)
        se = s.DV.std(ddof=1) / np.sqrt(len(s))
        nlo, nhi = s.DV.mean() - 1.96*se, s.DV.mean() + 1.96*se
        rows.append(dict(model=m, framing=c, n_games=s.game_uid.nunique(), n_players=len(s),
                         mean=s.DV.mean(), sd=s.DV.std(ddof=1), median=s.DV.median(),
                         ci_lo=lo, ci_hi=hi, naive_lo=nlo, naive_hi=nhi))
        print(f"  {m:10s} {c:10s} {s.game_uid.nunique():5d} {len(s):7d} {s.DV.mean():6.3f} "
              f"{s.DV.std(ddof=1):6.3f} {s.DV.median():6.3f} "
              f"{'['+format(lo,'.3f')+', '+format(hi,'.3f')+']':>26s} "
              f"{'['+format(max(0,nlo),'.2f')+','+format(min(1,nhi),'.2f')+']':>16s}")
pd.DataFrame(rows).to_csv(OUT / "descriptive_table.csv", index=False)

head("PART 2c  DEPENDENCE STRUCTURE  (why n = games, not players)")
print("  Players inside a game SEE each other's choices (feedback is on), so they")
print("  are not independent. ICC of player-level DV within game:\n")
for m in MODELS:
    for c in FRAMINGS:
        s = dv[(dv.model==m)&(dv.partner_condition==c)]
        gm = s.groupby("game_uid").DV.mean(); k = 6
        msb = k*((gm - s.DV.mean())**2).sum()/(len(gm)-1)
        msw = ((s.DV - s.groupby("game_uid").DV.transform("mean"))**2).sum()/(len(s)-len(gm))
        icc = (msb-msw)/(msb+(k-1)*msw) if (msb+(k-1)*msw) > 0 else np.nan
        print(f"  {m:10s} {c:10s} ICC={icc:6.3f}   game means: {np.round(gm.values,2)}")
print("\n  => the independent replicate is the GAME (n=10/cell). 60 player")
print("     trajectories are ~6 near-copies of 10 game outcomes.")

# ------------------------------------------- PART 3  PRIMARY MODEL ----------
head("PART 3  PRIMARY INFERENTIAL MODEL")
POST["cell"] = POST.model + ":" + POST.partner_condition
CELLS = [f"{m}:{c}" for m in MODELS for c in FRAMINGS]
POST["cell"] = pd.Categorical(POST.cell, categories=CELLS, ordered=True)
print(f"  trial-level rows (rounds 8-14): {len(POST)}  "
      f"= 2 models x 3 framings x 10 games x 6 players x 7 rounds")

print("\n  3a. SEPARATION DIAGNOSIS (conventional logistic, no random effects)")
try:
    g = smf.glm("mn ~ 0 + C(cell)", data=POST, family=sm.families.Binomial()).fit()
    print(f"      {'cell':22s} {'coef (logit)':>13s} {'SE':>10s}")
    for nm, b, se in zip(g.params.index, g.params.values, g.bse.values):
        lab = nm.replace("C(cell)[", "").replace("]", "")
        warn = "  <-- separated" if abs(b) > 10 or se > 10 else ""
        print(f"      {lab:22s} {b:13.2f} {se:10.2f}{warn}")
except Exception as e:
    print("      GLM failed:", e)
print("\n      Opus/human is 0/420 and gpt-5.4/self is 420/420: COMPLETE separation.")
print("      Conventional ML coefficients are therefore unbounded and their SEs")
print("      meaningless. We do NOT report them as inference.")

print("\n  3b. RANDOM-EFFECTS STRUCTURE")
print("      Rounds are nested in players, players in games. ICC(player|game) is")
print("      0.92-0.99 wherever there is variance, so game is the dominant level.")
print("      We fit random intercepts for GAME and for PLAYER-within-game; where")
print("      a cell has no variance the player term is weakly identified and the")
print("      prior carries it (that is the intended, honest behaviour).")

print("\n  3c. BAYESIAN MIXED LOGISTIC (statsmodels BinomialBayesMixedGLM, VB)")
print("      mn ~ 0 + C(cell)  +  (1|game) + (1|player)")
print("      Weakly informative N(0, 2.5) priors on the cell logits (fe_p=2.5),")
print("      which is what keeps the separated cells finite.")
vc = {"game": "0 + C(game_uid)", "player": "0 + C(player_uid)"}
bm = BinomialBayesMixedGLM.from_formula("mn ~ 0 + C(cell)", vc, POST, fe_p=2.5, vcp_p=1.0)
res = bm.fit_vb(verbose=False)
fe_names = [n.replace("C(cell)[", "").replace("]", "") for n in bm.exog_names[:len(CELLS)]]
fe_mean = np.asarray(res.fe_mean); fe_sd = np.asarray(res.fe_sd)
print(f"\n      {'cell':22s} {'post. mean':>11s} {'post. SD':>9s} {'P(M/N) at RE=0':>16s}")
for nm, mu, sd in zip(fe_names, fe_mean, fe_sd):
    print(f"      {nm:22s} {mu:11.3f} {sd:9.3f} {1/(1+np.exp(-mu)):16.3f}")
try:
    _vn = list(res.model.vcp_names)
except Exception:
    _vn = ["game", "player"]
print("\n      random-effect SDs (posterior means): " +
      ", ".join(f"{n}={v:.2f}" for n, v in zip(_vn, np.exp(np.asarray(res.vcp_mean)))))
print("\n      CAVEAT: mean-field VB is known to UNDERSTATE posterior variance, and")
print("      with ICC ~0.92-0.99 the 420 trial rows per cell carry roughly 10")
print("      independent games' worth of information. The trial-level CrIs below")
print("      are therefore anti-conservative; where they disagree with the exact")
print("      game-level permutation test, the permutation test governs.")

# posterior draws (mean-field VB => independent normals)
NS = 40000
draws = RNG.normal(fe_mean, fe_sd, size=(NS, len(fe_mean)))
idx = {n: i for i, n in enumerate(fe_names)}

# ------------------------------------------- PART 4  CONTRASTS --------------
head("PART 4  PLANNED CONTRASTS  (3 per model, Holm-adjusted within model)")

def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj

def perm_exact(a, b):
    """Exact two-sided permutation test on GAME-level means."""
    obs = a.mean() - b.mean(); pool = np.concatenate([a, b]); n = len(a)
    cnt = tot = 0
    for combo in itertools.combinations(range(len(pool)), n):
        m = np.zeros(len(pool), bool); m[list(combo)] = True
        d = pool[m].mean() - pool[~m].mean()
        tot += 1; cnt += abs(d) >= abs(obs) - 1e-12
    return obs, cnt / tot

PAIRS = [("self", "human"), ("self", "other_llm"), ("other_llm", "human")]
contrast_rows = []
for m in MODELS:
    print(f"\n  --- {m} ---")
    raw_p, recs = [], []
    for a, b in PAIRS:
        ca, cb = f"{m}:{a}", f"{m}:{b}"
        da, db = draws[:, idx[ca]], draws[:, idx[cb]]
        pa, pb = 1/(1+np.exp(-da)), 1/(1+np.exp(-db))
        dprob = pa - pb
        lor = da - db
        ptail = 2*min((lor <= 0).mean(), (lor >= 0).mean())
        ga = dv[(dv.model==m)&(dv.partner_condition==a)].groupby("game_uid").DV.mean().values
        gb = dv[(dv.model==m)&(dv.partner_condition==b)].groupby("game_uid").DV.mean().values
        obs, pperm = perm_exact(ga, gb)
        raw_p.append(pperm)
        recs.append(dict(model=m, contrast=f"{a} - {b}",
            p_a=pa.mean(), p_b=pb.mean(),
            d_prob=dprob.mean(), d_lo=np.percentile(dprob,2.5), d_hi=np.percentile(dprob,97.5),
            or_med=np.exp(np.median(lor)), or_lo=np.exp(np.percentile(lor,2.5)),
            or_hi=np.exp(np.percentile(lor,97.5)), p_post=max(ptail,1/NS),
            gm_diff=obs, p_perm=pperm))
    adj = holm(np.array(raw_p))
    for r, pa_ in zip(recs, adj):
        r["p_perm_holm"] = pa_; contrast_rows.append(r)
        print(f"    {r['contrast']:22s} P(model-implied): {r['p_a']:.3f} vs {r['p_b']:.3f}")
        print(f"      {'diff in prob':16s} {r['d_prob']:+.3f}  [{r['d_lo']:+.3f}, {r['d_hi']:+.3f}]  (95% CrI)")
        orstr = (f"{r['or_med']:.1f} [{r['or_lo']:.1f}, {r['or_hi']:.1f}]"
                 if r['or_hi'] < 1e6 else "unstable (separation)")
        print(f"      {'odds ratio':16s} {orstr}")
        print(f"      {'game-level diff':16s} {r['gm_diff']:+.3f}   exact perm p={r['p_perm']:.4f}"
              f"   Holm-adj p={pa_:.4f}")
pd.DataFrame(contrast_rows).to_csv(OUT / "planned_contrasts.csv", index=False)

head("PART 4b  FRAMING x MODEL INTERACTION")
for a, b in [("self", "human"), ("other_llm", "human"), ("self", "other_llm")]:
    d1 = draws[:, idx[f"Opus 4.8:{a}"]] - draws[:, idx[f"Opus 4.8:{b}"]]
    d2 = draws[:, idx[f"gpt-5.4:{a}"]]  - draws[:, idx[f"gpt-5.4:{b}"]]
    inter = d1 - d2
    pt = 2*min((inter <= 0).mean(), (inter >= 0).mean())
    print(f"  ({a} - {b}) Opus  minus  ({a} - {b}) gpt-5.4 :  "
          f"{np.median(inter):+.2f} logits [{np.percentile(inter,2.5):+.2f}, "
          f"{np.percentile(inter,97.5):+.2f}]  P={max(pt,1/NS):.4f}")
# game-level 2x2 interaction on the DV scale
def gmeans(m, c): return dv[(dv.model==m)&(dv.partner_condition==c)].groupby("game_uid").DV.mean().values
for a, b in [("self","human"), ("other_llm","human")]:
    i_o = gmeans("Opus 4.8",a).mean()-gmeans("Opus 4.8",b).mean()
    i_g = gmeans("gpt-5.4",a).mean()-gmeans("gpt-5.4",b).mean()
    print(f"  game-level DV scale: ({a}-{b}) Opus {i_o:+.3f} vs gpt-5.4 {i_g:+.3f} "
          f"-> interaction {i_o-i_g:+.3f}")

head("PART 4c  EQUIVALENCE (only where a null drives an interpretation)")
ga = gmeans("gpt-5.4","other_llm"); gb = gmeans("gpt-5.4","human")
margin = 0.20
t1 = stats.ttest_ind(ga, gb - margin, alternative="greater")
t2 = stats.ttest_ind(ga, gb + margin, alternative="less")
print(f"  gpt-5.4 other_llm vs human, TOST on game means, margin +/-{margin:.2f} DV units:")
print(f"    observed diff {ga.mean()-gb.mean():+.3f};  p_lower={t1.pvalue:.3f}  p_upper={t2.pvalue:.3f}"
      f"  -> {'EQUIVALENT within margin' if max(t1.pvalue,t2.pvalue)<.05 else 'NOT established as equivalent'}")
print(f"  NOTE: the +/-{margin:.2f} margin is a judgement call, not derived from theory.")
print("  Where TOST fails, the correct statement is 'the data do not establish a")
print("  difference', NOT 'the conditions are equivalent'.")

head("PART 4d  WHY THE TWO INFERENCES DISAGREE FOR gpt-5.4 other_llm vs human")
for c in ["human", "other_llm"]:
    gm = gmeans("gpt-5.4", c)
    print(f"  gpt-5.4 {c:10s} game means sorted: {np.round(np.sort(gm),2)}  "
          f"mean={gm.mean():.3f} median={np.median(gm):.3f}")
print("\n  The two cells differ in SHAPE, not level: human has 4 games that fully")
print("  tipped (>=.71) and 6 at ~0, other_llm has 2 high and 8 low-but-nonzero.")
print("  The mixed model reports a CONDITIONAL (median-game) probability, which")
print("  favours other_llm; the game-level MEAN favours human. Both are correct")
print("  answers to different questions, and with n=10 games neither is")
print("  distinguishable from no difference (exact perm p=.75).")
print("  CONCLUSION: for gpt-5.4 we report NO established other_llm/human difference.")

# ------------------------------------------- PART 5  ROBUSTNESS -------------
head("PART 5  ROBUSTNESS / DESCRIPTIVES  (secondary)")

print("\n  R1. Round-by-round P(M or N), rounds 8-14")
rr = (POST.groupby(["model", "partner_condition", "round"]).mn.mean().unstack())
print(rr.round(3).to_string())
rr.to_csv(OUT / "round_by_round.csv")

print("\n  R2. Distribution of player-level DV (counts by value)")
for m in MODELS:
    for c in FRAMINGS:
        s = dv[(dv.model==m)&(dv.partner_condition==c)].DV
        vc_ = s.round(3).value_counts().sort_index()
        print(f"    {m:10s} {c:10s} " + "  ".join(f"{k:.2f}x{v}" for k, v in vc_.items()))

print("\n  R3. Game-level means (is any cell driven by one game?)")
for m in MODELS:
    for c in FRAMINGS:
        gm = gmeans(m, c)
        drop = [round(np.delete(gm, i).mean(), 3) for i in range(len(gm))]
        print(f"    {m:10s} {c:10s} mean={gm.mean():.3f}  leave-one-out range="
              f"[{min(drop):.3f}, {max(drop):.3f}]  n_games_at_0={(gm==0).sum()}")

print("\n  R4. Including the EXCLUDED historical n=6 other_llm cells")
hist_other = []
for m, f, in [("Opus 4.8", HERE/"results/name_game_claude_3cond_20260717_153006.csv"),
              ("gpt-5.4", sorted(glob.glob(str(HERE.parent/"**/name_game_LLM_feedback_MN_20260507_134935.csv"), recursive=True))[0])]:
    h = longify(f, m, conds=["other_llm"])
    h["mn"] = h.choice.isin(["M","N"]).astype(int)
    hp = h[h["round"].between(8,14)]
    if not len(hp): continue
    hdv = hp.groupby(["trial","player"]).mn.mean()
    fresh = dv[(dv.model==m)&(dv.partner_condition=="other_llm")].DV
    pooled = (hdv.sum() + fresh.sum()) / (len(hdv) + len(fresh))
    print(f"    {m:10s} historical n={len(hdv)} DV={hdv.mean():.3f} | "
          f"fresh n={len(fresh)} DV={fresh.mean():.3f} | pooled DV={pooled:.3f}")
    hist_other.append((m, hdv.mean(), fresh.mean(), pooled))
print("    The Opus historical cell (1 game, DV=0.000) is one of the ~30% of games")
print("    that never tip; it is not representative. Pooling would bias the cell")
print("    downward. Conclusions are unchanged in direction either way.")

print("\n  R5. M vs N among switchers (primary DV collapses them; descriptive only)")
for m in MODELS:
    for c in FRAMINGS:
        s = POST[(POST.model==m)&(POST.partner_condition==c)]
        sw = s[s.choice.isin(["M","N"])].choice.value_counts().to_dict()
        print(f"    {m:10s} {c:10s} {sw if sw else '(no switches)'}")

print("\n  R6. Choices outside {J, M, N} in rounds 8-14")
oth = POST[~POST.choice.isin(["J","M","N"])]
if len(oth):
    print(f"    {len(oth)} rows: {oth.choice.value_counts().to_dict()}")
    print("    HANDLING: these count as 0 in the DV (not M/N), i.e. they are treated")
    print("    as 'did not adopt the new convention'. They are NOT dropped.")
else:
    print("    NONE. Every post-change choice in all six cells is J, M or N,")
    print("    so the 0/1 coding drops no data.")
print(f"\n  extract_letter fallback log: ", end="")
pl_ = sorted(glob.glob(str(HERE/"results/name_game_otherllm_*_parselog.csv")))
if pl_:
    pdl = pd.concat([pd.read_csv(x) for x in pl_])
    print(f"{len(pdl)} non-clean response(s) in the NEW runs: "
          f"{pdl.path.value_counts().to_dict()}; all resolved to a valid letter.")
else:
    print("empty (all responses were clean single letters).")

# ------------------------------------------- PART 6  FIGURE -----------------
import matplotlib.pyplot as plt
plt.style.use("academic")
head("PART 6  FIGURE")
fig, ax = plt.subplots(figsize=(5.5, 3.2))
x = np.arange(len(FRAMINGS))
STY = {"Opus 4.8": dict(marker="o", mfc="black", ls="-",  color="black"),
       "gpt-5.4":  dict(marker="s", mfc="white", ls="--", color="black")}
tab = pd.DataFrame(rows)
for m in MODELS:
    s = tab[tab.model == m].set_index("framing").reindex(FRAMINGS)
    st = STY[m]
    for j, c in enumerate(FRAMINGS):          # player-level points, jittered
        pts = dv[(dv.model==m)&(dv.partner_condition==c)].DV.values
        off = (-0.10 if m == "Opus 4.8" else 0.10)
        ax.scatter(x[j] + off + RNG.normal(0, 0.022, len(pts)), pts,
                   s=3.5, color=st["color"], alpha=0.16, linewidths=0, zorder=1)
    off = -0.10 if m == "Opus 4.8" else 0.10
    ax.errorbar(x + off, s["mean"],
                yerr=[np.clip(s["mean"]-s["ci_lo"],0,None), np.clip(s["ci_hi"]-s["mean"],0,None)],
                marker=st["marker"], mfc=st["mfc"], mec=st["color"], color=st["color"],
                ls=st["ls"], ms=5, lw=1.0, capsize=2.5, capthick=0.8, elinewidth=0.8,
                zorder=3, label=m)
ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in FRAMINGS])
ax.set_xlim(-0.45, len(FRAMINGS)-0.55); ax.set_ylim(-0.04, 1.06)
ax.set_xlabel("Partner framing"); ax.set_ylabel("Proportion of rounds 8-14\nchoosing M or N")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure_claimB.{ext}", dpi=300, bbox_inches="tight")
print(f"  -> {OUT/'figure_claimB.{pdf,png}'}")
print(f"\n  Tables + report -> {OUT}")
