# Claim B — do models behaviourally distinguish self copies, other LLMs, and humans?

History + feedback paradigm. Partner framing is the only manipulation; all six
seats are always the same model.

## 1. Provenance of every cell

| Model | Framing | Games | Players | Source file | Collected |
|---|---|---|---|---|---|
| Opus 4.8 | self | 10 | 60 | `name_game_3cond_claude_20260718_142407.csv` | Jul 18 |
| Opus 4.8 | human | 10 | 60 | `name_game_3cond_claude_20260718_142407.csv` | Jul 18 |
| Opus 4.8 | other_llm | 10 | 60 | `name_game_otherllm_claude_20260909_212333.csv` | **Sep 9 (new)** |
| gpt-5.4 | self | 10 | 60 | `name_game_3cond_gpt_20260718_143234.csv` | Jul 18 |
| gpt-5.4 | human | 10 | 60 | `name_game_3cond_gpt_20260718_143234.csv` | Jul 18 |
| gpt-5.4 | other_llm | 10 | 60 | `name_game_otherllm_gpt_20260909_212334.csv` | **Sep 9 (new)** |

Historical n=6 `other_llm` cells excluded from primary (robustness R4 only).

## 2. Prompt bodies matched

`Name_Game_both_3cond.py`, `Name_Game_Claude_3cond.py` and
`Name_Game_LLM_feedback_MN.py` have byte-identical system-prompt bodies
(verified by diff). The new runner `Name_Game_both_3cond_otherllm.py` was
diffed against `Name_Game_both_3cond.py` at run time: **identical**. The
`other_llm` string is byte-identical to `Name_Game_Claude_3cond.py:63`.

## 3. New run details

10 games x 6 players x 14 rounds per model = 840 calls each, 1,680 total.
Provider-returned model IDs: **`claude-opus-4-8`**, **`gpt-5.4-2026-03-05`**.
Deviations from the historical script, all approved and none experimental:
games/players parallelised (rounds still sequential, feedback still assembled
only after all six answer — identical information flow); provider model id and
run timestamp recorded; `extract_letter` left unchanged but instrumented.
Instrumentation found **1** non-clean response in 1,680 (`"J\n\nJ"` -> `J`).

## 4. Primary DV and the correction to the status table

`DV_i = (# rounds 8-14 with M or N) / 7`, one value per player trajectory.

**The values in the task's status table are round-8-only, not the primary DV.**

| Cell | reported | R8-only | **primary DV (R8-14)** |
|---|---|---|---|
| Opus / self | .967 | .967 | **.995** |
| Opus / human | .000 | .000 | **.000** |
| gpt-5.4 / self | 1.000 | 1.000 | **1.000** |
| gpt-5.4 / human | .117 | .117 | **.329** |

This matters: the old gpt-5.4 `other_llm` cell is .333 at R8 but .119 on the
primary DV, so the two definitions order `other_llm` vs `human` oppositely.

### Descriptive table (CIs clustered on game)

| Model | Framing | Games | Players | Mean | SD | Median | 95% CI |
|---|---|---|---|---|---|---|---|
| Opus 4.8 | human | 10 | 60 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] |
| Opus 4.8 | other_llm | 10 | 60 | 0.567 | 0.378 | 0.714 | [0.321, 0.786] |
| Opus 4.8 | self | 10 | 60 | 0.995 | 0.026 | 1.000 | [0.988, 1.000] |
| gpt-5.4 | human | 10 | 60 | 0.329 | 0.378 | 0.071 | [0.100, 0.560] |
| gpt-5.4 | other_llm | 10 | 60 | 0.279 | 0.339 | 0.143 | [0.098, 0.493] |
| gpt-5.4 | self | 10 | 60 | 1.000 | 0.000 | 1.000 | [1.000, 1.000] |

## 5. Model specification, and why n is 10 not 60

Feedback is on, so players inside a game see each other and converge as a unit.
ICC(player | game) = 0.92–0.99 in every cell with variance. **The independent
replicate is the game (n=10/cell).** The 60 player trajectories are ~6
near-copies of 10 game outcomes.

Primary model: `mn ~ 0 + C(model:framing) + (1|game) + (1|player)`, Bayesian
mixed logistic (statsmodels `BinomialBayesMixedGLM`, variational Bayes),
weakly informative N(0, 2.5) priors on cell logits. Cell-means coding is a
saturated reparameterisation of framing + model + framing×model.

## 6. Separation

Opus/human is 0/420 and gpt-5.4/self is 420/420 — **complete separation**.
Conventional ML gives |coef| ≈ 26.6 with SE ≈ 17,377. Those are not reported as
inference. The priors keep the Bayesian estimates finite.

Second diagnostic: mean-field VB understates posterior variance, and with
ICC≈0.99 each cell's 420 rows carry ~10 games of information. **Trial-level
CrIs are anti-conservative.** Exact game-level permutation tests are reported
alongside and govern where the two disagree.

## 7. Planned contrasts (Holm-adjusted within model)

**Opus 4.8**

| Contrast | Δ prob (95% CrI) | Game-level Δ | Exact perm p | Holm p |
|---|---|---|---|---|
| self − human | +0.997 [+0.992, +0.999] | +0.995 | <.0001 | **<.0001** |
| self − other_llm | +0.577 [+0.507, +0.645] | +0.429 | <.0001 | **<.0001** |
| other_llm − human | +0.420 [+0.352, +0.490] | +0.567 | .0031 | **.0031** |

**gpt-5.4**

| Contrast | Δ prob (95% CrI) | Game-level Δ | Exact perm p | Holm p |
|---|---|---|---|---|
| self − human | +0.883 [+0.848, +0.913] | +0.671 | <.0001 | **<.0001** |
| self − other_llm | +0.811 [+0.761, +0.854] | +0.721 | <.0001 | **<.0001** |
| other_llm − human | +0.073 [+0.017, +0.131] | −0.050 | .7493 | **.7493** |

The gpt-5.4 `other_llm − human` row is the one place the two inferences
disagree. The cells differ in *shape*, not level (human: 4 games fully tipped,
6 at ~0; other_llm: 2 high, 8 low-but-nonzero), so the conditional median-game
estimate and the game-level mean point opposite ways. With n=10 games neither
is distinguishable from zero. **Reported as: no established difference.**
TOST on game means (±0.20 DV margin) does **not** establish equivalence either
(p_lower=.187, p_upper=.073) — so the honest statement is "the data do not
establish a difference", not "the conditions are equivalent". The ±0.20 margin
is a judgement call, not derived from theory.

## 8. Framing × model interaction

| Interaction | logits (95% CrI) | P |
|---|---|---|
| (self − human): Opus − gpt | +3.70 [+0.79, +6.62] | .013 |
| (other_llm − human): Opus − gpt | +6.41 [+4.59, +8.23] | <.0001 |
| (self − other_llm): Opus − gpt | −2.72 [−5.08, −0.36] | .024 |

On the DV scale: (other_llm − human) is +0.567 for Opus and −0.050 for gpt-5.4,
an interaction of +0.617. **The framing effect differs by model.**

## 9. Robustness

- **Round-by-round:** Opus other_llm climbs 0.15 → 0.70 across R8–14 (gradual
  tipping); gpt-5.4 other_llm *declines* 0.35 → 0.20 (failed transitions
  reverting to J). Collapsing to a single proportion hides this.
- **Player-DV distribution:** strongly bimodal everywhere, as expected when
  games tip as a unit.
- **Game-level leave-one-out:** no cell depends on one game. Opus other_llm
  LOO range [0.534, 0.630]; gpt-5.4 human [0.272, 0.365].
- **Including historical n=6:** Opus other_llm .567 → .515 pooled; gpt-5.4
  .279 → .264. Direction of every conclusion unchanged.
- **M vs N:** switching is overwhelmingly to **M**. Opus other_llm is the only
  cell with meaningful N (202 M / 36 N); gpt-5.4 is 138/115/420 M with 2 N.
- **Outside {J, M, N}:** none, in any cell. The 0/1 coding drops no data.

## 10. Figure

`figure_claimB.pdf` / `.png` — human → other LLM → self on the x-axis, both
models, player-level points shown faintly, game-clustered 95% CIs.

## 11. Interpretation

**Do Opus 4.8 and gpt-5.4 behaviourally distinguish self copies, other LLMs,
and humans in this coordination task?**

**Opus 4.8: yes — all three.** All three planned contrasts survive Holm
correction, and the cell means are monotonically graded: human 0.000 <
other_llm 0.567 < self 0.995. This is outcome **A**.

**gpt-5.4: no — only self vs non-self.** Self differs sharply from both other
framings, but `other_llm` and `human` are not distinguishable (Holm p=.75), and
equivalence is not established either. This is outcome **B**.

**The two models differ (outcome D).** The framing × model interaction is
reliable on every contrast, so the three-way distinction must **not** be stated
as a general property of "LLMs". It is a property of Opus 4.8 in this task.

Scope constraints observed: actual partners are held fixed, so this licenses a
causal claim only about the effect of **partner framing text on behaviour** —
not about self-knowledge or introspection, not about whether either model's
implicit expectations of humans or other LLMs are accurate, and not about
partner-specific coordination ability.
