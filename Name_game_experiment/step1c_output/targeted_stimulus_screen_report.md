# Targeted stimulus screen (Step 1c)

_Generated 2026-08-07 from `step1c_output/`._

Goal: extend the coordination battery with stimuli showing clear, **genuine** GPT-vs-Claude focality disparities — cases where both models coordinate on FORM and one is simply more concentrated — as opposed to cases where the weaker model has no form rule and quietly defaults to a side.

## Procedure

One-shot coordination framing imported verbatim from `step1_form_choice.py` (prompt sha8 `3003ae1b`), the same module Step 1 and Step 1b used, so all three datasets are comparable. Model settings unchanged: GPT-5.4 with `reasoning.effort=low`, Claude Opus 4.8 with `max_tokens=2000`, adaptive thinking, `output_config.effort=low`. Neither API exposes a sampling seed; the fixed seeds here govern AB/BA allocation only.

- **Screen** — 27 new pairs × 2 models × n=30, exact 15/15 AB/BA.
- **Confirmation** — selected battery × 2 models × n=60 **fresh** trials, exact 30/30 AB/BA, separate allocation seed.

Screen and confirmation are never pooled for the confirmatory estimates: the screen chose the stimuli, so reusing it would bias them upward (winner's curse). Every confirmation number below is from the fresh sample alone.

### The position-fallback diagnostic

For a model's modal identity *m*, form consistency is

```
form_consistency = 1 − | P(choose m | m shown LEFT) − P(choose m | m shown RIGHT) |
```

A model choosing by form picks *m* at the same rate wherever it sits, scoring ~1. A model choosing by side picks *m* only when *m* lands on its preferred side, driving the two conditionals to 1 and 0 and the score to 0. Under balanced AB/BA a pure side default reads as ≈0.50 on the identity coding — **indistinguishable from indifference on the match rate alone**. That is why a low match rate is never reported here as indifference without this diagnostic agreeing.

### Selection criteria

| criterion | threshold |
|---|---|
| strong model dyadic match | ≥ 0.85 |
| weak model dyadic match | ≤ 0.70 |
| absolute difference | ≥ 0.20 |
| strong model form consistency | ≥ 0.90 |
| **weak model form consistency** | **≥ 0.80** |

Meeting all five ⇒ **PRIMARY**, `clean_form_disparity = TRUE`. Meeting the first four but not the last ⇒ **SECONDARY**, flagged as a difference in *kind* (partial position fallback), never counted as a clean focality difference.

## Screen results — all 27 candidates

`gFC` / `cFC` = form consistency. Sorted by |Δ|.

| stimulus | family | GPT | gFC | Claude | cFC | Δ (GPT−Claude) | 95% CI | status |
|---|---|---|---|---|---|---|---|---|
| `a_chev_ud` | A_semantic | 0.492 | 0.67 | 1.000 | 1.00 | -0.508 | [-0.517, -0.405] | SECONDARY |
| `b_tee_ell` | B_glyph | 1.000 | 1.00 | 0.540 | 0.33 | +0.460 | [+0.287, +0.515] | POSITION_DEFAULT |
| `c_contr_subtle` | C_contrast | 0.933 | 0.93 | 0.483 | 0.00 | +0.451 | [+0.294, +0.517] | POSITION_DEFAULT |
| `d_dens_extreme` | D_density | 0.933 | 0.93 | 0.540 | 0.60 | +0.393 | [+0.184, +0.515] | SECONDARY |
| `d_texture` | D_density | 0.871 | 1.00 | 0.492 | 0.13 | +0.379 | [+0.184, +0.515] | POSITION_DEFAULT |
| `e_convex_mild` | E_closure | 0.871 | 1.00 | 0.492 | 0.27 | +0.379 | [+0.184, +0.515] | POSITION_DEFAULT |
| `d_dens_subtle` | D_density | 0.630 | 0.67 | 1.000 | 1.00 | -0.370 | [-0.497, -0.186] | SECONDARY |
| `b_spiral_cw` | B_glyph | 0.630 | 0.93 | 0.933 | 0.93 | -0.303 | [-0.460, -0.058] | PRIMARY |
| `d_eq_ink` | D_density | 0.630 | 0.67 | 0.933 | 0.93 | -0.303 | [-0.460, -0.058] | SECONDARY |
| `c_weight_blob` | C_contrast | 0.713 | 0.67 | 1.000 | 1.00 | -0.287 | [-0.434, -0.067] | REJECT |
| `d_clustered` | D_density | 0.814 | 0.93 | 1.000 | 1.00 | -0.186 | [-0.370, +0.000] | REJECT |
| `e_convex_blob` | E_closure | 0.871 | 0.87 | 0.713 | 0.93 | +0.159 | [-0.101, +0.370] | REJECT |
| `b_s_z` | B_glyph | 0.595 | 0.73 | 0.519 | 0.27 | +0.076 | [-0.103, +0.274] | REJECT |
| `c_fill_outline` | C_contrast | 0.933 | 0.93 | 1.000 | 1.00 | -0.067 | [-0.186, +0.000] | REJECT |
| `b_hook_mirror` | B_glyph | 0.483 | 0.27 | 0.503 | 0.20 | -0.021 | [-0.145, +0.074] | REJECT |
| `a_arrow_lr` | A_semantic | 1.000 | 1.00 | 1.000 | 1.00 | +0.000 | [+0.000, +0.000] | REJECT |
| `a_arrow_ud` | A_semantic | 1.000 | 1.00 | 1.000 | 1.00 | +0.000 | [+0.000, +0.000] | REJECT |
| `a_arrow_updiag` | A_semantic | 1.000 | 1.00 | 1.000 | 1.00 | +0.000 | [+0.000, +0.000] | REJECT |
| `a_tee_rot90` | A_semantic | 1.000 | 1.00 | 1.000 | 1.00 | +0.000 | [+0.000, +0.000] | REJECT |
| `a_tee_upright` | A_semantic | 1.000 | 1.00 | 1.000 | 1.00 | +0.000 | [+0.000, +0.000] | REJECT |
| `a_tri_ud` | A_semantic | 1.000 | 1.00 | 1.000 | 1.00 | +0.000 | [+0.000, +0.000] | REJECT |
| `b_circ_bar` | B_glyph | 1.000 | 1.00 | 1.000 | 1.00 | +0.000 | [+0.000, +0.000] | REJECT |
| `b_cross_x` | B_glyph | 1.000 | 1.00 | 1.000 | 1.00 | +0.000 | [+0.000, +0.000] | REJECT |
| `c_contr_extreme` | C_contrast | 1.000 | 1.00 | 1.000 | 1.00 | +0.000 | [+0.000, +0.000] | REJECT |
| `c_stroke_wt` | C_contrast | 1.000 | 1.00 | 1.000 | 1.00 | +0.000 | [+0.000, +0.000] | REJECT |
| `e_closure_c` | E_closure | 1.000 | 1.00 | 1.000 | 1.00 | +0.000 | [+0.000, +0.000] | REJECT |
| `e_open_subtle` | E_closure | 1.000 | 1.00 | 1.000 | 1.00 | +0.000 | [+0.000, +0.000] | REJECT |

### Clean disparities (PRIMARY)

- **`b_spiral_cw`** (Claude>GPT) — screen PRIMARY, ranked by tier then |delta| | clean: both models track form (FC 0.93 / 0.93), strong 0.933 vs weak 0.630, delta 0.303

### Position-fallback disparities (SECONDARY)

Real and reportable, but a difference in kind rather than a focality difference — the weaker model is partially choosing a side.

- **`a_chev_ud`** (Claude>GPT) — weak-model form consistency 0.67; screen SECONDARY, ranked by tier then |delta| | weak model form consistency 0.67 < 0.8 (P(LEFT)=0.33): difference in KIND - weak model PARTIALLY falls back on position, not a clean focality difference
- **`d_dens_extreme`** (GPT>Claude) — weak-model form consistency 0.60; screen SECONDARY, ranked by tier then |delta| | weak model form consistency 0.60 < 0.8 (P(LEFT)=0.30): difference in KIND - weak model PARTIALLY falls back on position, not a clean focality difference
- **`d_dens_subtle`** (Claude>GPT) — weak-model form consistency 0.67; screen SECONDARY, ranked by tier then |delta| | weak model form consistency 0.67 < 0.8 (P(LEFT)=0.33): difference in KIND - weak model PARTIALLY falls back on position, not a clean focality difference
- **`d_eq_ink`** (Claude>GPT) — weak-model form consistency 0.67; screen SECONDARY, ranked by tier then |delta| | weak model form consistency 0.67 < 0.8 (P(LEFT)=0.33): difference in KIND - weak model PARTIALLY falls back on position, not a clean focality difference

### Which families were productive

| family | n | median \|Δ\| | at ceiling (both ≥0.95) | at floor (both ≤0.60) | PRIMARY | SECONDARY |
|---|---|---|---|---|---|---|
| A · semantic / canonical orientation | 7 | 0.000 | 6 | 0 | 0 | 1 |
| B · canonical abstract glyphs | 6 | 0.076 | 2 | 2 | 1 | 0 |
| C · contrast / visual weight | 5 | 0.067 | 2 | 0 | 0 | 0 |
| D · density / numerosity / texture | 5 | 0.370 | 0 | 0 | 0 | 3 |
| E · convexity / closure | 4 | 0.159 | 2 | 0 | 0 | 0 |

## Confirmation — fresh n=60 per model per stimulus

All figures below are from the confirmation sample only.

| stimulus | role | GPT (95% CI) | gFC | Claude (95% CI) | cFC | Δ (95% CI) | replicated |
|---|---|---|---|---|---|---|---|
| `b_tee_ell` | new candidate (GPT>Claude, POSITION_DEFAULT) | 1.000 [1.000, 1.000] | 1.00 | 0.528 [0.492, 0.619] | 0.27 | +0.472 [+0.381, +0.508] | **yes** (+0.460 → +0.472) |
| `a_chev_ud` | new candidate (Claude>GPT, SECONDARY) | 0.548 [0.497, 0.655] | 0.67 | 1.000 [1.000, 1.000] | 1.00 | -0.452 [-0.503, -0.345] | **yes** (-0.508 → -0.452) |
| `c_contr_subtle` | new candidate (GPT>Claude, POSITION_DEFAULT) | 0.903 [0.790, 1.000] | 0.90 | 0.494 [0.491, 0.548] | 0.07 | +0.410 [+0.294, +0.499] | **yes** (+0.451 → +0.410) |
| `d_texture` | new candidate (GPT>Claude, POSITION_DEFAULT) | 0.903 [0.790, 1.000] | 0.97 | 0.528 [0.492, 0.619] | 0.27 | +0.376 [+0.237, +0.475] | **yes** (+0.379 → +0.376) |
| `d_dens_subtle` | new candidate (Claude>GPT, SECONDARY) | 0.602 [0.519, 0.741] | 0.53 | 0.967 [0.903, 1.000] | 0.97 | -0.364 [-0.463, -0.226] | **yes** (-0.370 → -0.364) |
| `s_arrow` | incumbent (Claude>GPT) | 0.602 [0.519, 0.718] | 0.60 | 0.967 [0.903, 1.000] | 0.97 | -0.364 [-0.463, -0.226] | **yes** (-0.497 → -0.364) |
| `d_dens_extreme` | new candidate (GPT>Claude, SECONDARY) | 0.934 [0.845, 1.000] | 0.93 | 0.602 [0.519, 0.718] | 0.80 | +0.332 [+0.178, +0.452] | **yes** (+0.393 → +0.332) |
| `s_convexity` | incumbent (Claude>GPT) | 0.741 [0.619, 0.873] | 0.70 | 1.000 [1.000, 1.000] | 1.00 | -0.259 [-0.381, -0.127] | **yes** (-0.264 → -0.259) |
| `c_open_cplx` | incumbent (Claude>GPT) | 0.537 [0.494, 0.636] | 0.43 | 0.790 [0.675, 0.903] | 0.77 | -0.253 [-0.392, -0.099] | **yes** (-0.276 → -0.253) |
| `s_density` | incumbent (GPT>Claude) | 0.903 [0.790, 1.000] | 0.90 | 0.655 [0.548, 0.790] | 0.97 | +0.249 [+0.080, +0.386] | **yes** (+0.186 → +0.249) |
| `b_spiral_cw` | new candidate (Claude>GPT, PRIMARY) | 0.790 [0.675, 0.903] | 0.90 | 0.967 [0.903, 1.000] | 0.97 | -0.176 [-0.304, -0.031] | direction only (-0.303 → -0.176) |
| `s_contrast` | incumbent (GPT>Claude) | 1.000 [1.000, 1.000] | 1.00 | 0.845 [0.741, 0.967] | 0.97 | +0.155 [+0.033, +0.259] | direction only (+0.370 → +0.155) |
| `noise` | control - floor control | 0.619 [0.528, 0.741] | 0.77 | 0.491 [0.491, 0.537] | 0.13 | +0.127 [+0.025, +0.247] | n/a (control) |
| `mirror` | control - structured ambiguous control | 0.560 [0.501, 0.675] | 0.37 | 0.636 [0.537, 0.765] | 0.73 | -0.076 [-0.217, +0.072] | n/a (control) |
| `rot` | control - intermediate shared coordination | 0.655 [0.548, 0.790] | 0.97 | 0.718 [0.602, 0.845] | 0.80 | -0.063 [-0.231, +0.115] | n/a (control) |
| `d_eq_ink` | new candidate (Claude>GPT, SECONDARY) | 0.718 [0.602, 0.845] | 0.80 | 0.765 [0.636, 0.903] | 0.73 | -0.048 [-0.226, +0.133] | direction only (-0.303 → -0.048) |
| `proto` | control - shared-easy positive control | 1.000 [1.000, 1.000] | 1.00 | 1.000 [1.000, 1.000] | 1.00 | +0.000 [+0.000, +0.000] | n/a (control) |

### Confirmation status under the same five criteria

Applying the criteria to the fresh sample is **reporting**, not selecting — the stimuli were already fixed by the screen. Nothing below was chosen using confirmation data.

| stimulus | screen | confirmation | verdict |
|---|---|---|---|
| `b_tee_ell` | POSITION_DEFAULT | POSITION_DEFAULT | holds |
| `a_chev_ud` | SECONDARY | SECONDARY | holds |
| `c_contr_subtle` | POSITION_DEFAULT | POSITION_DEFAULT | holds |
| `d_texture` | POSITION_DEFAULT | POSITION_DEFAULT | holds |
| `d_dens_subtle` | SECONDARY | SECONDARY | holds |
| `s_arrow` | NOT_SCREENED | SECONDARY | changed tier |
| `d_dens_extreme` | SECONDARY | PRIMARY | **upgraded** |
| `s_convexity` | NOT_SCREENED | REJECT | changed tier |
| `c_open_cplx` | NOT_SCREENED | REJECT | changed tier |
| `s_density` | NOT_SCREENED | PRIMARY | **upgraded** |
| `b_spiral_cw` | PRIMARY | REJECT | **downgraded** |
| `s_contrast` | NOT_SCREENED | REJECT | changed tier |
| `noise` | NOT_SCREENED | REJECT | control |
| `mirror` | NOT_SCREENED | REJECT | control |
| `rot` | NOT_SCREENED | REJECT | control |
| `d_eq_ink` | SECONDARY | REJECT | changed tier |
| `proto` | NOT_SCREENED | REJECT | control |

**Clean disparities surviving confirmation: 2** — `d_dens_extreme`, `s_density`.

### Controls vs their Step 1 baselines

| control | GPT now | GPT Step 1 | Claude now | Claude Step 1 |
|---|---|---|---|---|
| `proto` | 1.000 | 1.000 | 1.000 | 1.000 |
| `rot` | 0.655 | 0.696 | 0.718 | 0.741 |
| `mirror` | 0.560 | 0.573 | 0.636 | 0.497 |
| `noise` | 0.619 | 0.506 | 0.491 | 0.494 |

## Figure

`targeted_stimulus_screen.png` / `.pdf` — (A) GPT vs Claude match for all candidates; (B) disparity effect sizes with bootstrap 95% CIs; (C) the form-consistency diagnostic, separating clean disparities from position fallback; (D) confirmation results for the selected battery.

## Outputs

- `targeted_stimulus_manifest.csv`
- `targeted_screen_calls_*.csv`
- `targeted_screen_summary.csv`
- `targeted_screen_comparison.csv`
- `targeted_screen_selected.csv`
- `targeted_confirmation_calls_*.csv`
- `targeted_confirmation_summary.csv`
- `targeted_confirmation_comparison.csv`
- `targeted_stimulus_screen.png / .pdf`
