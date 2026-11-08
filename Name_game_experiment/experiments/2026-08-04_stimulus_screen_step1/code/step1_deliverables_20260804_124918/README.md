# Step 1 deliverables — run `20260804_124918`

Choice distribution and within-model dyadic coordination rate for GPT-5.4 and
Claude Opus 4.8 across five binary stimulus pairs. One-shot, no communication, no
labels shown to the model.

Everything in `02_`–`05_` is derived from `01_raw_calls.csv`. Nothing else is
needed to reproduce it.

## Files

| File | Contents |
|---|---|
| `01_raw_calls.csv` | 600 rows, one per API attempt. Includes the full visible response, the reasoning trace, the raw LEFT/RIGHT answer, the recoded identity, and the serialized API settings. |
| `02_summary_model_x_stimulus.csv` | 10 rows (2 models × 5 stimuli). Identity counts and proportions, modal identity and share, unbiased dyadic match with bootstrap CI, plug-in match, entropy and normalized entropy, P(LEFT) and the position-coded match, variant balance, invalid/retry/error counts. |
| `03_comparison_gpt_vs_claude.csv` | 5 rows. Both models side by side plus the GPT−Claude difference with a bootstrap 95% CI and a flag for whether that CI excludes zero. |
| `04_run_config.md` | Model IDs, exact settings, full prompt and hash, per-cell n, failures, deviations, pooling status, caveats. |
| `05_match_plot.png` / `.pdf` | Visualization only. |

## Key columns in `01_raw_calls.csv`

- `run_id` — one planned trial, e.g. `claude__inkblot__017`. Unique per model × stimulus × replicate.
- `attempt_number` — 1, or 2 where a first attempt was unparseable. **This run: all 1.**
- `variant` — `AB` or `BA`. Determines the left/right layout.
- `left_identity` / `right_identity` — which underlying stimulus occupied each side *on this trial*.
- `response_side` — the model's raw answer, `LEFT` or `RIGHT`.
- `choice_identity` — `response_side` recoded through `variant`. **This is the analysis variable.**
- `included_in_analysis` — `True` for the row used downstream (the last valid attempt per `run_id`). **This run: all 600 True.**
- `reasoning` — the model's reasoning summary (GPT) or thinking summary (Claude). `response_raw` holds the visible one-sentence justification plus the CHOICE line.

## Verifying the recode

The whole analysis rests on one mapping, which you can check directly:

```
AB  ->  LEFT = left_identity  (identity 1),  RIGHT = right_identity (identity 2)
BA  ->  LEFT = left_identity  (identity 2),  RIGHT = right_identity (identity 1)
```

Equivalently, `choice_identity` should equal `left_identity` whenever
`response_side == "LEFT"` and `right_identity` whenever it is `RIGHT`, for every
row. One line:

```python
import csv
rows = list(csv.DictReader(open("01_raw_calls.csv")))
assert all(
    r["choice_identity"] == (r["left_identity"] if r["response_side"] == "LEFT" else r["right_identity"])
    for r in rows if r["included_in_analysis"] == "True"
)
```

## Reproducing the statistics

Per model × stimulus cell, over the `included_in_analysis` rows, with counts
`n_k` over `choice_identity` and `N = 60`:

```
unbiased dyadic match  =  sum_k n_k (n_k - 1) / (N (N - 1))
plug-in match          =  sum_k (n_k / N)^2
Shannon entropy        =  -sum_k (n_k/N) log2(n_k/N)          [bits]
normalized entropy     =  entropy / log2(2)                    [K = 2, so equal]
P(LEFT)                =  count(response_side == "LEFT") / N
```

The unbiased estimator is the probability that two *distinct* trials agree. It is
the right quantity for "would two independent instances pick the same stimulus";
the plug-in version is reported alongside as a check and runs slightly higher (at
N = 60 a perfect 30/30 split gives 0.4915 unbiased vs 0.5000 plug-in).

Bootstrap: 5,000 resamples with replacement of the 60 rows in the cell, percentile
95% CI, seeded per cell as `20260804|<model>|<pair>|id` (and `|pos` for the
position coding). Between-model differences resample each model independently and
difference by draw index.

## Two things to check first

**Counterbalancing.** Every cell should be exactly 30 `AB` / 30 `BA`
(`n_AB_variant`, `n_BA_variant` in `02_`). This matters because it is what makes a
pure position strategy read as chance on the identity coding rather than as a
spurious preference.

**Position bias.** `P(LEFT)` in `02_`. A value near 0.5 with a high identity match
is genuine form-tracking. A value near 0 or 1 means the model answered by side —
in which case the identity match is pinned near chance by construction and the
modal identity is meaningless. Both extremes occur in this dataset; see the
caveats in `04_run_config.md`.

## Headline result, for orientation only

The one between-model difference whose CI excludes zero is **`inkblot`**:
GPT 0.967 [0.903, 1.000] vs Claude 0.492 [0.491, 0.537], difference **+0.475
[+0.384, +0.508]**. `proto` is at ceiling for both, `rot` is mid-range for both
with overlapping CIs, and `noise` and `mirror` are at or near chance for both.

Treat this as a pointer to what to check in the raw data, not as the finding.
