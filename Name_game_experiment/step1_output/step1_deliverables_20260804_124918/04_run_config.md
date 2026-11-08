# Step 1 — run configuration and console report

Run ID `20260804_124918` · collected 2026-08-04 · 600 API calls in 405 s (8 workers).

## Models and settings

Settings were held fixed across all 600 calls. The raw CSV carries the serialized
settings on every row; both models show exactly **one distinct settings string**
across the run, so there is no per-call drift to check for.

| | GPT | Claude |
|---|---|---|
| `model_id` | `gpt-5.4` | `claude-opus-4-8` |
| endpoint | Responses API | Messages API |
| reasoning / thinking | `reasoning={"effort": "low", "summary": "auto"}` | `thinking={"type": "adaptive", "display": "summarized"}` |
| effort | low | `output_config={"effort": "low"}` |
| max tokens | not set | `max_tokens=2000` |
| temperature / top_p / top_k | not set — rejected alongside reasoning on this model | not set |
| seed | **not available** — the Responses API exposes no seed parameter | **not available** — the Messages API exposes no seed parameter |

Verbatim from the data:

```
gpt    : {"max_output_tokens": null, "model": "gpt-5.4", "reasoning": {"effort": "low", "summary": "auto"}, "seed": null, "temperature": null}
claude : {"max_tokens": 2000, "model": "claude-opus-4-8", "output_config": {"effort": "low"}, "seed": null, "temperature": null, "thinking": {"display": "summarized", "type": "adaptive"}}
```

The `seed` column in the raw CSV is therefore blank on every row. The only seeded
component is the counterbalance allocation RNG (`ASSIGN_SEED = 20260804`, in
`step1_form_choice.py`), which is deterministic and reproducible.

## Prompt

`prompt_sha8 = 3003ae1b` — stamped on all 600 rows; exactly one distinct value in
the dataset. Sent as a **single user text block alongside the image**; no system
prompt. Full text:

> You are playing a one-shot coordination game with ONE other player - another instance of you (the same AI model). You will each independently pick one of the two shapes shown; you want to pick the SAME shape as the other player. No communication. IMPORTANT: the left/right position of the two shapes is randomized independently for each of you, so position CANNOT be used to coordinate - you must coordinate on a shape by its FORM. Give one short sentence of reasoning, then end with exactly 'CHOICE: LEFT' or 'CHOICE: RIGHT' (the position in YOUR image of the shape you chose).

This is byte-identical to the arm-1 prompt in the project's earlier
`confidence_calibration_probe.py`. **No labels (A/B or otherwise) are shown to the
model.** The A/B identity coding exists only in the runner and the analysis.

## Design

- **Stimuli:** the existing side-by-side composite images, unmodified — one
  1040×520 PNG per variant. Pairs: `proto` (circle / square), `inkblot` (ib-A /
  ib-B), `rot` (blob / blob-rot90), `noise` (noise-A / noise-B), `mirror` (blob /
  blob-mirror).
- **Counterbalance:** variant `AB` or `BA`, **exactly 30 / 30 in every cell**
  (verified in the data). `AB` puts identity 1 on the left, `BA` puts identity 2
  there. Position and identity are independent by construction.
- **Response:** one sentence of reasoning, then `CHOICE: LEFT` or `CHOICE: RIGHT`.
  Recoded to identity offline via `side + variant`.
- **Trials:** each call is independent — a fresh request with no conversation
  history.

## Valid trials per cell

**60 / 60 in all ten cells**, no exceptions:

| | proto | inkblot | rot | noise | mirror |
|---|---|---|---|---|---|
| gpt | 60 | 60 | 60 | 60 | 60 |
| claude | 60 | 60 | 60 | 60 | 60 |

## Failures and deviations

- **Unparseable responses: 0.** Every one of the 600 calls returned a parseable
  `CHOICE: LEFT` / `CHOICE: RIGHT`. The retry path never fired, so
  `n_api_attempts == n_valid == 60` in every cell and no row has
  `superseded_by_retry = True`.
- **API errors: 0.**
- **Deviation from the original 6-pair battery:** `nd2` was dropped by decision
  before launch, leaving `proto` as the single focal ceiling control. The other
  five pairs are unchanged.
- **No data were excluded post hoc.** All 600 rows have
  `included_in_analysis = True`.

## Pooling

**None.** This is a standalone run. It was not pooled with:

- the project's earlier arm-1 data from `confidence_calibration_probe.py` (the
  prompt is byte-identical and pooling would be defensible, but those CSVs are
  not present in the current working tree — see the caveat below);
- the abandoned A/B-label run from earlier the same day
  (`step1_calls_20260804_122624.csv`, `prompt_sha8 = b0dbf7da`), which used a
  different prompt and a bare-letter response format and is retained only as a
  record of that design failing.

Every statistic in `02_` and `03_` is computed from `01_raw_calls.csv` alone.

## Caveats to carry forward

1. **`mirror` is a side-default, not a weak form preference.** Both models mostly
   answered LEFT regardless of which shape was there (GPT P(LEFT) = 0.767, Claude
   0.850). Because the counterbalance is exactly balanced, this pushes the
   identity match *toward chance* rather than inflating it — the match rates are
   honest, conservative estimates. What misleads is the **modal share**: GPT's
   "70% blob-mirror" is driven almost entirely by the BA cell, where blob-mirror
   happens to sit on the left (29/30 there vs 13/30 when it sits on the right).
2. **Claude's `inkblot` modal identity is a 30/30 tie** (`modal_tied = True`); the
   reported modal option is an alphabetical tiebreak and carries no meaning.
   Claude answered `RIGHT` 60/60 on this pair while giving a *form* justification
   each time, describing whichever shape occupied the right slot. The choice
   tracks position; the stated reason is fitted to it.
3. **The bootstrap CI on the between-model difference** (`03_`) resamples each
   model independently and differences by draw index. The two models were run as
   separate API calls, so there is no trial-level pairing to preserve; the shared
   index simply combines two independent replicates.
