# Study 2 — Human Prediction Data (Qualtrics)

Second human prediction study for the Name Game. Participants act as **observers**
predicting how players behave in later rounds given round-8 evidence of switching
from convention **J** to the higher-value **M/N** options.

## Files
- `Name_game_predictions_p2_2026-07-07.csv` — Qualtrics raw export.
  - Survey: `Name_game_predictions_p2`
  - Exported: **July 7, 2026, 07:59**
  - **145 responses**, 47 columns.
  - Qualtrics format: rows 1–3 are headers (column name, question text, importId
    JSON); response data begins at row 4.

## Provenance / notes
- Downloaded from Qualtrics as `Name_game_predictions_p2_July 7, 2026_07.59.csv`.
- Three same-day exports existed in Downloads (07.45 → 66 resp, 07.53 → 114 resp,
  07.59 → 145 resp). Qualtrics exports are **cumulative**, so the 07:59 file is the
  authoritative superset; the earlier two were not retained.
- Some response rows contain embedded newlines in free-text answers, so the raw
  line count (148) exceeds the response count (145).

## Relationship to Study 1
Study 1 (the first prediction survey, `Name_game_predictions`, exported July 2 2026)
is a separate survey. This `study2_predictions/` folder holds the p2 rerun only.
