# Study 1 — Human Prediction Data (Qualtrics)

First human prediction study for the Name Game. Participants act as **observers**
predicting how players behave in later rounds given round-8 evidence of switching
from convention **J** to the higher-value **M/N** options.

## Files
- `Name_game_predictions_p1_2026-07-02.csv` — Qualtrics raw export.
  - Survey: `Name_game_predictions`
  - Exported: **July 2, 2026, 09:52**
  - **180 responses**, 46 columns.
  - Qualtrics format: rows 1–3 are headers (column name, question text, importId
    JSON); response data begins at row 4.
- `Name_game_free_responses_p1_readable.csv` — cleaned long-format extract of the
  open-text answers (columns: `ResponseId, condition, round9_pick, reasoning,
  other_comments`); **165 rows**, one per response.

## Provenance / notes
- Raw export downloaded as `Name_game_predictions_July 2, 2026_09.52.csv`.
- An earlier cumulative export existed (`June 23, 2026_04.47.csv`, 123 responses,
  identical columns). Qualtrics exports are cumulative, so the July 2 file is the
  authoritative superset; the June 23 file was not retained.
- Free-response file downloaded as `Name_game_free_responses_readable.csv` (July 2).

## Relationship to Study 2
Study 2 (`Name_game_predictions_p2`, exported July 7 2026, 145 responses) is a
separate rerun survey and lives in `../study2_predictions/`.
