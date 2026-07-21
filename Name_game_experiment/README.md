# Name Game — LLM-agent pilots

LLM-agent experiments for the **Name Game** coordination paradigm — the
computational companion to the human *"Mutual Observability Determines When
Precedent is Moralized in Multi-Agent Coordination Games"* study
(`../qualtrics_human_study/`, and the manuscript). The paradigm follows Centola &
Baronchelli (2015).

## The game

Six model instances play 14 rounds. Each round every seat picks one option
simultaneously; **all six score only if they all match** (no communication).

- **Rounds 1–7:** only **J** scores (10 pts) → the group converges on J, an
  established *convention*.
- **Rounds 8–14:** J still = 10, but **M = 15 and N = 15** appear. M and N are
  *equal* — coordinating on the higher payoff requires the group to agree on
  *which* one.

Core question: do the agents hold the convention (J) or coordinate a switch to a
higher-paying option — and *how*?

**Models:** `gpt-5.4` (OpenAI, Responses API) and `claude-opus-4-8` (Anthropic,
Messages API). Every seat is the same model on the same prompt — the "human"
framings are a **belief manipulation**, not real humans.

## Reproduction

Requires a `.env` in the **repo root** with `OPENAI_API_KEY` and
`ANTHROPIC_API_KEY` (git-ignored; never committed). Runners write their CSV to
the working directory — move outputs into `results/` for the analysis scripts.

```
python3 Name_Game_both_3cond.py claude    # or gpt
python3 make_shape_stimuli.py             # (re)build visual stimuli
python3 visual_probe_posrand.py           # a visual probe
```

## Folder map

| Path | Contents |
|---|---|
| `stimuli/`  | All coordination stimulus sets + the roundabout diagrams — see `stimuli/README.md` |
| `results/`  | Pilot output CSVs (per round × seat: choice, raw response, **reasoning trace**, points) |
| `figures/`  | Analysis figures (design, results, hackability) |
| `*.py`      | Runners, stimulus generators, and analysis (indexed below) |

## Pilots (this round)

| Pilot | Runner | Data (`results/`) | Tests | Key finding |
|---|---|---|---|---|
| **Belief × 3 conditions** | `Name_Game_both_3cond.py` | `name_game_3cond_{claude,gpt}_*` | self / mixed(self+human) / human, n=10 | Abandoning J is a **self-recognition** effect: self → M 10/10 both models; human → J. Claude monotonic (self 10 → mixed 3 → human 0); GPT self+mixed high (10, 9) then drops |
| **Generic-AI mix** | `Name_Game_both_mixedAI.py` | `name_game_mixedAI_{claude,gpt}_*` | "a mix of AI models + humans" | **mixed_ai ≈ human** for both models — generic-AI partners are *not* treated like copies-of-self |
| **Hack: random strings** | `Name_Game_strings_selfplay.py` | `name_game_strings_selfplay_{claude,gpt}_*` | replace M/N with random 3-letter strings, self-play | 5/5 hacked — models **alphabetize noise** (Claude) / take first-listed (GPT) |
| **Hack: car makes** | `Name_Game_carmakes_selfplay.py` | `name_game_carmakes_selfplay_{claude,gpt}_*` | replace M/N with 6 car-make names | 5/5 hacked — alphabetical / prominence |
| **Visual (baseline)** | `visual_probe.py` | `visual_probe_*` | circle/square & blobs, position available | coordinate on **position** (leftmost), even for circle/square |
| **Visual (position removed)** | `visual_probe_posrand.py` | `visual_probe_posrand_*` | told position is randomized | circle→circle 12/12 (prototypicality); mirror blobs weak & models disagree |
| **Inkblots + noise** | `visual2_posrand_probe.py` | `visual2_posrand_*` | internally-symmetric blobs; random textures | inkblots **split the models** (GPT hacks via global symmetry, Claude can't); noise weak |
| **Nondescript mountains** | `mountain_probe.py`, `mountain_posrand_probe.py` | `mountain_*` | matched procedural peaks | ~90% — both pick the "most distinctive peak" and agree |
| **Five landscapes** | `landscape_probe.py` | `landscape_probe_*` | 5 distinct scenes, pick one | 100% **mountains** (salience) — most hackable |
| **Generic cars** | `car_probe.py` | `car_probe_*` | 6 unnamed car silhouettes | ~65% **van** (largest) — names stripped, shape cue remains |
| **Roundabout (solution)** | `Name_Game_roundabout_selfplay.py` | `roundabout_selfplay_{claude,gpt}_*` | private per-seat rotation+mirror; convention=10, two 15-exits | **0/10 leaks** — un-hackable by construction; group holds the convention (can't a-priori coordinate the two symmetric 15-exits) |
| **Roundabout (devalued)** | `Name_Game_roundabout_devalued.py` | `roundabout_devalued_{claude,gpt}_*` | convention dropped to 3 in rounds 8–14 | still **0/10 leaks**; group holds the 3-pt convention even at a 5× incentive to switch — self-play does **not** initiate |

Analysis / figures: `analyze_roundabout.py`, `make_pilot_figures.py`
(→ `figures/fig1–3`), `plot_trial_choices.py`, `build_summary_pdf.py`. The
hackability rankings (`figures/hackability_*.png`, `hackable_stimuli_gallery.png`)
were built from the `results/` CSVs.

## Headline findings

1. **Convention-breaking is a self-recognition effect.** Both models abandon the
   J convention for a higher payoff *only* when they believe partners are copies
   of themselves. Generic-AI or human partners → hold J.
2. **Self-play "hacks" almost any option set.** Whenever the options carry a
   shared salient feature (alphabetical order, position, prototypicality,
   salience), six identical instances coordinate a priori — even on random noise.
   Only genuinely *matched* option sets resist (and then fall back to position).
3. **The roundabout design is un-hackable by construction.** Private per-seat
   rotation+mirror makes every focal rule point to a *different* real exit, so the
   two equal higher-value exits cannot be coordinated a priori. Self-play then
   holds the convention and — unlike humans in the mutual-observability study —
   **does not initiate** the costly first move that would let the group tip to the
   higher payoff, even at a 5× incentive.

## Legacy / precursor scripts

Earlier single-provider or single-condition runners, superseded by the `both_*`
versions above but kept for provenance:
`Name_Game_LLM_feedback_MN.py`, `…_MN_HUMAN_ONLY.py`, `…_MIXED.py`,
`Name_Game_Claude_3cond.py`, `Name_Game_bot_follow_human.py`,
`Name_Game_reasoning_openAI_participants_implicit.py`,
`Name_Game_two_letters_no_feedback_openAI_participants.py`.
