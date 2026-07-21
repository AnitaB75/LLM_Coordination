# Stimulus sets

Every stimulus set below is an **option set for a coordination game**: N model
instances each pick one option, and all score only if they match. The research
question these probe was *whether self-play can coordinate on one of several
equal-value options a priori* — i.e. whether the models manufacture a shared
focal point from the options' structure (alphabetical order, position,
prototypicality, salience). See the top-level `README.md` for the pilots that
use each set, and `figures/hackability_*.png` for the quantified results.

Visual pairs are rendered in **two position-counterbalanced versions** — `*_AB`
(option A left, B right) and `*_BA` (reversed) — so a probe can randomize
left/right per sample and separate *shape*-preference from *position*-preference.

| Folder | Contents | # options | Used by | Hackable? (a-priori self-play coordination) |
|---|---|---|---|---|
| `visual/`   | `proto` = circle vs square; `mirror` = an irregular blob vs its mirror image; `rot` = the blob vs its 90° rotation. Neutral gray fill, equal size. | 2 | `make_shape_stimuli.py`, `visual_probe.py`, `visual_probe_posrand.py` | **circle/square 100%** (prototypicality); **rotation ~80%** (orientation); **mirror blobs weak & models disagree** — the cleanest "matched" pair |
| `visual2/`  | `inkblot` = two internally-symmetric Rorschach blobs (no handedness cue); `noise` = two independent random grayscale textures. | 2 | `visual2_posrand_probe.py` | **inkblots split the models** — GPT hacks via a global-symmetry cue (100%), Claude finds no shape cue and falls to position; **noise weak & inconsistent** |
| `mountains/`| `nd1`–`nd3`, each a pair of **nondescript** procedural peaks (matched height/ruggedness, different profiles). | 2 | `mountain_probe.py`, `mountain_posrand_probe.py` | **~90–95%** — both models converge on the "most distinctive" peak (twin-summit / clean central peak) and **agree** |
| `landscapes/`| Five distinct scenes: `mountains`, `hills`, `desert`, `coast`, `forest` (+ `_composite`). | 5 | `landscape_probe.py` | **100%** — both models pick the snow **mountain** every time (salience); the most hackable set |
| `cars/`     | Six generic same-color car silhouettes: `sedan`, `coupe`, `hatch`, `suv`, `van`, `pickup` (+ `_composite`). No brand names → no fame cue. | 6 | `car_probe.py` | **~65%** — both converge on the **van** (largest / most distinctive silhouette) |
| `diagrams/` | The roundabout paradigm, as figures (not coordination stimuli): `roundabout.png`, `six_arrows.png` (payoff layout), `seat_views.png` (per-seat rotation+mirror), `feedback_views.png` (per-seat feedback). | — | design docs for `Name_Game_roundabout_selfplay.py` | n/a — illustrates the **un-hackable** design |

## The through-line

Every *labelled/feature-rich* set has a "most distinctive" member a shared rule
can single out, so self-play coordinates a priori — even on random noise
(alphabetized) and same-color car silhouettes (largest). The only sets that
resist are the genuinely **matched** ones (mirror blobs, random noise), and even
they fall back to *position*. This is why the roundabout paradigm
(`diagrams/`, and `Name_Game_roundabout_selfplay.py`) does not rely on the
stimulus lacking cues: it gives each of the 6 seats a **private, fixed rotation +
mirror** of the same junction, so any focal rule ("pick NE", "pick the top exit")
resolves to a *different* real exit for each seat. Coordination on the two equal
higher-value exits then becomes impossible a priori — the group can only converge
through feedback.
