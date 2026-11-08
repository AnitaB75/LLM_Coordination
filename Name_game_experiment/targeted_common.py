"""
Shared registry and metrics for the targeted screen (Step 1c) and its confirmation.

Two jobs:

REGISTRY  resolve a stimulus_id to its image files and identity coding, across the
          three places this project keeps stimuli:
            stimuli/targeted/   the 27 new pairs (manifest.csv)
            stimuli/screen/     the 30 Step 1b pairs (manifest.csv)
            stimuli/visual{,2}/ the 5 Step 1 originals (hardcoded in step1_form_choice.SETS)
          The confirmation battery mixes all three, so it needs one lookup.

METRICS   every per-cell statistic, defined once and imported by both analysis
          stages, so screen and confirmation numbers are computed by the same code.

A note on the position diagnostic, because it differs subtly from Step 1b.

Step 1b computed the conditional-position rates keyed to the LAYOUT LABEL:
    p_ab = P(choose modal | variant == AB)
That is only "P(choose modal | modal shown LEFT)" when the modal identity happens
to be identity_1. When the modal identity is identity_2, AB puts it on the RIGHT
and the two conditionals are swapped. `form_consistency` is unaffected, since it
takes the absolute difference and is symmetric under the swap - so the numbers
here remain directly comparable to Step 1b. But the individual conditionals were
mislabelled half the time, and the spec asks for them explicitly, so here they are
keyed to where the modal identity ACTUALLY appeared.
"""
import os, csv, math, random, re
from collections import Counter

POS_WORDS = re.compile(r"\b(left|right|leftmost|rightmost|first|second)\b", re.I)

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stimuli")

BOOT_B = 5000
BOOT_SEED = 20260807

# Step 1 originals: id -> (subdir, identity_1, identity_2). Mirrors
# step1_form_choice.SETS; kept here so this module has no import side effects.
LEGACY = {
    "proto":   ("visual",  "circle",  "square"),
    "inkblot": ("visual2", "ib-A",    "ib-B"),
    "rot":     ("visual",  "blob",    "blob-rot90"),
    "noise":   ("visual2", "noise-A", "noise-B"),
    "mirror":  ("visual",  "blob",    "blob-mirror"),
}


def _read_manifest(path, id_col, fam_col, i1, i2, manip_col=None):
    if not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        out[r[id_col]] = dict(
            stimulus_id=r[id_col], family=r[fam_col],
            identity_1=r[i1], identity_2=r[i2],
            manipulation=r.get(manip_col, "") if manip_col else
                         (r.get("cue_1", "") + ("/" + r["cue_2"] if r.get("cue_2") else "")))
    return out


def registry():
    """stimulus_id -> dict(stimulus_id, family, identity_1, identity_2, manipulation, source, dir)."""
    reg = {}

    tg = _read_manifest(os.path.join(STIM, "targeted", "manifest.csv"),
                        "stimulus_id", "family", "identity_1", "identity_2", "manipulation")
    for k, v in tg.items():
        reg[k] = {**v, "source": "targeted", "dir": os.path.join(STIM, "targeted")}

    # Expansion battery (2026-08-14 generation-ladder study). Additive and
    # backward-compatible: _read_manifest returns {} when the directory is absent,
    # and setdefault below never overwrites an existing id.
    ex = _read_manifest(os.path.join(STIM, "expansion", "manifest.csv"),
                        "stimulus_id", "family", "identity_1", "identity_2",
                        "manipulation")
    for k, v in ex.items():
        reg.setdefault(k, {**v, "source": "expansion",
                            "dir": os.path.join(STIM, "expansion")})

    sc = _read_manifest(os.path.join(STIM, "screen", "manifest.csv"),
                        "pair", "family", "identity_1", "identity_2")
    for k, v in sc.items():
        reg.setdefault(k, {**v, "source": "step1b_screen",
                            "dir": os.path.join(STIM, "screen")})

    for k, (sub, i1, i2) in LEGACY.items():
        reg.setdefault(k, dict(stimulus_id=k, family="step1_original", manipulation="",
                               identity_1=i1, identity_2=i2, source="step1",
                               dir=os.path.join(STIM, sub)))
    return reg


def paths(entry):
    return (os.path.join(entry["dir"], f"{entry['stimulus_id']}_AB.png"),
            os.path.join(entry["dir"], f"{entry['stimulus_id']}_BA.png"))


def slot_identities(entry, variant):
    """(identity shown LEFT, identity shown RIGHT). AB puts identity_1 on the left."""
    i1, i2 = entry["identity_1"], entry["identity_2"]
    return (i1, i2) if variant == "AB" else (i2, i1)


# ----------------------------------------------------------------- metrics ----
def unbiased_match(counts):
    """Unbiased dyadic self-match: sum_k n_k(n_k-1) / [N(N-1)].

    The probability that two independent draws from this cell agree, with the
    same-draw pairs removed - so it is not inflated by finite n the way the
    plug-in estimator is.
    """
    n = sum(counts.values())
    if n < 2:
        return float("nan")
    return sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))


def plugin_match(counts):
    """Plug-in match: sum_k p_k^2. Biased upward by ~1/n relative to unbiased."""
    n = sum(counts.values())
    if not n:
        return float("nan")
    return sum((v / n) ** 2 for v in counts.values())


def entropy_bits(counts):
    n = sum(counts.values())
    if not n:
        return float("nan")
    return -sum((v / n) * math.log2(v / n) for v in counts.values() if v)


def boot_draws(labels, seed, b=BOOT_B):
    """Bootstrap resamples of the unbiased match probability. Fixed seed."""
    rng = random.Random(seed)
    n = len(labels)
    return [unbiased_match(Counter(rng.choices(labels, k=n))) for _ in range(b)]


def pct_ci(draws):
    s = sorted(d for d in draws if d == d)          # drop NaN
    if not s:
        return float("nan"), float("nan")
    return s[int(0.025 * len(s))], s[int(0.975 * len(s))]


def cell_metrics(rows, entry):
    """All per-(model x stimulus) statistics from that cell's valid trials.

    `rows` are the deduplicated VALID trials; each needs choice_identity,
    response_side and variant.
    """
    n = len(rows)
    ident = [r["choice_identity"] for r in rows]
    ic = Counter(ident)
    # deterministic tie-break so a 50/50 split does not flip between runs
    modal = max(ic, key=lambda k: (ic[k], k)) if ic else ""

    # where did the modal identity actually appear on each trial?
    on_left = [r for r in rows if slot_identities(entry, r["variant"])[0] == modal]
    on_right = [r for r in rows if slot_identities(entry, r["variant"])[1] == modal]
    p_ml = (sum(r["choice_identity"] == modal for r in on_left) / len(on_left)
            if on_left else float("nan"))
    p_mr = (sum(r["choice_identity"] == modal for r in on_right) / len(on_right)
            if on_right else float("nan"))
    form_consistency = 1 - abs(p_ml - p_mr) if on_left and on_right else float("nan")

    sides = Counter(r["response_side"] for r in rows)
    # Lexical check on the rationale, reported for review and never used to
    # auto-reject. It is the interesting half of the position-fallback story: a
    # cell can sit at P(LEFT)=1.0 while almost never SAYING anything positional,
    # because the model supplies a form-based rationale after the fact - and the
    # rationale's stated rule then flips to match whichever form landed on the
    # preferred side.
    pos_lex = (sum(bool(POS_WORDS.search(r["response_raw"].split("CHOICE:")[0]))
                   for r in rows) / n) if n else float("nan")
    return dict(
        n_valid=n,
        modal_identity=modal,
        modal_share=round(ic[modal] / n, 4) if n else float("nan"),
        match_unbiased=round(unbiased_match(ic), 4),
        match_plugin=round(plugin_match(ic), 4),
        entropy_bits=round(entropy_bits(ic), 4),
        p_left=round(sides.get("LEFT", 0) / n, 4) if n else float("nan"),
        n_modal_shown_left=len(on_left), n_modal_shown_right=len(on_right),
        p_modal_given_modal_left=round(p_ml, 4),
        p_modal_given_modal_right=round(p_mr, 4),
        form_consistency=round(form_consistency, 4),
        rationale_mentions_position=round(pos_lex, 4),
        n_AB=sum(r["variant"] == "AB" for r in rows),
        n_BA=sum(r["variant"] == "BA" for r in rows),
    )


def load_valid(path):
    """Deduplicate a calls CSV to one valid trial per run_id, plus per-cell error counts.

    The runner writes the unparseable first attempt AND its retry, both tagged with
    the same run_id. Keep the valid one. A run_id whose retry also failed
    contributes no trial and is counted as a loss, never silently ignored.
    """
    rows = list(csv.DictReader(open(path)))
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)

    valid, losses = [], []
    parse_fail = Counter()
    api_err = Counter()
    for run_id, rs in by_run.items():
        key = (rs[0]["model"], rs[0]["stimulus_pair"])
        for r in rs:
            if r["superseded_by_retry"] == "True":
                parse_fail[key] += 1
            if r["response_raw"].startswith("__ERR__"):
                api_err[key] += 1
        ok = [r for r in rs if r["valid_response"] == "True"]
        if ok:
            valid.append(ok[0])
        else:
            losses.append(run_id)
            parse_fail[key] += 0
    return valid, dict(parse_fail=parse_fail, api_err=api_err, losses=losses,
                       n_rows=len(rows), n_runs=len(by_run))
