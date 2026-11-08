"""
Generate the TARGETED screening battery (Step 1c).

Step 1b screened 30 pairs broadly and was inefficient: 19/30 sat at 1.000/1.000
for both models, and every Claude>GPT dissociation it found (s_arrow, s_convexity,
c_open_cplx) had GPT partially defaulting to a side (form_consistency 0.47-0.60,
P(LEFT) 0.23-0.30) rather than genuinely tracking form. Only the GPT>Claude
dissociations (s_contrast, s_density) were clean on both sides.

So this battery is targeted, not broad. Five families, chosen to hunt for the
thing Step 1b is missing - a CLEAN Claude>GPT disparity, where both models track
form and one is simply more concentrated:

  A  semantic / canonical orientation   the family s_arrow belongs to; if the
                                        s_arrow effect is real rather than a GPT
                                        side-default, siblings should replicate it
  B  canonical abstract glyphs          same question without directional semantics:
                                        does one form read as the "conventional" one
  C  contrast / visual weight           the family s_contrast belongs to (GPT>Claude);
                                        geometry held fixed, only ink varies
  D  density / numerosity / texture     the family s_density belongs to (GPT>Claude)
  E  convexity / closure                the family s_convexity belongs to, at
                                        SUBTLER settings - Step 1b's version may have
                                        been too coarse to separate the models

Step 1b's `compete` (competing-cue) family is deliberately NOT extended: 10 of its
14 members hit 1.000/1.000 on both models. Single-cue manipulations are the
efficient design here.

Geometry is held as close to matched as the manipulation allows. Where a cue
necessarily changes ink (contrast, density) the silhouette is held constant
instead. No size contrasts except where size IS the cue - Step 1b showed size
dominates everything it competes with.

Output format is byte-compatible with the Step 1/1b battery (1040x520 PNG, two
panels, `_AB`/`_BA`), so `targeted_screen.py` runs on it with the imported
Step 1 prompt unchanged.

  stimuli/targeted/<pair>_AB.png , <pair>_BA.png
  stimuli/targeted/manifest.csv           -> targeted_stimulus_manifest.csv
  stimuli/targeted/_contact_<family>.png  eyeball these before spending calls

Run:  python3 make_targeted_stimuli.py
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Wedge, FancyArrow

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stimuli", "targeted")

# identical palette to the Step 1b battery so the two screens are comparable
FILL, EDGE = "#9AA3B2", "#3A4152"
LIGHT, DARK = "#C9CEd6", "#5A6172"
VLIGHT, VDARK = "#E8EAEE", "#2B3040"


# ------------------------------------------------------------------ shapes ----
def poly_xy(n, r=1.0, rot=0.0, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rot
    rad = r * (1 + jitter * (rng.random(n) - 0.5) * 2)
    return np.c_[rad * np.cos(ang), rad * np.sin(ang)]


def star_xy(points, r_out=1.0, r_in=0.45, rot=0.0):
    ang = np.linspace(0, 2 * np.pi, points * 2, endpoint=False) + rot
    rad = np.where(np.arange(points * 2) % 2 == 0, r_out, r_in)
    return np.c_[rad * np.cos(ang), rad * np.sin(ang)]


def blob_xy(seed, n=9, r0=0.62, jit=0.33):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = r0 + jit * rng.random(n)
    return np.c_[r * np.cos(ang), r * np.sin(ang)]


def smooth_closed(xy, k=8):
    """Fourier-smooth a closed polygon into a rounded blob."""
    z = xy[:, 0] + 1j * xy[:, 1]
    f = np.fft.fft(z)
    f[k:-k] = 0
    zs = np.fft.ifft(f)
    t = np.linspace(0, len(zs), 200)
    grid = np.arange(len(zs) + 1)
    dense = (np.interp(t, grid, np.r_[zs, zs[:1]].real)
             + 1j * np.interp(t, grid, np.r_[zs, zs[:1]].imag))
    return np.c_[dense.real, dense.imag]


def rot_xy(xy, deg):
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return xy @ R.T


def bar_xy(w, h):
    """Axis-aligned rectangle as a point array, centred on the origin."""
    return np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])


def thick_path(pts, w):
    """Turn an open polyline into a closed constant-width ribbon."""
    pts = np.asarray(pts, float)
    d = np.gradient(pts, axis=0)
    nrm = np.c_[-d[:, 1], d[:, 0]]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return np.r_[pts + nrm * w / 2, (pts - nrm * w / 2)[::-1]]


def spiral_pts(turns=2.2, r_max=0.95, sign=1, n=260):
    t = np.linspace(0.12, 1, n)
    a = sign * t * turns * 2 * np.pi
    r = t * r_max
    return np.c_[r * np.cos(a), r * np.sin(a)]


def arc_pts(cx, cy, r, a0, a1, n=110):
    a = np.deg2rad(np.linspace(a0, a1, n))
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def s_pts(r=0.44):
    """A proper S: two semicircular arcs meeting at the origin, curving opposite ways.

    The earlier sine-curve version was offset into a ribbon whose radius of
    curvature at the extremes (~0.27) was smaller than half the stroke width, so
    the outline self-intersected and rendered as a folded blob rather than an S.
    """
    top = arc_pts(0, r, r, 30, 270)           # upper right -> over the top -> origin
    bot = arc_pts(0, -r, r, 90, -150)         # origin -> right -> under -> lower left
    return np.r_[top, bot]


# -------------------------------------------------------------------- draw ----
def draw(ax, spec):
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    kind = spec["kind"]
    fc = spec.get("fc", FILL)
    ec = spec.get("ec", EDGE)
    lw = spec.get("lw", 3)
    kw = dict(facecolor=fc, edgecolor=ec, linewidth=lw)

    if kind == "circle":
        ax.add_patch(Circle((0, 0), spec.get("r", 0.82), **kw))
    elif kind == "poly":
        ax.add_patch(Polygon(poly_xy(spec["n"], spec.get("r", 0.9), spec.get("rot", 0.0),
                                     spec.get("jitter", 0.0), spec.get("seed", 0)),
                             closed=True, **kw))
    elif kind == "star":
        ax.add_patch(Polygon(star_xy(spec["points"], spec.get("r_out", 0.95),
                                     spec.get("r_in", 0.45), spec.get("rot", 0.0)),
                             closed=True, **kw))
    elif kind == "smoothblob":
        xy = smooth_closed(blob_xy(spec["seed"]) * spec.get("scale", 1.3))
        ax.add_patch(Polygon(rot_xy(xy, spec.get("deg", 0)), closed=True, **kw))
    elif kind == "arrow":
        a = np.deg2rad(spec.get("deg", 90))
        ax.add_patch(FancyArrow(-0.7 * np.cos(a), -0.7 * np.sin(a),
                                1.4 * np.cos(a), 1.4 * np.sin(a),
                                width=0.28, head_width=0.62, head_length=0.5,
                                length_includes_head=True, **kw))
    elif kind == "chevron":
        # V of constant limb length; deg rotates the whole glyph
        half = np.deg2rad(spec.get("half_angle", 52))
        L = spec.get("L", 0.95)
        pts = np.array([[-L * np.sin(half), L * np.cos(half)], [0, 0],
                        [L * np.sin(half), L * np.cos(half)]])
        pts = rot_xy(pts, spec.get("deg", 0))
        ax.add_patch(Polygon(thick_path(pts, spec.get("w", 0.3)), closed=True, **kw))
    elif kind == "cross":
        # plus sign; deg=45 makes it an X. Identical ink either way.
        arm, w = spec.get("arm", 1.9), spec.get("w", 0.42)
        for xy in (bar_xy(arm, w), bar_xy(w, arm)):
            ax.add_patch(Polygon(rot_xy(xy, spec.get("deg", 0)), closed=True, **kw))
    elif kind == "tee":
        # T / L / rotations built from the SAME two bars, so ink is matched
        arm, w = spec.get("arm", 1.6), spec.get("w", 0.4)
        cap = bar_xy(arm, w) + np.array([0, arm / 2 - w / 2])
        stem = bar_xy(w, arm)
        if spec.get("shape") == "ell":
            cap = bar_xy(arm, w) + np.array([arm / 2 - w / 2, -arm / 2 + w / 2])
        for xy in (cap, stem):
            ax.add_patch(Polygon(rot_xy(xy, spec.get("deg", 0)), closed=True, **kw))
    elif kind == "hook":
        # three-quarter ring with a straight tail; sign mirrors it
        s = spec.get("sign", 1)
        a = np.linspace(np.deg2rad(-40), np.deg2rad(215), 140)
        arc = np.c_[0.62 * np.cos(a), 0.62 * np.sin(a)]
        tail = np.c_[np.full(30, 0.62 * np.cos(a[0])),
                     np.linspace(0.62 * np.sin(a[0]), -0.98, 30)]
        pts = np.r_[tail[::-1], arc]
        pts[:, 0] *= s
        ax.add_patch(Polygon(thick_path(pts, spec.get("w", 0.26)), closed=True, **kw))
    elif kind == "scurve":
        # smooth S vs angular Z, matched extent and stroke width
        if spec.get("angular"):
            v = np.array([[-0.62, 0.88], [0.62, 0.88], [-0.62, -0.88], [0.62, -0.88]])
            pts = np.r_[[v[0] + (v[1] - v[0]) * t for t in np.linspace(0, 1, 60)],
                        [v[1] + (v[2] - v[1]) * t for t in np.linspace(0, 1, 80)],
                        [v[2] + (v[3] - v[2]) * t for t in np.linspace(0, 1, 60)]]
        else:
            pts = s_pts()
        ax.add_patch(Polygon(thick_path(pts, spec.get("w", 0.2)), closed=True, **kw))
    elif kind == "smoothstar":
        # Fourier-smoothed star: concave but ROUNDED, so a convexity contrast
        # against a smooth blob is not also a curvature contrast.
        xy = smooth_closed(star_xy(spec.get("points", 5), spec.get("r_out", 0.98),
                                   spec.get("r_in", 0.76)), k=spec.get("k", 6))
        ax.add_patch(Polygon(xy, closed=True, **kw))
    elif kind == "circbar":
        # identical circular frame, inscribed bar upright vs rotated: ink and
        # extent exactly matched, only the bar's orientation differs
        ax.add_patch(Circle((0, 0), 0.86, facecolor="none", edgecolor=ec, linewidth=4))
        ax.add_patch(Polygon(rot_xy(bar_xy(0.3, 1.44), spec.get("deg", 0)),
                             closed=True, **kw))
    elif kind == "spiral":
        pts = spiral_pts(sign=spec.get("sign", 1))
        ax.add_patch(Polygon(thick_path(pts, spec.get("w", 0.2)), closed=True, **kw))
    elif kind == "ring":
        r, gap = spec.get("r", 0.88), spec.get("gap", 0)
        ax.add_patch(Wedge((0, 0), r, gap, 360 - gap, width=r * spec.get("frac", 0.3), **kw))
    elif kind == "outline":
        ax.add_patch(Circle((0, 0), spec.get("r", 0.82), facecolor="none",
                            edgecolor=spec.get("ec", EDGE), linewidth=spec.get("lw", 4)))
    elif kind == "dots":
        rng = np.random.default_rng(spec.get("seed", 0))
        n = spec["n"]
        if spec.get("grid"):
            side = int(np.ceil(np.sqrt(n)))
            g = np.linspace(-0.8, 0.8, side)
            pts = np.array([(x, y) for y in g for x in g])[:n]
        elif spec.get("cluster"):
            pts = rng.normal(0, spec.get("sd", 0.26), size=(n, 2)).clip(-0.88, 0.88)
        else:
            pts = rng.uniform(-0.85, 0.85, size=(n, 2))
        for x, y in pts:
            ax.add_patch(Circle((x, y), spec.get("r", 0.11), facecolor=fc,
                                edgecolor=EDGE, linewidth=1.4))
    elif kind == "stipple":
        # constant circular silhouette, varying internal grain
        rng = np.random.default_rng(spec.get("seed", 0))
        n, rr = spec["n"], spec.get("r", 0.05)
        pts = []
        while len(pts) < n:
            p = rng.uniform(-0.85, 0.85, size=2)
            if np.hypot(*p) < 0.85 - rr:
                pts.append(p)
        ax.add_patch(Circle((0, 0), 0.87, facecolor="none", edgecolor=EDGE, linewidth=2))
        for x, y in pts:
            ax.add_patch(Circle((x, y), rr, facecolor=EDGE, edgecolor="none"))
    else:
        raise ValueError(kind)


# ------------------------------------------------------------------- pairs ----
# (pair, id1, spec1, id2, spec2, family, manipulation, note)
P = []


def add(name, id1, s1, id2, s2, fam, manip, note):
    P.append((name, id1, s1, id2, s2, fam, manip, note))


# ---- A. semantic / canonical orientation -------------------------------------
# The family s_arrow belongs to. If Claude's 1.000 on s_arrow is a real canonical-
# orientation focal point, these should reproduce it; if it was a one-off, they won't.
add("a_arrow_ud", "up", dict(kind="arrow", deg=90), "down", dict(kind="arrow", deg=270),
    "A_semantic", "arrow up vs down",
    "identical glyph, 180 deg apart; 'up' is the canonical direction")
add("a_arrow_updiag", "up", dict(kind="arrow", deg=90), "diag", dict(kind="arrow", deg=45),
    "A_semantic", "arrow up vs diagonal",
    "smaller angular separation than a_arrow_ud; tests cue strength")
add("a_arrow_lr", "left", dict(kind="arrow", deg=180), "right", dict(kind="arrow", deg=0),
    "A_semantic", "arrow left vs right",
    "horizontal axis has a weaker canonical pole than vertical; near-control for A")
add("a_tri_ud", "up", dict(kind="poly", n=3, r=0.95, rot=np.pi / 2),
    "down", dict(kind="poly", n=3, r=0.95, rot=-np.pi / 2),
    "A_semantic", "triangle apex up vs down",
    "canonical orientation without arrow semantics; area identical")
add("a_chev_ud", "up", dict(kind="chevron", deg=180), "down", dict(kind="chevron", deg=0),
    "A_semantic", "chevron pointing up vs down",
    "directional but not an arrow; same ink, 180 deg apart")
add("a_tee_upright", "upright", dict(kind="tee", deg=0), "inverted", dict(kind="tee", deg=180),
    "A_semantic", "T-glyph upright vs inverted",
    "canonical vs non-canonical orientation of one structural form; ink identical")
add("a_tee_rot90", "upright", dict(kind="tee", deg=0), "sideways", dict(kind="tee", deg=90),
    "A_semantic", "T-glyph upright vs rotated 90",
    "90 deg rather than 180; the rot pair in Step 1 used a blob, this uses a canonical form")

# ---- B. canonical abstract glyphs --------------------------------------------
# Same question as A with the directional semantics stripped out: is one of two
# equally simple invented forms the 'conventional-looking' one?
add("b_cross_x", "plus", dict(kind="cross", deg=0), "ex", dict(kind="cross", deg=45),
    "B_glyph", "plus vs multiplication-sign orientation",
    "one form, two orientations, ink exactly equal; the purest canonicality test")
add("b_tee_ell", "tee", dict(kind="tee", shape="tee"), "ell", dict(kind="tee", shape="ell"),
    "B_glyph", "T-glyph vs L-glyph",
    "same two bars rearranged; T is symmetric, L is not - mild symmetry confound, noted")
add("b_hook_mirror", "hook_r", dict(kind="hook", sign=1), "hook_l", dict(kind="hook", sign=-1),
    "B_glyph", "hook glyph vs its mirror",
    "pure chirality; matched-by-construction like `mirror`, expected near floor")
add("b_spiral_cw", "cw", dict(kind="spiral", sign=-1), "ccw", dict(kind="spiral", sign=1),
    "B_glyph", "clockwise vs counter-clockwise spiral",
    "chirality again but with a conventional reading direction available")
add("b_s_z", "s_smooth", dict(kind="scurve"), "z_angular", dict(kind="scurve", angular=True),
    "B_glyph", "smooth S vs angular Z",
    "two pseudo-letters, matched extent and stroke width")
add("b_circ_bar", "upright", dict(kind="circbar", deg=0), "tilted", dict(kind="circbar", deg=45),
    "B_glyph", "circled bar upright vs tilted 45",
    "logo-like; identical circular frame and identical bar, only orientation differs, "
    "so ink and extent are exactly matched. Replaces an annulus-vs-disc pair whose "
    "areas were 58% apart and which therefore confounded the glyph cue with size.")

# ---- C. contrast / visual weight ---------------------------------------------
# The family s_contrast belongs to (clean GPT>Claude). Geometry fixed throughout.
add("c_fill_outline", "filled", dict(kind="circle", r=0.82),
    "outline", dict(kind="outline", r=0.82, lw=4),
    "C_contrast", "filled vs outline disc",
    "identical silhouette; only ink differs")
add("c_contr_extreme", "vlight", dict(kind="circle", r=0.82, fc=VLIGHT),
    "vdark", dict(kind="circle", r=0.82, fc=VDARK),
    "C_contrast", "very light vs very dark disc",
    "wider tonal separation than s_contrast, which was LIGHT vs DARK")
add("c_contr_subtle", "light", dict(kind="circle", r=0.82, fc="#B8BDC7"),
    "midgrey", dict(kind="circle", r=0.82, fc="#8B92A0"),
    "C_contrast", "light vs mid-grey disc",
    "narrower separation than s_contrast; tests whether the cue is graded")
add("c_stroke_wt", "thick", dict(kind="outline", r=0.82, lw=9),
    "thin", dict(kind="outline", r=0.82, lw=2),
    "C_contrast", "thick vs thin stroke",
    "visual weight with no fill and no size change")
add("c_weight_blob", "heavy", dict(kind="smoothblob", seed=3, fc=VDARK),
    "light", dict(kind="smoothblob", seed=3, fc=VLIGHT),
    "C_contrast", "same blob, heavy vs light ink",
    "contrast cue on a non-canonical form, so shape cannot carry the answer")

# ---- D. density / numerosity / texture ---------------------------------------
# The family s_density belongs to (clean GPT>Claude).
add("d_dens_extreme", "sparse", dict(kind="dots", n=4, seed=2, r=0.13),
    "dense", dict(kind="dots", n=60, seed=2, r=0.055),
    "D_density", "4 vs 60 dots",
    "wider than s_density's 6 vs 34; expected ceiling, calibrates the top of the range")
add("d_dens_subtle", "sparse", dict(kind="dots", n=12, seed=3, r=0.09),
    "dense", dict(kind="dots", n=20, seed=3, r=0.09),
    "D_density", "12 vs 20 dots",
    "narrow ratio; the informative setting if the extreme version ceilings")
add("d_clustered", "clustered", dict(kind="dots", n=16, seed=5, cluster=True, r=0.09),
    "spread", dict(kind="dots", n=16, seed=6, r=0.09),
    "D_density", "equal count, clustered vs spread",
    "numerosity held EQUAL at 16; only spatial density differs")
add("d_eq_ink", "few_large", dict(kind="dots", n=5, seed=4, grid=True, r=0.20),
    "many_small", dict(kind="dots", n=20, seed=4, grid=True, r=0.10),
    "D_density", "equal total ink, few large vs many small",
    "5*0.20^2 = 20*0.10^2, so filled area matches; numerosity vs element size")
add("d_texture", "fine", dict(kind="stipple", n=340, seed=7, r=0.028),
    "coarse", dict(kind="stipple", n=85, seed=7, r=0.056),
    "D_density", "fine vs coarse stipple in a fixed disc",
    "identical silhouette and near-identical ink; only grain differs")

# ---- E. convexity / closure --------------------------------------------------
# The family s_convexity / c_open_cplx belong to. Step 1b used r_in=0.38 (deep
# spikes) and a 70 deg gap - both coarse. These are the subtle settings.
add("e_convex_mild", "convex", dict(kind="poly", n=7, r=0.9),
    "concave", dict(kind="star", points=7, r_in=0.74, r_out=0.95),
    "E_closure", "convex heptagon vs mildly concave",
    "r_in 0.74 vs Step 1b s_convexity's 0.38; a much shallower indentation")
add("e_open_subtle", "closed", dict(kind="ring", r=0.88, gap=0),
    "open", dict(kind="ring", r=0.88, gap=12),
    "E_closure", "closed ring vs 24 deg gap",
    "Step 1b's s_closure used a 55 deg gap and ceilinged; this is near threshold")
add("e_convex_blob", "convex", dict(kind="smoothblob", seed=14),
    "concave", dict(kind="smoothstar", points=5, r_in=0.76, r_out=0.98),
    "E_closure", "smooth convex blob vs smooth shallow-lobed concave",
    "the concave form is Fourier-smoothed, so BOTH forms are rounded and curvature "
    "cannot separate them - only convexity does. An unsmoothed star here would have "
    "made this a curvature contrast as much as a convexity one.")
add("e_closure_c", "closed", dict(kind="ring", r=0.88, gap=0, frac=0.3),
    "c_shape", dict(kind="ring", r=0.88, gap=48, frac=0.3),
    "E_closure", "closed ring vs C-shape",
    "the classic closure contrast at a mid gap; brackets e_open_subtle")


# ------------------------------------------------------------------ render ----
def make(name, left_spec, right_spec):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=130)
    for ax, spec in zip(axes, (left_spec, right_spec)):
        draw(ax, spec)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.15)
    fig.savefig(os.path.join(OUT, f"{name}.png"), facecolor="white")
    plt.close(fig)


def contact(rows, fname):
    from PIL import Image
    cell_w, cell_h = 460, 232
    sheet = Image.new("RGB", (cell_w + 20, cell_h * len(rows) + 20), "white")
    for i, name in enumerate(rows):
        im = Image.open(os.path.join(OUT, f"{name}_AB.png"))
        im.thumbnail((cell_w - 10, cell_h - 10))
        sheet.paste(im, (10, 10 + i * cell_h))
    sheet.save(os.path.join(OUT, fname))


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, _id1, s1, _id2, s2, *_ in P:
        make(f"{name}_AB", s1, s2)      # identity_1 on the LEFT
        make(f"{name}_BA", s2, s1)      # identity_2 on the LEFT

    manifest = os.path.join(OUT, "manifest.csv")
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stimulus_id", "family", "manipulation",
                    "identity_1", "identity_1_description",
                    "identity_2", "identity_2_description",
                    "image_path_AB", "image_path_BA", "generation_notes"])
        for name, id1, _s1, id2, _s2, fam, manip, note in P:
            lo, hi = manip.split(" vs ") if " vs " in manip else (manip, manip)
            w.writerow([name, fam, manip, id1, lo, id2, hi,
                        f"stimuli/targeted/{name}_AB.png",
                        f"stimuli/targeted/{name}_BA.png", note])

    fams = {}
    for p in P:
        fams.setdefault(p[5], []).append(p[0])
    for fam, names in fams.items():
        contact(names, f"_contact_{fam}.png")

    print(f"{len(P)} targeted pairs -> {len(P) * 2} images in {OUT}")
    for fam, names in sorted(fams.items()):
        print(f"  {fam:12} {len(names):2}  {', '.join(names)}")
    print(f"manifest: {manifest}")
    print("contact sheets: " + ", ".join(f"_contact_{f}.png" for f in sorted(fams)))


if __name__ == "__main__":
    main()
