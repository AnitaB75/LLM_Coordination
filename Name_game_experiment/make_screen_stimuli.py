"""
Generate a screening battery of binary stimulus pairs for Step 1b.

Two families:

  SINGLE  one visual cue varied, everything else matched. Establishes which cues
          each model uses at all. Expect many of these at ceiling for both models
          (they agree on the obvious focal option) - that is the point: they
          calibrate the cue inventory.

  COMPETE two cues placed in OPPOSITION - option 1 wins on cue A, option 2 wins
          on cue B. A model that weights symmetry picks one, a model that weights
          size picks the other. This is where model-by-stimulus crossovers should
          appear, and single-cue contrasts will not produce them.

Output matches the existing composite format exactly - 1040x520 PNG, two panels
side by side, `_AB` and `_BA` variants - so `step1_form_choice.py` runs on it
unchanged.

  stimuli/screen/<pair>_AB.png , <pair>_BA.png
  stimuli/screen/manifest.csv          pair, identity_1, identity_2, family, cue_1, cue_2
  stimuli/screen/_contact_sheet_*.png  for eyeballing before spending calls

Run:  python3 make_screen_stimuli.py
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Ellipse, Wedge, FancyArrow

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stimuli", "screen")
FILL, EDGE = "#9AA3B2", "#3A4152"       # same neutral slate as the existing battery
LIGHT, DARK = "#C9CEd6", "#5A6172"

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
    dense = np.interp(np.linspace(0, len(zs), 200), np.arange(len(zs) + 1),
                      np.r_[zs, zs[:1]].real) + 1j * np.interp(
        np.linspace(0, len(zs), 200), np.arange(len(zs) + 1), np.r_[zs, zs[:1]].imag)
    return np.c_[dense.real, dense.imag]


def draw(ax, spec):
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal"); ax.axis("off")
    kind = spec["kind"]
    fc = spec.get("fc", FILL)
    kw = dict(facecolor=fc, edgecolor=EDGE, linewidth=3)

    if kind == "circle":
        ax.add_patch(Circle((0, 0), spec.get("r", 0.82), **kw))
    elif kind == "square":
        s = spec.get("s", 1.44)
        ax.add_patch(Rectangle((-s / 2, -s / 2), s, s, **kw))
    elif kind == "rect":
        w, h = spec["w"], spec["h"]
        ax.add_patch(Rectangle((-w / 2, -h / 2), w, h, **kw))
    elif kind == "ellipse":
        ax.add_patch(Ellipse((0, 0), spec["w"], spec["h"], **kw))
    elif kind == "poly":
        ax.add_patch(Polygon(poly_xy(spec["n"], spec.get("r", 0.9), spec.get("rot", 0.0),
                                     spec.get("jitter", 0.0), spec.get("seed", 0)),
                             closed=True, **kw))
    elif kind == "star":
        ax.add_patch(Polygon(star_xy(spec["points"], spec.get("r_out", 0.95),
                                     spec.get("r_in", 0.45), spec.get("rot", 0.0)),
                             closed=True, **kw))
    elif kind == "blob":
        ax.add_patch(Polygon(blob_xy(spec["seed"], jit=spec.get("jit", 0.33)) * spec.get("scale", 1.3),
                             closed=True, **kw))
    elif kind == "smoothblob":
        ax.add_patch(Polygon(smooth_closed(blob_xy(spec["seed"]) * spec.get("scale", 1.3)),
                             closed=True, **kw))
    elif kind == "annulus":
        r = spec.get("r", 0.85)
        ax.add_patch(Wedge((0, 0), r, 0, 360, width=r * spec.get("frac", 0.38), **kw))
    elif kind == "arc":                       # open (broken) ring
        r = spec.get("r", 0.85)
        ax.add_patch(Wedge((0, 0), r, spec.get("gap", 55), 360 - spec.get("gap", 55),
                           width=r * 0.38, **kw))
    elif kind == "dots":
        rng = np.random.default_rng(spec.get("seed", 0))
        n = spec["n"]
        if spec.get("grid"):
            side = int(np.ceil(np.sqrt(n)))
            g = np.linspace(-0.8, 0.8, side)
            pts = np.array([(x, y) for y in g for x in g])[:n]
        else:
            pts = rng.uniform(-0.85, 0.85, size=(n, 2))
        for x, y in pts:
            ax.add_patch(Circle((x, y), spec.get("r", 0.11), facecolor=fc,
                                edgecolor=EDGE, linewidth=1.4))
    elif kind == "cluster":                   # n equal circles, area-matched total
        n = spec["n"]
        r = spec.get("total_r", 0.82) / np.sqrt(n)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        d = 0.0 if n == 1 else 0.82 - r
        for a in ang:
            ax.add_patch(Circle((d * np.cos(a), d * np.sin(a)), r, **kw))
    elif kind == "stripes":
        n, horiz = spec["n"], spec.get("horiz", False)
        edges = np.linspace(-0.9, 0.9, n * 2 + 1)
        for i in range(n):
            lo, hi = edges[2 * i], edges[2 * i + 1]
            if horiz:
                ax.add_patch(Rectangle((-0.9, lo), 1.8, hi - lo, facecolor=fc,
                                       edgecolor=EDGE, linewidth=1.2))
            else:
                ax.add_patch(Rectangle((lo, -0.9), hi - lo, 1.8, facecolor=fc,
                                       edgecolor=EDGE, linewidth=1.2))
    elif kind == "nested":
        for i, rr in enumerate(np.linspace(spec.get("r", 0.85), 0.2, spec["rings"])):
            ax.add_patch(Circle((0, 0), rr, facecolor="none", edgecolor=EDGE, linewidth=3))
    elif kind == "arrow":
        a = np.deg2rad(spec.get("deg", 90))
        ax.add_patch(FancyArrow(-0.7 * np.cos(a), -0.7 * np.sin(a),
                                1.4 * np.cos(a), 1.4 * np.sin(a),
                                width=0.28, head_width=0.62, head_length=0.5,
                                length_includes_head=True, **kw))
    else:
        raise ValueError(kind)


# ------------------------------------------------------------------- pairs ----
# (pair, identity_1, spec_1, identity_2, spec_2, family, cue_1, cue_2)
P = []
def add(name, id1, s1, id2, s2, family, cue1, cue2=""):
    P.append((name, id1, s1, id2, s2, family, cue1, cue2))

# ---- SINGLE-CUE: which cues does each model use at all? ----
add("s_symmetry",  "symmetric",  dict(kind="poly", n=8, r=0.9),
                   "asymmetric", dict(kind="blob", seed=11), "single", "symmetry")
add("s_complexity","simple",     dict(kind="poly", n=3, r=0.95),
                   "complex",    dict(kind="poly", n=20, r=0.9), "single", "complexity")
add("s_regularity","regular",    dict(kind="poly", n=6, r=0.9),
                   "irregular",  dict(kind="poly", n=6, r=0.9, jitter=0.45, seed=5), "single", "regularity")
add("s_curvature", "smooth",     dict(kind="smoothblob", seed=3),
                   "spiky",      dict(kind="star", points=9, r_in=0.4), "single", "curvature")
add("s_convexity", "convex",     dict(kind="poly", n=7, r=0.9),
                   "concave",    dict(kind="star", points=7, r_in=0.38), "single", "convexity")
add("s_elongation","compact",    dict(kind="circle", r=0.8),
                   "elongated",  dict(kind="ellipse", w=2.0, h=0.66), "single", "elongation")
add("s_size",      "small",      dict(kind="circle", r=0.42),
                   "large",      dict(kind="circle", r=0.95), "single", "size")
add("s_tilt",      "upright",    dict(kind="square", s=1.35),
                   # rot=0 puts 4-gon vertices at the cardinal points -> diamond.
                   # rot=pi/4 would put them on the diagonals, i.e. an upright square.
                   "tilted",     dict(kind="poly", n=4, r=0.95, rot=0.0), "single", "orientation")
add("s_closure",   "closed",     dict(kind="annulus"),
                   "open",       dict(kind="arc"), "single", "closure")
add("s_hole",      "solid",      dict(kind="circle", r=0.85),
                   "holed",      dict(kind="annulus"), "single", "hole")
add("s_count",     "one",        dict(kind="cluster", n=1),
                   "many",       dict(kind="cluster", n=7), "single", "numerosity")
add("s_density",   "sparse",     dict(kind="dots", n=6, seed=2),
                   "dense",      dict(kind="dots", n=34, seed=2, r=0.07), "single", "density")
add("s_alignment", "aligned",    dict(kind="dots", n=16, grid=True),
                   "scattered",  dict(kind="dots", n=16, seed=7), "single", "alignment")
add("s_stripe_or", "horizontal", dict(kind="stripes", n=5, horiz=True),
                   "vertical",   dict(kind="stripes", n=5), "single", "line orientation")
add("s_contrast",  "light",      dict(kind="circle", r=0.82, fc=LIGHT),
                   "dark",       dict(kind="circle", r=0.82, fc=DARK), "single", "contrast")
add("s_arrow",     "up",         dict(kind="arrow", deg=90),
                   "right",      dict(kind="arrow", deg=0), "single", "semantic orientation")

# ---- COMPETING-CUE: two cues in opposition — where crossovers should live ----
add("c_sym_size",   "sym_small",    dict(kind="poly", n=8, r=0.5),
                    "asym_large",   dict(kind="blob", seed=11, scale=1.75), "compete", "symmetry", "size")
add("c_sym_count",  "sym_one",      dict(kind="cluster", n=1),
                    "asym_many",    dict(kind="dots", n=9, seed=4), "compete", "symmetry", "numerosity")
add("c_reg_size",   "reg_small",    dict(kind="poly", n=6, r=0.5),
                    "irreg_large",  dict(kind="poly", n=6, r=1.0, jitter=0.45, seed=5), "compete", "regularity", "size")
add("c_spiky_size", "spiky_small",  dict(kind="star", points=7, r_out=0.55, r_in=0.22),
                    "smooth_large", dict(kind="circle", r=1.0), "compete", "curvature", "size")
add("c_smooth_sym", "smooth_asym",  dict(kind="smoothblob", seed=3),
                    "spiky_sym",    dict(kind="star", points=8, r_in=0.42), "compete", "curvature", "symmetry")
add("c_hole_size",  "holed_small",  dict(kind="annulus", r=0.5),
                    "solid_large",  dict(kind="circle", r=1.0), "compete", "hole", "size")
add("c_count_size", "many_small",   dict(kind="dots", n=12, seed=8, r=0.09),
                    "one_large",    dict(kind="circle", r=1.0), "compete", "numerosity", "size")
add("c_tilt_reg",   "upright_irreg",dict(kind="poly", n=5, r=0.9, jitter=0.4, seed=9),
                    "tilted_reg",   dict(kind="poly", n=5, r=0.9, rot=np.pi / 5), "compete", "orientation", "regularity")
add("c_contr_size", "dark_small",   dict(kind="circle", r=0.48, fc=DARK),
                    "light_large",  dict(kind="circle", r=1.0, fc=LIGHT), "compete", "contrast", "size")
add("c_elong_sym",  "elong_sym",    dict(kind="ellipse", w=2.05, h=0.6),
                    "compact_asym", dict(kind="blob", seed=21), "compete", "elongation", "symmetry")
add("c_open_cplx",  "open_simple",  dict(kind="arc", gap=70),
                    "closed_cplx",  dict(kind="star", points=12, r_in=0.55), "compete", "closure", "complexity")
add("c_dens_align", "dense_scatter",dict(kind="dots", n=30, seed=6, r=0.07),
                    "sparse_grid",  dict(kind="dots", n=9, grid=True), "compete", "density", "alignment")
add("c_star_circle","star_eqarea",  dict(kind="star", points=6, r_out=1.05, r_in=0.52),
                    "circle_eqarea",dict(kind="circle", r=0.72), "compete", "curvature", "area-matched")
add("c_nested_big", "nested_small", dict(kind="nested", rings=4, r=0.55),
                    "single_large", dict(kind="circle", r=1.0), "compete", "nestedness", "size")

# ------------------------------------------------------------------ render ----
def make(name, left_spec, right_spec):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=130)
    for ax, spec in zip(axes, (left_spec, right_spec)):
        draw(ax, spec)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.15)
    fig.savefig(os.path.join(OUT, f"{name}.png"), facecolor="white")
    plt.close(fig)


def contact(rows, fname, title):
    from PIL import Image
    cell_w, cell_h = 420, 210
    sheet = Image.new("RGB", (cell_w + 20, cell_h * len(rows) + 20), "white")
    for i, name in enumerate(rows):
        im = Image.open(os.path.join(OUT, f"{name}_AB.png"))
        im.thumbnail((cell_w - 10, cell_h - 10))
        sheet.paste(im, (10, 10 + i * cell_h))
    sheet.save(os.path.join(OUT, fname))


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, id1, s1, id2, s2, *_ in P:
        make(f"{name}_AB", s1, s2)      # identity_1 on the LEFT
        make(f"{name}_BA", s2, s1)      # identity_2 on the LEFT
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "identity_1", "identity_2", "family", "cue_1", "cue_2"])
        for name, id1, _s1, id2, _s2, fam, c1, c2 in P:
            w.writerow([name, id1, id2, fam, c1, c2])

    singles = [p[0] for p in P if p[5] == "single"]
    competes = [p[0] for p in P if p[5] == "compete"]
    contact(singles, "_contact_single.png", "single-cue")
    contact(competes, "_contact_compete.png", "competing-cue")
    print(f"{len(P)} pairs ({len(singles)} single-cue, {len(competes)} competing-cue)"
          f" -> {len(P) * 2} images in {OUT}")
    print("contact sheets: _contact_single.png, _contact_compete.png")


if __name__ == "__main__":
    main()
