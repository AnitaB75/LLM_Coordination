"""
Generate shape-pair stimuli for the visual coordination probe.

Each pair is rendered as ONE image, two shapes side by side, neutral gray fill,
equal size, no color. Two versions per pair (A-left/B-right and B-left/A-right)
so we can randomize position per sample and separate shape-preference from
position-preference.

Pairs:
  proto   circle vs square      -> prototypicality gap (predict: focal point)
  mirror  blob vs its mirror     -> perceptual symmetry (predict: 50/50 scatter)
  rot     blob vs its 90 rot     -> perceptual symmetry
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, RegularPolygon, Polygon

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stimuli", "visual")
os.makedirs(OUT, exist_ok=True)
FILL, EDGE = "#9AA3B2", "#3A4152"   # neutral slate, no hue salience


def blob_xy(seed, n=9, r0=0.62, jit=0.33):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = r0 + jit * rng.random(n)
    return np.c_[r * np.cos(ang), r * np.sin(ang)]


def rot(xy, deg):
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return xy @ R.T


def draw_shape(ax, kind, xy=None):
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal"); ax.axis("off")
    kw = dict(facecolor=FILL, edgecolor=EDGE, linewidth=3)
    if kind == "circle":
        ax.add_patch(Circle((0, 0), 0.82, **kw))
    elif kind == "square":
        ax.add_patch(Rectangle((-0.72, -0.72), 1.44, 1.44, **kw))
    elif kind == "triangle":
        ax.add_patch(RegularPolygon((0, 0), 3, radius=0.9, orientation=0, **kw))
    elif kind == "poly":
        ax.add_patch(Polygon(xy, closed=True, **kw))


def make_pair(name, left_spec, right_spec):
    """Each spec is ('circle'|'square'|'triangle') or ('poly', xy). Save one PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=130)
    for ax, spec in zip(axes, (left_spec, right_spec)):
        if isinstance(spec, tuple) and spec[0] == "poly":
            draw_shape(ax, "poly", spec[1])
        else:
            draw_shape(ax, spec)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.15)
    path = os.path.join(OUT, f"{name}.png")
    fig.savefig(path, facecolor="white"); plt.close(fig)
    print("wrote", os.path.basename(path))


B = blob_xy(seed=7)                # the base irregular blob
B_MIR = B * np.array([-1, 1])      # mirror across vertical axis
B_ROT = rot(B, 90)                 # 90-degree rotation

PAIRS = {
    "proto":  ("circle", "square"),
    "mirror": (("poly", B), ("poly", B_MIR)),
    "rot":    (("poly", B), ("poly", B_ROT)),
}

for name, (a, b) in PAIRS.items():
    make_pair(f"{name}_AB", a, b)   # A left, B right
    make_pair(f"{name}_BA", b, a)   # B left, A right (position counterbalance)

print("done ->", OUT)
