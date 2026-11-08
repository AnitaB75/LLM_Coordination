"""
Derive SINGLE-OPTION images from the existing side-by-side composites.

The six binary pairs (proto, nd2, inkblot, rot, noise, mirror) exist in this repo
only as one 1040x520 composite per variant, showing both options next to each
other. A label-free protocol has to show the two options as separate images, so
this script splits each composite into its two panels.

Method, per pair (read from the `_AB` composite, where the left panel is the
identity listed first in SETS):

  1. Find the single internal all-white column gap and split there. Every
     composite has exactly one, so the panel boundary is measured, not assumed
     at the midpoint (nd2's panels sit off-centre and would be clipped by a
     naive half-split).
  2. Crop each panel to its ink bounding box. This removes the ~14px x-offset
     difference between the left and right matplotlib axes, which would
     otherwise make the two singles differ in framing.
  3. Paste both onto a common square canvas sized from the LARGER of the two
     bounding boxes plus a margin, centred. Relative scale is preserved — the
     two options keep their true size difference, because rescaling each to fill
     its canvas would erase a real stimulus property (and manufacture a cue).

NON-DESTRUCTIVE: writes to stimuli/singles/, touches nothing existing.

Run:  python3 make_single_option_stimuli.py
      python3 make_single_option_stimuli.py --contact-sheet   # + gallery for eyeballing
"""
import os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stimuli", "singles")
MARGIN = 40          # px of white around the larger bbox
WHITE_CUTOFF = 245   # a pixel counts as ink if any channel is below this

# pair -> (subdir, identity in the LEFT panel of _AB, identity in the RIGHT panel)
SETS = {
    "proto":   ("visual",    "circle",  "square"),
    "nd2":     ("mountains", "p2a",     "p2b"),
    "inkblot": ("visual2",   "ib-A",    "ib-B"),
    "rot":     ("visual",    "blob",    "blob-rot90"),
    "noise":   ("visual2",   "noise-A", "noise-B"),
    "mirror":  ("visual",    "blob",    "blob-mirror"),
}


def ink_mask(arr):
    return arr.min(axis=2) < WHITE_CUTOFF


def split_columns(arr):
    """Column index to split at: the middle of the one internal all-white gap."""
    ink = ink_mask(arr).any(axis=0)
    cols = np.where(ink)[0]
    gaps = []
    prev = cols[0]
    for c in cols[1:]:
        if c - prev > 20:
            gaps.append((prev, c))
        prev = c
    if len(gaps) != 1:
        sys.exit(f"expected exactly one internal gap, found {len(gaps)}: {gaps}")
    lo, hi = gaps[0]
    return (lo + hi) // 2


def bbox(arr):
    m = ink_mask(arr)
    rows, cols = np.where(m)
    return cols.min(), rows.min(), cols.max() + 1, rows.max() + 1


def main():
    os.makedirs(OUT, exist_ok=True)
    made = []
    for pair, (sub, ident_left, ident_right) in SETS.items():
        path = os.path.join(HERE, "stimuli", sub, f"{pair}_AB.png")
        im = Image.open(path).convert("RGB")
        arr = np.array(im)
        cut = split_columns(arr)

        panels = []
        for x0, x1, identity in ((0, cut, ident_left), (cut, im.width, ident_right)):
            sub_arr = arr[:, x0:x1]
            bx0, by0, bx1, by1 = bbox(sub_arr)
            panels.append((im.crop((x0 + bx0, by0, x0 + bx1, by1)), identity))

        # common canvas from the larger bbox — preserves relative scale
        side = max(max(p.width, p.height) for p, _ in panels) + 2 * MARGIN
        for crop, identity in panels:
            canvas = Image.new("RGB", (side, side), "white")
            canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
            fn = os.path.join(OUT, f"{pair}__{identity}.png")
            canvas.save(fn)
            made.append((pair, identity, crop.size, canvas.size))
        print(f"{pair:8} split@{cut:4}  canvas {side}x{side}  "
              f"{panels[0][1]} {panels[0][0].size} / {panels[1][1]} {panels[1][0].size}")

    if "--contact-sheet" in sys.argv:
        cell = 240
        rows = len(SETS)
        sheet = Image.new("RGB", (cell * 2 + 30, cell * rows + 30), "white")
        for r, pair in enumerate(SETS):
            for c, identity in enumerate(SETS[pair][1:]):
                im = Image.open(os.path.join(OUT, f"{pair}__{identity}.png"))
                im.thumbnail((cell - 20, cell - 20))
                sheet.paste(im, (10 + c * cell + (cell - 20 - im.width) // 2,
                                 10 + r * cell + (cell - 20 - im.height) // 2))
        sheet.save(os.path.join(OUT, "_contact_sheet.png"))
        print(f"\ncontact sheet -> {os.path.join(OUT, '_contact_sheet.png')}")

    print(f"\nwrote {len(made)} single-option images to {OUT}")


if __name__ == "__main__":
    main()
