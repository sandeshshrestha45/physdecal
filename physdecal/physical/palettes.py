"""
Build the ink sets that test hypothesis H1.

H1 says the measured gamut helps closed-set segmentation because a reproducible
palette is also a low-chroma, low-frequency one. The controlled pair cannot
test it: the measured set differs from the cube in BOTH chromatic extent and
cardinality, and both reduce the optimiser's freedom.

This writes three families of ink set, each a CSV of sRGB triples in [0,1] that
physdecal optimize --inks reads directly. Cardinality is the row count, so the
k-means reduction inside load_inks never fires and the count is exact.

  chroma_g<gamma>   cardinality fixed at the measured set's, chroma scaled by
                    gamma about the neutral axis with L* and hue held. gamma
                    1.00 reproduces the measured set and is the control.
  card_k<k>         chroma fixed at the measured level by drawing real
                    measured colours, cardinality varied by k-means in CIELAB.
  rand_s<seed>      cardinality, L* and C* copied from the measured set,
                    hue angles redrawn uniformly. Matched on everything H1
                    says should matter except where the colours sit.

Chroma is scaled in CIELAB and the result is clipped to sRGB, so the achieved
mean chroma is not the requested one at high gamma. The achieved value is what
is printed and what the paper must quote.
"""
import argparse
import csv
import os

import numpy as np
import torch

from physdecal import config as N
from physdecal.core import decal as D

OUT = os.path.join(N.REFERENCE_DIR, "palettes")


def _lab(rgb):
    t = torch.as_tensor(rgb, dtype=torch.float32)
    return D.rgb_to_lab(t.T.reshape(1, 3, -1, 1)).reshape(3, -1).T.numpy()


def _rgb(lab):
    t = torch.as_tensor(lab, dtype=torch.float32)
    out = D.lab_to_rgb(t.T.reshape(1, 3, -1, 1)).reshape(3, -1).T.numpy()
    return np.clip(out, 0.0, 1.0)


def _stats(rgb):
    lab = _lab(rgb)
    c = np.hypot(lab[:, 1], lab[:, 2])
    return len(rgb), float(lab[:, 0].mean()), float(c.mean())


def _write(name, rgb):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in np.asarray(rgb, dtype=np.float64):
            w.writerow(["%.6f" % x for x in r])
    k, L, C = _stats(rgb)
    print("  %-14s k=%2d  mean L*=%5.1f  mean C*=%5.1f  ->  %s"
          % (name, k, L, C, os.path.relpath(p)))
    return p


def measured_base(k=None):
    """The ink set the measured-gamut run actually used, before the anchor."""
    return D.load_inks(device="cpu", k=k or N.INK_COUNT).numpy()


def main():
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    base = measured_base()
    k0, L0, C0 = _stats(base)
    print("\nmeasured baseline: k=%d, mean L*=%.1f, mean C*=%.1f" % (k0, L0, C0))

    print("\nchroma sweep, cardinality fixed at %d" % k0)
    lab = _lab(base)
    for g in (0.50, 1.00, 1.50, 2.00):
        m = lab.copy()
        m[:, 1:] *= g
        _write("chroma_g%03d" % round(g * 100), _rgb(m))

    print("\ncardinality sweep, colours drawn from the measured chart")
    for k in (3, 5, 16, 32):
        _write("card_k%02d" % k, measured_base(k=k))

    print("\nrandom palettes, cardinality/L*/C* matched to the measured set")
    C = np.hypot(lab[:, 1], lab[:, 2])
    for seed in (11, 22, 33):
        rng = np.random.default_rng(seed)
        h = rng.uniform(0.0, 2.0 * np.pi, size=len(lab))
        m = np.stack([lab[:, 0], C * np.cos(h), C * np.sin(h)], axis=1)
        _write("rand_s%02d" % seed, _rgb(m))

    print("\nNote: chroma is scaled in CIELAB then clipped to sRGB, so the "
          "achieved\nmean C* above is what to quote, not the requested gamma.")


if __name__ == "__main__":
    main()
