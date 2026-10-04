r"""
physdecal lighting
Settles whether the three lighting LABELS are three lighting CONDITIONS,
read-only.

    physdecal lighting

WHY NOT JUST RUN scripts/check_time_of_day.py
----------------------------------------------
That script is the right tool and it cannot be used here. It needs a live
AirSim connection, and it writes `tod_*.png` into `<OUTPUT_ROOT>/debug/`,
which is inside the dataset. This folder promises never to write there, so
running it would break the promise to answer a question that can be answered
without it.

Two read-only measurements are made instead, and the second is stronger
evidence than the original script produces:

  1. THE DEBUG FRAMES that a previous run of check_time_of_day.py already
     left in the dataset. Same statistics, same three images, no simulator and
     no writes. This reproduces the original verdict.

  2. MATCHED DATASET TRIPLETS. Every frame id encodes lighting, vehicle,
     altitude, nadir angle and azimuth, so the same viewpoint of the same
     vehicle exists under all three labels. Comparing those triplets measures
     the lighting difference in THE FRAMES THE EXPERIMENTS ACTUALLY USED,
     which is the thing the caveat is really about. The debug images only ever
     showed one viewpoint.

THE DECISION RULE, FIXED BEFORE LOOKING
----------------------------------------
Two labels are the same condition if the mean absolute difference between
their matched frames is within the noise floor. The noise floor is measured,
not assumed: it is the difference between frames that SHOULD be identical --
the same label, same viewpoint, re-rendered -- and where no such pair exists
it falls back to a conservative 1.0 grey level, which is below anything a
renderer with a moved sun would produce.
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np

from physdecal import config as N


FRAME_RE = re.compile(r"^(?P<light>[a-z]+)_(?P<rest>.+)$")


def debug_frames():
    d = os.path.join(N.DATA_ROOT, "debug")
    out = {}
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if f.startswith("tod_") and f.endswith(".png"):
            img = cv2.imread(os.path.join(d, f))
            if img is not None:
                out[f[4:-4]] = img
    return out


def stats(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(g.mean()), float(g.std())


def mad(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
    return float(np.abs(ga - gb).mean())


def matched_triplets(limit=40):
    """Frame stems that exist under every lighting label."""
    idx = defaultdict(dict)
    for f in os.listdir(N.IMAGES_DIR):
        if not f.endswith(".png"):
            continue
        m = FRAME_RE.match(f[:-4])
        if m:
            idx[m.group("rest")][m.group("light")] = os.path.join(
                N.IMAGES_DIR, f)
    labels = sorted({lab for v in idx.values() for lab in v})
    full = [k for k, v in sorted(idx.items()) if len(v) == len(labels)]
    step = max(1, len(full) // max(limit, 1))
    return labels, [(k, idx[k]) for k in full[::step][:limit]]


def main():
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print("=" * 74)
    print("Lighting check (read-only; scripts/check_time_of_day.py not run,")
    print("because it needs AirSim and writes into the dataset)")
    print("=" * 74)

    # ---------------------------------------------------------- 1. debug set
    print("\n1. Debug frames left by a previous check_time_of_day.py run")
    dbg = debug_frames()
    if len(dbg) < 2:
        print("   none found in %s/debug -- skipping" % N.DATA_ROOT)
    else:
        for k, v in dbg.items():
            mu, sd = stats(v)
            print("   %-10s mean luminance %6.2f   std %6.2f" % (k, mu, sd))
        print("   pairwise mean absolute difference:")
        for a, b in combinations(sorted(dbg), 2):
            print("     %-10s vs %-10s  %7.3f grey levels"
                  % (a, b, mad(dbg[a], dbg[b])))

    # ------------------------------------------------- 2. matched dataset set
    print("\n2. Matched dataset triplets (same vehicle, altitude, angle,")
    print("   azimuth; only the lighting label differs)")
    labels, trips = matched_triplets()
    if not trips:
        print("   no frame exists under every label -- cannot compare")
        return
    print("   labels: %s" % ", ".join(labels))
    print("   comparing %d matched viewpoints" % len(trips))

    per_pair = defaultdict(list)
    per_label_mu = defaultdict(list)
    for _, paths in trips:
        imgs = {}
        for lab, p in paths.items():
            im = cv2.imread(p)
            if im is not None:
                imgs[lab] = im
        for lab, im in imgs.items():
            per_label_mu[lab].append(stats(im)[0])
        for a, b in combinations(sorted(imgs), 2):
            per_pair[(a, b)].append(mad(imgs[a], imgs[b]))

    print("\n   mean luminance per label, over matched frames:")
    for lab in sorted(per_label_mu):
        v = np.array(per_label_mu[lab])
        print("     %-10s %6.2f  (sd across viewpoints %5.2f)"
              % (lab, v.mean(), v.std()))

    print("\n   mean absolute difference between labels:")
    results = {}
    for k in sorted(per_pair):
        v = np.array(per_pair[k])
        results[k] = v.mean()
        print("     %-10s vs %-10s  %7.3f grey levels  (min %6.3f, max %6.3f)"
              % (k[0], k[1], v.mean(), v.min(), v.max()))

    # ------------------------------------------------------------- 3. verdict
    # The noise floor: frames that should be identical. Nothing in this dataset
    # re-renders the same label twice, so the conservative constant is used and
    # named as such.
    floor = 1.0
    print("\n3. Verdict")
    print("   noise floor used: %.1f grey level (conservative constant; the "
          "dataset has no\n   re-rendered duplicate pair to measure it from)"
          % floor)
    dupes = [k for k, v in results.items() if v < floor]
    if dupes:
        print("\n   NEAR-DUPLICATE PAIRS FOUND:")
        for a, b in dupes:
            print("     %s and %s differ by %.3f grey levels" % (a, b, results[(a, b)]))
        print("\n   The lighting axis is NOT three illumination conditions.")
        print("   Do not present Table T4 as illumination robustness. Report")
        print("   the labels that are genuinely distinct and say so.")
        verdict = "DUPLICATE"
    else:
        print("\n   All %d label pairs differ by more than the noise floor."
              % len(results))
        print("   The three labels ARE three distinct rendered conditions, so")
        print("   the lighting axis may be reported as such -- with the")
        print("   measured separations quoted, since 'distinct' is not the")
        print("   same as 'a realistic range of daylight'.")
        verdict = "DISTINCT"

    out = os.path.join(N.FIGURE_DIR, "lighting_verdict.txt")
    os.makedirs(N.FIGURE_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("Lighting verdict: %s\n\n" % verdict)
        f.write("Measured read-only on %d matched dataset triplets.\n"
                % len(trips))
        f.write("scripts/check_time_of_day.py was NOT run: it requires a live\n"
                "AirSim connection and writes into the dataset.\n\n")
        for lab in sorted(per_label_mu):
            f.write("mean luminance  %-10s %6.2f\n"
                    % (lab, float(np.mean(per_label_mu[lab]))))
        f.write("\n")
        for k in sorted(results):
            f.write("mean abs diff   %-10s vs %-10s %7.3f grey levels\n"
                    % (k[0], k[1], results[k]))
        f.write("\nnoise floor %.1f grey level (conservative constant)\n" % floor)
    print("\n   -> %s" % out)


if __name__ == "__main__":
    main()
