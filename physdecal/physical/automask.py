r"""
physdecal automask
Draft ground-truth vehicle masks for a physical session, for review.

    physdecal automask --dir out/physical/<session>
    physdecal automask --dir out/physical/<session> --redo

The session's photographs are a toy car on a plain floor, so the car is found
by its colour distance from the floor (CIELAB, floor = median of the image
border), the largest connected region is kept (which drops the fiducial
markers), holes are filled, and GrabCut refines the edge from that initial
mask. It is the same GrabCut step `physdecal physical annotate` runs, seeded from
colour instead of a dragged box.

Masks are written where `annotate` would write them, one per scene from the
clean photograph, and a contact sheet of every mask outlined on its photograph
is written for review. A mask that is wrong is redrawn by hand with
`physdecal physical annotate --dir <session> --redo`; nothing here is final until
it has been looked at.
"""

import argparse
import glob
import os

import cv2
import numpy as np


def draft_mask(bgr, thr=30, seed=None):
    """seed="green": pick the region holding the most saturated-green pixels
    (a camouflage-painted car), for wide frames where floor glare or furniture
    stands out from the floor more than the small car does."""
    h, w = bgr.shape[:2]
    s = 1000.0 / max(h, w)                      # work at ~1000 px, scale back
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(10, int(0.05 * min(small.shape[:2])))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    floor = np.median(border, 0)
    # chroma counts double: a soft shadow on the floor is darker, not a
    # different colour, and must not join the car
    d = np.sqrt(0.25 * (lab[..., 0] - floor[0]) ** 2
                + 2.0 * ((lab[..., 1] - floor[1]) ** 2 + (lab[..., 2] - floor[2]) ** 2))
    fg = (d > thr).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return None
    # the car is the region with the most strongly off-floor pixels, not the
    # largest one: a wide shot can hold a bigger patch of differently lit floor
    seeds = d > 60
    if seed == "green":
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        seeds = ((hsv[..., 0] > 30) & (hsv[..., 0] < 80)
                 & (hsv[..., 1] > 90) & (hsv[..., 2] > 60))
    score = np.bincount(lbl[seeds], minlength=n)
    score[0] = 0
    if score.max() == 0:                        # no strong pixels: largest region
        score = stats[:, cv2.CC_STAT_AREA].copy()
        score[0] = 0
    k = int(np.argmax(score))
    car = (lbl == k).astype(np.uint8)
    cnts, _ = cv2.findContours(car, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    car = cv2.drawContours(np.zeros_like(car), cnts, -1, 1, -1)   # fill holes

    gc = np.where(car > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    gc[cv2.erode(car, np.ones((15, 15), np.uint8)) > 0] = cv2.GC_FGD
    ring = cv2.dilate(car, np.ones((25, 25), np.uint8)) - car
    gc[ring > 0] = cv2.GC_PR_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    out = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(out)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = (lbl == k).astype(np.uint8)
    # A toy car seen from above is a convex outline. Its pale paint can match
    # the floor and leave notches in the mask; the convex hull closes them
    # without readmitting the shadow, which lies outside the body.
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(cnts))
    out = cv2.fillConvexPoly(np.zeros_like(out), hull, 1)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST) * 255


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--condition", default="clean", choices=["clean", "patched"],
                    help="patched: one mask per patched photograph too, for "
                         "sessions where the camera moved within a pair")
    ap.add_argument("--thr", type=float, default=30,
                    help="colour distance from the floor that counts as car; "
                         "lower it for pale cars on a pale floor")
    ap.add_argument("--seed", default=None, choices=["green"],
                    help="choose the car by its green paint (see draft_mask)")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="with --redo, only redraw these scenes")
    a = ap.parse_args()
    suffix = "" if a.condition == "clean" else "_patched"

    ann = os.path.join(a.dir, "annotations")
    os.makedirs(ann, exist_ok=True)
    tiles = []
    for p in sorted(glob.glob(os.path.join(a.dir, "*__%s__00.jpg" % a.condition))):
        scene = os.path.basename(p).split("__")[0]
        out = os.path.join(ann, "%s%s_mask.png" % (scene, suffix))
        bgr = cv2.imread(p)
        redo = a.redo and (not a.scenes or scene in a.scenes)
        if os.path.isfile(out) and not redo:
            m = cv2.imread(out, cv2.IMREAD_GRAYSCALE)
        else:
            m = draft_mask(bgr, a.thr, a.seed)
            if m is None:
                print("  %s: no vehicle found, annotate by hand" % scene)
                continue
            cv2.imwrite(out, m)
        # review tile: crop round the mask, outline it in magenta
        ys, xs = np.nonzero(m)
        pad = int(0.25 * max(np.ptp(ys), np.ptp(xs)))
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, m.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, m.shape[1])
        crop = bgr[y0:y1, x0:x1].copy()
        cnts, _ = cv2.findContours((m[y0:y1, x0:x1] > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, cnts, -1, (255, 0, 255), max(2, crop.shape[1] // 150))
        crop = cv2.resize(crop, (360, int(360 * crop.shape[0] / crop.shape[1])))
        crop = cv2.copyMakeBorder(crop, 0, 360 - crop.shape[0] if crop.shape[0] < 360 else 0,
                                  0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))[:360]
        cv2.putText(crop, scene.replace("t00_", ""), (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(crop)
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    path = os.path.join(a.dir, "mask_review%s.jpg" % suffix)
    cv2.imwrite(path, sheet)
    print("wrote %d masks, review sheet %s" % (len(tiles), path))


if __name__ == "__main__":
    main()
