r"""
physdecal inkchart
What your printer can actually make, measured instead of assumed.

    physdecal inkchart make
    # print inkchart.pdf at 100 percent, colour management OFF, then scan it
    physdecal inkchart read --scan scan.png

WHY THIS EXISTS
---------------
PRINTABLE_COLOURS_CSV is the set of colours L_ink snaps the decal toward. The
file this experiment ships is a 6x6x6 cube: an evenly spaced grid of RGB, which
is a mathematical object and not any printer's gamut. Optimising against it
produces a patch made of colours the printer may be unable to hit, and an NPS
computed from it is a number about a cube.

So: chart the requested colours, print them, scan the print, and read back what
the paper actually holds. The measured set replaces the cube and INK_MEASURED
becomes a true statement rather than a promise.

THE ORDER MATTERS AND IT IS THE ONE IRREVERSIBLE THING HERE
-----------------------------------------------------------
printable.csv is an INPUT TO THE OPTIMISER, not only a reporting artefact.
Scanning a chart after a patch is optimised does not change that patch; it only
tells you how wrong its palette was. Measure first, then optimise, then export
and print the decal.

WHAT THE MARKERS ARE FOR
------------------------
Four ArUco markers sit at the corners of the chart, from the same dictionary
the decal sheets use. A scan is never perfectly square, centred or scaled, and
counting pixels in from the page edge fails the first time someone scans at a
slight angle or with the lid ajar. The markers give the scan a homography back
to chart millimetres, so every patch is sampled where it actually landed rather
than where it was meant to land.

WHAT THIS DOES NOT MEASURE
--------------------------
A flatbed scanner has its own illuminant and its own colour response, so what
comes back is what the SCANNER thinks the printer made, not absolute colour.
That is adequate for a soft loss term over a palette and it is not
spectrophotometry. Report it as scanner RGB.
"""

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

from physdecal import config as N


CHART_DIR = os.path.join(N.REFERENCE_DIR, "inkchart")
META = "inkchart.json"
MARKER_MM = 15.0           # large enough that any scan resolution decodes it
MARKER_GAP_MM = 3.0        # clear space between the markers and the grid
GUTTER_MM = 1.5            # white space between patches, so edges never bleed
SAMPLE_FRAC = 0.5          # centred fraction of each patch that is sampled
WARP_PX_PER_MM = 8.0       # resolution the scan is rectified to


def mm_to_px(mm, dpi):
    return int(round(mm / 25.4 * dpi))


# ------------------------------------------------------------------ geometry

def _grid(n, w_mm, h_mm):
    """Largest square cell that fits n patches in w x h millimetres."""
    cell = 30.0
    while cell >= 4.0:
        cols, rows = int(w_mm // cell), int(h_mm // cell)
        if cols * rows >= n:
            return cols, rows, cell
        cell -= 0.25
    raise SystemExit(
        "%d patches do not fit on %s at a usable size. Use a larger sheet."
        % (n, N.SHEET))


def layout_for(n, sheet=None):
    """
    Chart geometry in millimetres, origin at the top left of the content area.

    The markers sit at its corners, ids 0..3 clockwise from top left, which is
    the order export_print.py writes them and capture.py expects them in.
    """
    sheet = sheet or N.SHEET
    sw, sh = N.SHEET_MM[sheet]
    cw = sw - 2 * N.SHEET_MARGIN_MM
    ch = sh - 2 * N.SHEET_MARGIN_MM

    inset = MARKER_MM + MARKER_GAP_MM
    cols, rows, cell = _grid(n, cw - 2 * inset, ch - 2 * inset)
    ox = inset + (cw - 2 * inset - cols * cell) / 2.0
    oy = inset + (ch - 2 * inset - rows * cell) / 2.0

    h = MARKER_MM / 2.0
    centres = {0: (h, h), 1: (cw - h, h), 2: (cw - h, ch - h), 3: (h, ch - h)}
    return {"sheet": sheet, "content_mm": [cw, ch], "marker_mm": MARKER_MM,
            "marker_centres_mm": {str(k): list(v) for k, v in centres.items()},
            "cols": cols, "rows": rows, "cell_mm": cell,
            "grid_origin_mm": [ox, oy], "n": n}


def cell_centre_mm(lay, k):
    ox, oy = lay["grid_origin_mm"]
    c, r = k % lay["cols"], k // lay["cols"]
    half = lay["cell_mm"] / 2.0
    return ox + c * lay["cell_mm"] + half, oy + r * lay["cell_mm"] + half


# ---------------------------------------------------------------------- make

def load_requested(path=None):
    path = path or N.PRINTABLE_COLOURS_CSV
    if not os.path.isfile(path):
        raise SystemExit("No %s to chart." % path)
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32).reshape(-1, 3)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0, 1), path


def make(colours_csv=None, sheet=None, dpi=None, out_dir=None):
    sheet = sheet or N.SHEET
    dpi = dpi or N.PRINT_DPI
    rgb, src = load_requested(colours_csv)
    lay = layout_for(len(rgb), sheet)
    out_dir = out_dir or CHART_DIR
    os.makedirs(out_dir, exist_ok=True)

    sw, sh = N.SHEET_MM[sheet]
    page = np.full((mm_to_px(sh, dpi), mm_to_px(sw, dpi), 3), 255, np.uint8)
    off = N.SHEET_MARGIN_MM

    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, N.FIDUCIAL_DICT))
    m_px = mm_to_px(MARKER_MM, dpi)
    for i, (cx, cy) in lay["marker_centres_mm"].items():
        img = cv2.aruco.generateImageMarker(adict, int(i), m_px)
        x = mm_to_px(off + cx - MARKER_MM / 2.0, dpi)
        y = mm_to_px(off + cy - MARKER_MM / 2.0, dpi)
        page[y:y + m_px, x:x + m_px] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    half = (lay["cell_mm"] - GUTTER_MM) / 2.0
    for k, c in enumerate(rgb):
        cx, cy = cell_centre_mm(lay, k)
        x0, y0 = mm_to_px(off + cx - half, dpi), mm_to_px(off + cy - half, dpi)
        x1, y1 = mm_to_px(off + cx + half, dpi), mm_to_px(off + cy + half, dpi)
        bgr = (float(c[2]) * 255, float(c[1]) * 255, float(c[0]) * 255)
        cv2.rectangle(page, (x0, y0), (x1, y1), bgr, -1)

    cv2.putText(page, "physdecal inkchart  %d patches  %s  %d dpi  from %s"
                % (len(rgb), sheet, dpi, os.path.basename(src)),
                (mm_to_px(off, dpi), page.shape[0] - mm_to_px(3.0, dpi)),
                cv2.FONT_HERSHEY_SIMPLEX, dpi / 500.0, (40, 40, 40),
                max(1, dpi // 300))

    png = os.path.join(out_dir, "inkchart.png")
    cv2.imwrite(png, page)
    pdf = os.path.join(out_dir, "inkchart.pdf")
    Image.open(png).convert("RGB").save(pdf, "PDF", resolution=float(dpi))

    lay.update({"dpi": dpi, "source_csv": src,
                "colours": [[round(float(v), 4) for v in c] for c in rgb]})
    with open(os.path.join(out_dir, META), "w", encoding="utf-8") as f:
        json.dump(lay, f, indent=2)

    print("inkchart -> %s" % out_dir)
    print("  %d patches, %d x %d grid, %.2f mm cells on %s at %d dpi"
          % (len(rgb), lay["cols"], lay["rows"], lay["cell_mm"], sheet, dpi))
    print("  -> %s" % pdf)
    print("")
    print("  1. Print at 100 percent. Turn OFF 'fit to page'.")
    print("  2. Turn OFF every colour option the driver offers: vivid, auto")
    print("     correct, ICC and printer colour management. You are measuring")
    print("     the printer, so nothing may sit between the file and the ink.")
    print("  3. Same paper you will print the decal on.")
    print("  4. Scan flat at 300 dpi or more, scanner colour correction OFF.")
    print("  5. physdecal inkchart read --scan <the scan>")
    return out_dir


# ---------------------------------------------------------------------- read

def read(scan_path, chart_dir=None, out_csv=None):
    chart_dir = chart_dir or CHART_DIR
    meta_path = os.path.join(chart_dir, META)
    if not os.path.isfile(meta_path):
        raise SystemExit("No %s. Run  physdecal inkchart make  first." % meta_path)
    with open(meta_path, encoding="utf-8") as f:
        lay = json.load(f)

    bgr = cv2.imread(scan_path)
    if bgr is None:
        raise SystemExit("Cannot read %s" % scan_path)

    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, N.FIDUCIAL_DICT))
    det = cv2.aruco.ArucoDetector(adict, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(bgr)
    found = ({} if ids is None else
             {int(i): c.reshape(4, 2).mean(axis=0)
              for i, c in zip(ids.flatten(), corners)})
    if len(found) < 4:
        raise SystemExit(
            "Found %d of 4 markers in %s. All four are needed to rectify a "
            "chart: patches are sampled BY POSITION rather than by finding "
            "squares in the image, so three markers is not enough. Rescan "
            "with the whole sheet flat on the glass and nothing cropped."
            % (len(found), scan_path))

    # rectify the scan into chart millimetres
    cw, ch = lay["content_mm"]
    ppm = WARP_PX_PER_MM
    src = np.array([found[i] for i in range(4)], np.float32)
    dst = np.array([lay["marker_centres_mm"][str(i)] for i in range(4)],
                   np.float32) * ppm
    H = cv2.getPerspectiveTransform(src, dst)
    flat = cv2.warpPerspective(bgr, H, (int(cw * ppm), int(ch * ppm)))
    cv2.imwrite(os.path.join(chart_dir, "inkchart_rectified.png"), flat)

    req = np.asarray(lay["colours"], np.float32)
    half = (lay["cell_mm"] - GUTTER_MM) * SAMPLE_FRAC / 2.0
    meas = np.zeros_like(req)
    for k in range(lay["n"]):
        cx, cy = cell_centre_mm(lay, k)
        x0, y0 = int((cx - half) * ppm), int((cy - half) * ppm)
        x1, y1 = int((cx + half) * ppm), int((cy + half) * ppm)
        win = flat[y0:y1, x0:x1].reshape(-1, 3)
        # median, not mean: a speck of dust or a scanner streak should not move
        # an ink, and half the patch would have to be dirty to shift a median
        med = np.median(win, axis=0)
        meas[k] = (med[2] / 255.0, med[1] / 255.0, med[0] / 255.0)
    meas = np.clip(meas, 0, 1)

    out_csv = out_csv or N.PRINTABLE_COLOURS_CSV
    if os.path.abspath(out_csv) == os.path.abspath(N.PRINTABLE_COLOURS_CSV):
        keep = os.path.join(os.path.dirname(out_csv), "printable_requested.csv")
        if not os.path.isfile(keep):
            os.replace(out_csv, keep)
            print("  kept the colours that were asked for as %s"
                  % os.path.basename(keep))
    np.savetxt(out_csv, meas, delimiter=",", fmt="%.4f")

    _drift(req, meas, chart_dir)
    print("")
    print("  -> %s  (%d measured colours)" % (out_csv, len(meas)))
    print("     rectified scan: %s"
          % os.path.join(chart_dir, "inkchart_rectified.png"))
    print("     side by side:   %s"
          % os.path.join(chart_dir, "inkchart_check.png"))
    print("")
    print("  Set INK_MEASURED = True in config.py, then run physdecal check.")
    return out_csv


def _drift(req, meas, chart_dir):
    """
    How far the print moved the palette, and therefore whether a patch
    optimised against the requested set has to be optimised again.
    """
    import torch
    from physdecal.core import decal as D

    to_t = lambda a: torch.from_numpy(a.astype(np.float32)).T.reshape(1, 3, -1, 1)
    de = D.ciede2000_rgb(to_t(req), to_t(meas)).flatten().numpy()
    print("")
    print("  requested vs measured, CIEDE2000:")
    print("    median %.1f   p95 %.1f   max %.1f"
          % (np.median(de), np.percentile(de, 95), de.max()))
    if np.median(de) < 5.0:
        print("    -> the print tracks the requested set closely. Report the")
        print("       drift and keep the patches you have already optimised.")
    else:
        print("    -> the printer does NOT reproduce the requested set. A patch")
        print("       optimised against it is made of colours this printer")
        print("       misses by this much, so re-optimise, then re-export every")
        print("       decal already exported from the old patch.")

    n = len(req)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / float(cols)))
    s = 28
    img = np.full((rows * s, cols * s, 3), 255, np.uint8)
    for k in range(n):
        r, c = divmod(k, cols)
        y, x = r * s, c * s
        for arr, xa, xb in ((req, 0, s // 2), (meas, s // 2, s)):
            img[y:y + s - 2, x + xa:x + xb - 1] = (arr[k][2] * 255,
                                                   arr[k][1] * 255,
                                                   arr[k][0] * 255)
    cv2.imwrite(os.path.join(chart_dir, "inkchart_check.png"), img)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make", help="render the chart to print")
    p.add_argument("--colours", default=None, help="CSV to chart")
    p.add_argument("--sheet", default=None, choices=list(N.SHEET_MM))
    p.add_argument("--dpi", type=int, default=None)
    p.add_argument("--dir", default=None)

    p = sub.add_parser("read", help="measure a scan of the printed chart")
    p.add_argument("--scan", required=True)
    p.add_argument("--dir", default=None)
    p.add_argument("--out", default=None)

    a = ap.parse_args()
    if a.cmd == "make":
        make(a.colours, a.sheet, a.dpi, a.dir)
    else:
        read(a.scan, a.dir, a.out)


if __name__ == "__main__":
    main()
