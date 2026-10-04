"""Fiducial-free altitude recovery, recomputed from the archived frames.

Z = f_px * L_model / L_px, with L_px the model's apparent body length taken as
the long side of the minimum-area rectangle around the vehicle mask, and f_px
from the EXIF 35 mm-equivalent focal length. The effective altitude is then
Z scaled by the model-to-real ratio the print geometry records.
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image, ExifTags

from physdecal import config as N


def focal_px(path):
    im = Image.open(path)
    w, h = im.size
    ex = im.getexif()
    sub = ex.get_ifd(0x8769) or {}
    tag = {ExifTags.TAGS.get(k, k): v for k, v in list(ex.items()) + list(sub.items())}
    f35 = tag.get("FocalLengthIn35mmFilm")
    if not f35:
        return None, (w, h), None
    return float(f35) / 36.0 * max(w, h), (w, h), float(f35)


def body_px(mask_path):
    m = np.asarray(Image.open(mask_path))
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return max(w, h)


def main():
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(N.PHYSICAL_DIR, "*"))):
        pj = os.path.join(d, "print.json")
        if not os.path.isfile(pj):
            continue
        g = json.load(open(pj, encoding="utf-8"))["geometry"]
        ratio = g["vehicle_len_m"] / (g["model_vehicle_len_mm"] / 1000.0)
        L_model = g["model_vehicle_len_mm"] / 1000.0
        for mp in sorted(glob.glob(os.path.join(d, "annotations", "*_mask.png"))):
            scene = os.path.basename(mp)[:-len("_mask.png")]
            img = os.path.join(d, "%s__clean__00.jpg" % scene)
            if not os.path.isfile(img):
                continue
            f, (w, h), f35 = focal_px(img)
            Lpx = body_px(mp)
            if not f or not Lpx:
                print("   skip %s (f=%s Lpx=%s)" % (scene, f, Lpx))
                continue
            Z = f * L_model / Lpx
            rows.append({
                "vehicle": os.path.basename(d).split("__")[-1],
                "scene": scene, "f35": f35, "f_px": f, "L_px": Lpx,
                "Z_m": Z, "eff_alt_m": Z * ratio, "ratio": ratio,
            })

    by = {}
    for r in rows:
        by.setdefault(r["vehicle"], []).append(r)

    print("%-12s %-28s %8s %9s %10s" % ("vehicle", "scene", "L_px", "Z (m)", "alt (m)"))
    for v in sorted(by):
        for r in sorted(by[v], key=lambda x: x["scene"]):
            print("%-12s %-28s %8.1f %9.3f %10.2f"
                  % (v, r["scene"], r["L_px"], r["Z_m"], r["eff_alt_m"]))
        a = [r["eff_alt_m"] for r in by[v]]
        z = [r["Z_m"] for r in by[v]]
        print("%-12s %-28s  ratio %.3f   Z %.2f-%.2f m   alt %.1f-%.1f m\n"
              % (v, "RANGE", by[v][0]["ratio"], min(z), max(z), min(a), max(a)))

    alls = [r["eff_alt_m"] for r in rows]
    print("ALL FRAMES: n=%d  alt %.1f - %.1f m" % (len(alls), min(alls), max(alls)))
    inb = [a for a in alls if 19.0 <= a <= 63.0]
    print("in 19-63 m band: %d of %d" % (len(inb), len(alls)))
    with open(os.path.join(N.PHYSICAL_DIR, "recovered_altitude.csv"), "w", newline="",
              encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print("wrote %s" % os.path.join(N.PHYSICAL_DIR, "recovered_altitude.csv"))


if __name__ == "__main__":
    main()
