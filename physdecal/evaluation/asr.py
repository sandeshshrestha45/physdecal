r"""
physdecal asr
One-line ASR summary for a patch, split by nadir angle.

    physdecal asr full_r0 L_steps L_size

Exists because the headline ASR over the whole holdout grid mixes the
viewpoints the patch was optimised at with the ones it was not, and a lever
that helps at nadir can be invisible in the mixed number. Every diagnostic run
is read here, so the comparison between levers is always like for like.
"""

import argparse
import csv
import glob
import os

import numpy as np

from physdecal import config as N


def rows_for(tag, arm, victim):
    p = os.path.join(N.RESULT_DIR, "%s_%s__%s.csv" % (arm, tag, victim))
    if not os.path.isfile(p):
        return None
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def block(rows, lo, hi):
    sel = [r for r in rows
           if lo <= float(r["theta_deg"]) < hi and r["attackable"] == "1"]
    if not sel:
        return None
    return {
        "n": len(sel),
        "asr": float(np.mean([int(r["success"]) for r in sel])),
        "drop": float(np.mean([float(r.get("iou_drop") or r.get("score_drop"))
                               for r in sel])),
    }


def report(tags, victims=(("seg", "segformer_b0"), ("det", "retinanet"))):
    print("%-14s %-6s %-14s %6s %7s %8s   %6s %7s %8s"
          % ("tag", "arm", "victim", "n", "ASR", "drop", "n", "ASR", "drop"))
    print("%-14s %-6s %-14s %-23s %s"
          % ("", "", "", "  ---- nadir, 0-10 deg ----", "  ---- all viewpoints ----"))
    print("-" * 92)
    for tag in tags:
        for arm, victim in victims:
            rows = rows_for(tag, arm, victim)
            if rows is None:
                print("%-14s %-6s %-14s  (not evaluated)" % (tag, arm, victim))
                continue
            nad = block(rows, 0, 10)
            allv = block(rows, 0, 999)
            fmt = lambda d: ("%6d %7.3f %8.3f" % (d["n"], d["asr"], d["drop"])
                             if d else "%6s %7s %8s" % ("-", "-", "-"))
            print("%-14s %-6s %-14s %s   %s"
                  % (tag, arm, victim, fmt(nad), fmt(allv)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="*",
                    help="patch tags; default every tag with seg results")
    tags = ap.parse_args().tags
    if not tags:
        tags = sorted({os.path.basename(p).split("__")[0][4:]
                       for p in glob.glob(os.path.join(N.RESULT_DIR, "seg_*.csv"))})
    report(tags)


if __name__ == "__main__":
    main()
