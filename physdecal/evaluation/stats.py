r"""
physdecal stats
Confidence intervals and leave-one-vehicle-out, for every ASR the paper
reports.

    physdecal stats --patch roof_r4 --arm seg
    physdecal stats --compare L_size roof_r4 --arm seg
    physdecal stats --coverage L_size L_dof roof_r4 full_r0

WHY THIS FILE IS NOT OPTIONAL
-----------------------------
The holdout is THREE vehicles. Every ASR in this study is an average over
instances that are not independent: a hundred rows of the same parked pickup
from adjacent azimuths are close to a hundred copies of one measurement. A
naive binomial interval over instances would therefore be far too narrow, and
a reviewer who notices will be right to.

Two estimators are reported for that reason and they answer different
questions:

  instance bootstrap   resamples instances. Answers "how precise is this
                       number on this fleet". Optimistic about generalisation
                       and labelled as such.

  vehicle bootstrap    resamples VEHICLES with replacement, then takes all
                       their instances. Answers "would this survive a
                       different fleet". This is the honest interval and it is
                       the one that goes in the abstract. With three holdout
                       vehicles it will be wide, and that width is a real
                       property of the experiment, not a defect of the
                       estimator.

  leave-one-vehicle-out  refits the number with each vehicle removed. Shows
                       whether a single vehicle is carrying the result, which
                       with n=3 is a live possibility.

Nothing here recomputes a metric. Everything is read back from the
per-instance CSVs physdecal evaluate wrote.
"""

import argparse
import csv
import glob
import os

import numpy as np

from physdecal import config as N


def load(tag, arm, victim):
    p = os.path.join(N.RESULT_DIR, "%s_%s__%s.csv" % (arm, tag, victim))
    if not os.path.isfile(p):
        return None
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def attackable(rows, theta_max=None):
    out = [r for r in rows if r.get("attackable") == "1"]
    if theta_max is not None:
        out = [r for r in out if float(r["theta_deg"]) < theta_max]
    return out


def asr(rows):
    return float(np.mean([int(r["success"]) for r in rows])) if rows else float("nan")


def bootstrap_instances(rows, n=5000, seed=None):
    if not rows:
        return float("nan"), float("nan")
    rng = np.random.RandomState(N.SEED if seed is None else seed)
    y = np.array([int(r["success"]) for r in rows], float)
    idx = rng.randint(0, len(y), size=(n, len(y)))
    dist = y[idx].mean(axis=1)
    return float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


def bootstrap_vehicles(rows, n=5000, seed=None):
    """
    Cluster bootstrap: resample vehicles, keep all of each one's instances.

    This is the interval that answers the question a reviewer actually asks,
    which is whether the result is about the attack or about these three
    vehicles.
    """
    if not rows:
        return float("nan"), float("nan")
    rng = np.random.RandomState(N.SEED if seed is None else seed)
    by = {}
    for r in rows:
        by.setdefault(r["target"], []).append(int(r["success"]))
    keys = sorted(by)
    arrs = [np.array(by[k], float) for k in keys]
    dist = []
    for _ in range(n):
        pick = rng.randint(0, len(arrs), size=len(arrs))
        vals = np.concatenate([arrs[i] for i in pick])
        dist.append(vals.mean())
    return float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


def leave_one_vehicle_out(rows):
    by = {}
    for r in rows:
        by.setdefault(r["target"], []).append(int(r["success"]))
    keys = sorted(by)
    out = []
    for k in keys:
        rest = np.concatenate([np.array(by[j], float) for j in keys if j != k]) \
            if len(keys) > 1 else np.array([])
        out.append((k, len(by[k]), float(np.mean(by[k])),
                    float(rest.mean()) if len(rest) else float("nan")))
    return out


def summarise(tag, arm, victim, theta_max=None, label=""):
    rows = load(tag, arm, victim)
    if rows is None:
        print("  %-22s (not evaluated)" % tag)
        return None
    att = attackable(rows, theta_max)
    if not att:
        print("  %-22s attackable set EMPTY -- ASR undefined" % tag)
        return None
    point = asr(att)
    ilo, ihi = bootstrap_instances(att)
    vlo, vhi = bootstrap_vehicles(att)
    print("  %-22s n=%-5d ASR %.3f   instance CI [%.3f, %.3f]   "
          "vehicle CI [%.3f, %.3f]%s"
          % (tag, len(att), point, ilo, ihi, vlo, vhi,
             ("  " + label) if label else ""))
    return {"tag": tag, "n": len(att), "asr": point,
            "inst_ci": (ilo, ihi), "veh_ci": (vlo, vhi), "rows": att}


def compare(tag_a, tag_b, arm, victim, theta_max=None, n=5000):
    """
    Paired difference between two patches on the SAME instances.

    Paired, because both patches were scored on the identical holdout rows and
    an unpaired interval would throw away that pairing and be needlessly wide.
    The resampling unit is still the vehicle.
    """
    ra, rb = load(tag_a, arm, victim), load(tag_b, arm, victim)
    if ra is None or rb is None:
        print("  cannot compare: missing results")
        return
    key = lambda r: (r["frame_id"], r["target"])
    da = {key(r): r for r in ra if r.get("attackable") == "1"}
    db = {key(r): r for r in rb if r.get("attackable") == "1"}
    common = sorted(set(da) & set(db))
    if theta_max is not None:
        common = [k for k in common if float(da[k]["theta_deg"]) < theta_max]
    if not common:
        print("  no common attackable instances")
        return

    by = {}
    for k in common:
        by.setdefault(k[1], []).append(int(da[k]["success"]) - int(db[k]["success"]))
    keys = sorted(by)
    arrs = [np.array(by[k], float) for k in keys]
    point = float(np.concatenate(arrs).mean())

    rng = np.random.RandomState(N.SEED)
    dist = []
    for _ in range(n):
        pick = rng.randint(0, len(arrs), size=len(arrs))
        dist.append(np.concatenate([arrs[i] for i in pick]).mean())
    lo, hi = np.percentile(dist, 2.5), np.percentile(dist, 97.5)
    sig = "" if (lo <= 0 <= hi) else "  SIGNIFICANT at 95%"
    print("  %s minus %s: %+.3f  vehicle CI [%+.3f, %+.3f]  n=%d%s"
          % (tag_a, tag_b, point, lo, hi, len(common), sig))
    if lo <= 0 <= hi:
        print("       interval spans zero: these two are not distinguishable "
              "on three vehicles.")


def coverage_table(tags, arm, victim, theta_max=None):
    """
    ASR against coverage fraction. This is the RQ2 collapse: if coverage is
    the governing variable, points from different vehicle classes and
    different patch sizes should lie on one curve.
    """
    print("\n%-22s %-22s %8s %8s %7s %7s"
          % ("tag", "vehicle", "decal m", "coverage", "n", "ASR"))
    print("-" * 82)
    pts = []
    for tag in tags:
        rows = load(tag, arm, victim)
        if rows is None:
            continue
        att = attackable(rows, theta_max)
        by = {}
        for r in att:
            if "coverage_frac" not in r:
                continue
            k = (r["target"], round(float(r["patch_size_m"]), 2),
                 round(float(r["coverage_frac"]), 3), r.get("fits_roof", "?"))
            by.setdefault(k, []).append(int(r["success"]))
        if not by:
            print("  %-20s (no coverage columns: re-evaluate to add them)" % tag)
            continue
        for k in sorted(by):
            a = float(np.mean(by[k]))
            flag = "" if k[3] == "1" else "   OVERHANGS"
            print("%-22s %-22s %8.2f %7.1f%% %7d %7.3f%s"
                  % (tag, k[0], k[1], 100 * k[2], len(by[k]), a, flag))
            pts.append((k[2], a, len(by[k]), k[3] == "1", tag, k[0]))
    return pts


def plot_coverage(pts, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(pts) < 3:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for fits, colour, marker, lab in ((True, "#2b6cb0", "o", "fits the roof"),
                                      (False, "#c53030", "x",
                                       "overhangs: not deployable")):
        xs = [100 * p[0] for p in pts if p[3] == fits]
        ys = [p[1] for p in pts if p[3] == fits]
        if xs:
            ax.scatter(xs, ys, c=colour, marker=marker, s=46, label=lab,
                       zorder=3)
    ax.axvspan(0, 21, color="#2b6cb0", alpha=0.08, zorder=0)
    ax.text(10.5, ax.get_ylim()[1] * 0.94, "deployable\n(single panel)",
            ha="center", va="top", fontsize=8, color="#2b6cb0")
    ax.set_xlabel("coverage: decal area as a percentage of vehicle footprint")
    ax.set_ylabel("attack success rate")
    ax.set_title("Attack strength is governed by coverage, not patch size")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = out or os.path.join(N.FIGURE_DIR, "coverage_curve.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=N.PANEL_DPI)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch", nargs="*", default=None)
    ap.add_argument("--compare", nargs=2, default=None,
                    metavar=("A", "B"))
    ap.add_argument("--coverage", nargs="*", default=None,
                    help="tags to place on the coverage curve")
    ap.add_argument("--arm", default="seg", choices=["seg", "det"])
    ap.add_argument("--victim", default=None)
    ap.add_argument("--nadir", action="store_true",
                    help="restrict to theta < 10, the angles optimised at")
    a = ap.parse_args()

    victim = a.victim or (N.ATTACK_MODELS[0] if a.arm == "seg"
                          else N.DET_ATTACK_MODELS[0])
    tmax = 10.0 if a.nadir else None
    scope = "nadir only" if a.nadir else "all viewpoints"
    print("arm=%s victim=%s  (%s)\n" % (a.arm, victim, scope))

    if a.patch:
        print("ASR with confidence intervals")
        print("  instance CI answers precision on this fleet; vehicle CI "
              "answers generalisation")
        for t in a.patch:
            summarise(t, a.arm, victim, tmax)
        print("\nleave-one-vehicle-out")
        for t in a.patch:
            rows = load(t, a.arm, victim)
            if rows is None:
                continue
            att = attackable(rows, tmax)
            if not att:
                continue
            print("  %s" % t)
            for veh, n, own, rest in leave_one_vehicle_out(att):
                print("     without %-22s rest ASR %.3f   (that vehicle "
                      "alone %.3f, n=%d)" % (veh, rest, own, n))

    if a.compare:
        print("\npaired difference")
        compare(a.compare[0], a.compare[1], a.arm, victim, tmax)

    if a.coverage:
        pts = coverage_table(a.coverage, a.arm, victim, tmax)
        p = plot_coverage(pts)
        if p:
            print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
