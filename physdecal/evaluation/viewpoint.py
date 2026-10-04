r"""
physdecal viewpoint
RQ3, the transfer envelope: how the attack varies with nadir angle, altitude
and lighting, reported as a surface over the grid rather than a single number.

    physdecal viewpoint --patch L_dof --victims all
    physdecal viewpoint --patch L_dof --victims segformer_b0 --arm seg

Produces, per victim:
    envelope_angle_<arm>_<patch>__<victim>.png    ASR vs nadir angle
    envelope_alt_<arm>_<patch>__<victim>.png      ASR vs altitude
    surface_<arm>_<patch>__<victim>.png           angle x altitude heatmap
    lighting_<arm>_<patch>__<victim>.png          the three lighting labels
and across victims:
    envelope_all_<arm>_<patch>.png                every victim on one axis
    viewpoint_<patch>.txt / .tex                  the binned tables

THREE THINGS THIS DOES THAT A PLAIN BINNED MEAN DOES NOT
---------------------------------------------------------
**The control is drawn with the attack.** Every panel plots the unoptimised
anchor decal as a dashed line on the same axes. Without it a rising curve
cannot be told apart from "a square on a roof gets more disruptive at this
angle", which is a property of occlusion geometry rather than of the attack.

**Intervals are over VEHICLES, not instances.** The holdout is three vehicles
and a hundred views of one parked truck are not a hundred measurements. The
bands are cluster bootstraps over vehicles, so they are wide, and that width
is a real property of the experiment.

**Empty cells stay empty.** A bin with no attackable instance is drawn as a
gap and printed as a dash, never as zero. Zero means the attack was tried and
failed; a gap means the victim could not see the vehicle there in the first
place, and conflating them is how a transfer envelope acquires a cliff that
is really a hole in the data.

THE LIGHTING CAVEAT IS PRINTED, NOT ASSUMED
--------------------------------------------
The three lighting labels are only three illumination conditions if
`check_time_of_day.py` says they are. Until its verdict is recorded, the
lighting panel is labelled as three labels on a possibly-duplicated condition
and must not be presented as illumination robustness.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physdecal import config as N
from physdecal.evaluation import stats as ST


ATT = "#2b6cb0"
CTL = "#a0aec0"


def load(patch, arm, victim, kind=None):
    tag = "%s__%s" % (patch, kind) if kind else patch
    p = os.path.join(N.RESULT_DIR, "%s_%s__%s.csv" % (arm, tag, victim))
    if not os.path.isfile(p):
        return None
    with open(p, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["attackable"] == "1"]


def victims_for(patch, arm):
    out, pre = [], "%s_%s__" % (arm, patch)
    import glob
    for p in sorted(glob.glob(os.path.join(N.RESULT_DIR, "%s*.csv" % pre))):
        k = os.path.basename(p)[len(pre):-4]
        if k.split("__")[0] in ("anchor", "grey", "random"):
            continue
        out.append(k)
    return out


def binned(rows, field, bins):
    """[(label, centre, n, asr, lo, hi)] with empty bins preserved as None."""
    out = []
    for lo_b, hi_b in bins:
        sel = [r for r in rows if lo_b <= float(r[field]) < hi_b]
        lab = "%g-%g" % (lo_b, hi_b)
        if not sel:
            out.append((lab, 0.5 * (lo_b + hi_b), 0, None, None, None))
            continue
        a = float(np.mean([int(r["success"]) for r in sel]))
        lo, hi = ST.bootstrap_vehicles(sel, n=2000)
        out.append((lab, 0.5 * (lo_b + hi_b), len(sel), a, lo, hi))
    return out


def _curve(ax, pts, colour, label, band=True):
    x = [p[1] for p in pts if p[3] is not None]
    y = [p[3] for p in pts if p[3] is not None]
    if not x:
        return
    ax.plot(x, y, "o-", color=colour, label=label, zorder=3)
    if band:
        lo = [p[4] for p in pts if p[3] is not None]
        hi = [p[5] for p in pts if p[3] is not None]
        ax.fill_between(x, lo, hi, color=colour, alpha=0.15, lw=0, zorder=1)


def _verdict():
    """DISTINCT, DUPLICATE or UNVERIFIED, from physdecal lighting's record."""
    p = os.path.join(N.FIGURE_DIR, "lighting_verdict.txt")
    if not os.path.isfile(p):
        return "UNVERIFIED"
    with open(p, encoding="utf-8") as f:
        head = f.readline()
    return "DISTINCT" if "DISTINCT" in head else "DUPLICATE"


def _lighting_note():
    """
    The footnote under the lighting panel, driven by the measurement rather
    than by an assumption in either direction.

    While unverified it is a warning, because three labels are not three
    conditions until something has checked. Once checked it becomes the
    measured separation, so the figure carries its own evidence instead of a
    claim the reader has to take on trust.
    """
    v = _verdict()
    if v == "UNVERIFIED":
        return ("CAVEAT: three LABELS, not three verified illumination "
                "conditions.\nRun physdecal lighting before calling this "
                "illumination robustness.")
    if v == "DUPLICATE":
        return ("CAVEAT: physdecal lighting found near-duplicate labels. This "
                "axis is NOT illumination robustness.")
    seps = []
    with open(os.path.join(N.FIGURE_DIR, "lighting_verdict.txt"),
              encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("mean abs diff"):
                seps.append(float(ln.split()[-3]))
    rng = ("%.0f-%.0f" % (min(seps), max(seps))) if seps else "?"
    return ("Verified distinct by physdecal lighting: the labels differ by %s grey "
            "levels over matched\nviewpoints, so this axis is genuine "
            "illumination variation, not three names for one condition." % rng)


def one_axis(patch, arm, victim, field, bins, xlabel, fname, mark_nadir=False):
    rows = load(patch, arm, victim)
    ctl = load(patch, arm, victim, "anchor")
    if not rows:
        return None
    pa = binned(rows, field, bins)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _curve(ax, pa, ATT, "optimised decal")
    if ctl:
        pc = binned(ctl, field, bins)
        _curve(ax, pc, CTL, "unoptimised anchor (control)", band=False)

    for lab, c, n, a, lo, hi in pa:
        if a is None:
            continue
        ax.annotate("n=%d" % n, (c, a), textcoords="offset points",
                    xytext=(0, 9), fontsize=6.5, ha="center", color="#4a5568")
    gaps = [p for p in pa if p[3] is None]
    if gaps:
        ax.text(0.99, 0.02,
                "%d bin(s) empty: no attackable instance, not zero ASR"
                % len(gaps), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="#c53030")
    if mark_nadir:
        ax.axvspan(0, N.NADIR_MAX_THETA_DEG, color=ATT, alpha=0.10, zorder=0)
        ax.text(N.NADIR_MAX_THETA_DEG / 2, ax.get_ylim()[1] * 0.97,
                "optimised\nhere", ha="center", va="top", fontsize=7.5,
                color=ATT)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("attack success rate")
    ax.set_title("%s on %s" % (patch, victim), fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, fname)


def surface(patch, arm, victim):
    """
    The angle x altitude grid, which is what the proposal asks for and what a
    pair of 1-D curves cannot show: whether the attack decays with angle
    independently of altitude, or only at one end of the altitude range.
    """
    rows = load(patch, arm, victim)
    if not rows:
        return None
    A, L = N.REPORT_ANGLE_BINS, N.REPORT_ALT_BINS
    grid = np.full((len(L), len(A)), np.nan)
    cnt = np.zeros((len(L), len(A)), int)
    for i, (lo_a, hi_a) in enumerate(L):
        for j, (lo_t, hi_t) in enumerate(A):
            sel = [r for r in rows
                   if lo_t <= float(r["theta_deg"]) < hi_t
                   and lo_a <= float(r["altitude_m"]) < hi_a]
            cnt[i, j] = len(sel)
            if sel:
                grid[i, j] = np.mean([int(r["success"]) for r in sel])

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#e2e8f0")
    vmax = max(0.05, float(np.nanmax(grid)) if np.isfinite(grid).any() else 0.05)
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=vmax,
                   aspect="auto", origin="lower")
    for i in range(len(L)):
        for j in range(len(A)):
            if cnt[i, j] == 0:
                ax.text(j, i, "--", ha="center", va="center", fontsize=8,
                        color="#718096")
            else:
                v = grid[i, j]
                # magma runs DARK at low values and LIGHT at high ones, so the
                # label colour has to invert against the value: white on the
                # dark cells, black on the bright ones. The other way round
                # makes exactly the strongest cells unreadable.
                dark_bg = v < 0.55 * vmax
                ax.text(j, i, "%.2f\nn=%d" % (v, cnt[i, j]), ha="center",
                        va="center", fontsize=6.5,
                        color="white" if dark_bg else "black")
    ax.set_xticks(range(len(A)))
    ax.set_xticklabels(["%g-%g" % b for b in A], fontsize=8)
    ax.set_yticks(range(len(L)))
    ax.set_yticklabels(["%g-%g" % b for b in L], fontsize=8)
    ax.set_xlabel("nadir angle (degrees)")
    ax.set_ylabel("altitude (m)")
    ax.set_title("Transfer envelope: %s on %s" % (patch, victim), fontsize=10)
    ax.text(0.5, -0.32, "'--' = no attackable instance in that cell, which is "
            "undefined and not zero.   n is printed per cell; below about "
            "n=5 a cell is noise.", transform=ax.transAxes, ha="center",
            fontsize=6.8, color="#4a5568")
    fig.colorbar(im, ax=ax, label="ASR", fraction=0.03)
    fig.tight_layout()
    return _save(fig, "surface_%s_%s__%s.png" % (arm, patch, victim))


def lighting(patch, arm, victim):
    rows = load(patch, arm, victim)
    ctl = load(patch, arm, victim, "anchor")
    if not rows:
        return None
    labs = sorted({r["lighting"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    x = np.arange(len(labs))
    for off, src, colour, name in ((-0.18, rows, ATT, "optimised decal"),
                                   (0.18, ctl, CTL, "anchor control")):
        if not src:
            continue
        vals, errs, ns = [], [[], []], []
        for lb in labs:
            sel = [r for r in src if r["lighting"] == lb]
            if not sel:
                vals.append(0); errs[0].append(0); errs[1].append(0); ns.append(0)
                continue
            a = float(np.mean([int(r["success"]) for r in sel]))
            lo, hi = ST.bootstrap_vehicles(sel, n=2000)
            vals.append(a); errs[0].append(max(0, a - lo))
            errs[1].append(max(0, hi - a)); ns.append(len(sel))
        ax.bar(x + off, vals, width=0.34, color=colour, label=name,
               yerr=np.array(errs), capsize=3, error_kw=dict(lw=1))
        if colour == ATT:
            for xi, v, nn in zip(x, vals, ns):
                ax.annotate("n=%d" % nn, (xi + off, v),
                            textcoords="offset points", xytext=(0, 3),
                            fontsize=6.5, ha="center")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("attack success rate")
    ax.set_title("%s on %s" % (patch, victim), fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, frameon=False)
    fig.text(0.5, 0.005, _lighting_note(), ha="center", fontsize=6.8,
             color="#4a5568" if _verdict() == "DISTINCT" else "#c53030")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    return _save(fig, "lighting_%s_%s__%s.png" % (arm, patch, victim))


def all_victims(patch, arm, victims):
    """Every victim's angle envelope on one axis: the paper's RQ3 figure."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cmap = plt.get_cmap("tab10")
    drawn = 0
    for i, v in enumerate(victims):
        rows = load(patch, arm, v)
        if not rows:
            continue
        pts = binned(rows, "theta_deg", N.REPORT_ANGLE_BINS)
        x = [p[1] for p in pts if p[3] is not None]
        y = [p[3] for p in pts if p[3] is not None]
        if not x:
            continue
        ax.plot(x, y, "o-", color=cmap(i % 10), label=v, lw=1.6, ms=4)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.axvspan(0, N.NADIR_MAX_THETA_DEG, color=ATT, alpha=0.10, zorder=0)
    ax.text(N.NADIR_MAX_THETA_DEG / 2, ax.get_ylim()[1] * 0.97,
            "optimised\nhere", ha="center", va="top", fontsize=7.5, color=ATT)
    ax.set_xlabel("nadir angle (degrees)")
    ax.set_ylabel("attack success rate")
    ax.set_title("Transfer envelope across the %s roster: %s"
                 % ("segmentation" if arm == "seg" else "detection", patch),
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, "envelope_all_%s_%s.png" % (arm, patch))


def tables(patch, victims_by_arm):
    lines, tex = [], []
    for arm, vs in victims_by_arm.items():
        for v in vs:
            rows = load(patch, arm, v)
            if not rows:
                continue
            for field, bins, name in (("theta_deg", N.REPORT_ANGLE_BINS,
                                       "nadir angle (deg)"),
                                      ("altitude_m", N.REPORT_ALT_BINS,
                                       "altitude (m)")):
                lines += ["", "%s / %s -- %s" % (arm, v, name),
                          "%-12s %7s %8s %-20s" % (name.split()[0], "n", "ASR",
                                                   "95% CI (vehicles)"),
                          "-" * 52]
                for lab, c, n, a, lo, hi in binned(rows, field, bins):
                    if a is None:
                        lines.append("%-12s %7d %8s %-20s"
                                     % (lab, 0, "--", "no attackable instance"))
                        continue
                    lines.append("%-12s %7d %8.3f [%.3f, %.3f]"
                                 % (lab, n, a, lo, hi))
                    tex.append(r"%s & %s & %s & %d & %.3f & [%.3f, %.3f] \\"
                               % (arm, v.replace("_", r"\_"), lab, n, a, lo, hi))
    head = ("Transfer envelope for patch '%s'\n"
            "Intervals are cluster bootstraps over VEHICLES (n=3 holdout "
            "vehicles), not over\ninstances, so they are wide by construction. "
            "'--' means the bin held no instance the\nvictim could segment or "
            "detect without the patch: undefined, not zero.\n" % patch)
    p1 = os.path.join(N.FIGURE_DIR, "viewpoint_%s.txt" % patch)
    with open(p1, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(lines) + "\n")
    p2 = os.path.join(N.FIGURE_DIR, "viewpoint_%s.tex" % patch)
    with open(p2, "w", encoding="utf-8") as f:
        f.write("\n".join([r"\begin{tabular}{lllrrl}", r"\toprule",
                           r"Arm & Victim & Bin & $n$ & ASR & 95\% CI \\",
                           r"\midrule"] + tex +
                          [r"\bottomrule", r"\end{tabular}"]) + "\n")
    print(head + "\n".join(lines))
    return p1, p2


def _save(fig, name):
    os.makedirs(N.FIGURE_DIR, exist_ok=True)
    p = os.path.join(N.FIGURE_DIR, name)
    fig.savefig(p, dpi=N.PANEL_DPI, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--victims", nargs="*", default=["all"])
    ap.add_argument("--arm", default="both", choices=["seg", "det", "both"])
    a = ap.parse_args()

    arms = ["seg", "det"] if a.arm == "both" else [a.arm]
    by_arm, made = {}, []
    for arm in arms:
        vs = (victims_for(a.patch, arm) if a.victims == ["all"] else a.victims)
        vs = [v for v in vs if load(a.patch, arm, v)]
        by_arm[arm] = vs
        for v in vs:
            for fn, args in (
                    (one_axis, (a.patch, arm, v, "theta_deg",
                                N.REPORT_ANGLE_BINS, "nadir angle (degrees)",
                                "envelope_angle_%s_%s__%s.png" % (arm, a.patch, v),
                                True)),
                    (one_axis, (a.patch, arm, v, "altitude_m",
                                N.REPORT_ALT_BINS, "altitude (m)",
                                "envelope_alt_%s_%s__%s.png" % (arm, a.patch, v),
                                False)),
                    (surface, (a.patch, arm, v)),
                    (lighting, (a.patch, arm, v))):
                p = fn(*args)
                if p:
                    made.append(p)
        p = all_victims(a.patch, arm, vs)
        if p:
            made.append(p)

    print("\n%d viewpoint figures written" % len(made))
    for p in made:
        print("   %s" % os.path.basename(p))
    tables(a.patch, by_arm)


if __name__ == "__main__":
    main()
