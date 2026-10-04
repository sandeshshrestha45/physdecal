r"""
physdecal report
Collects every result CSV into the tables and plots the paper needs.

    physdecal report
    physdecal report --patch main
    physdecal report --tables-only

Nothing here computes a metric. Everything is read back from the per-instance
CSVs that physdecal evaluate and physdecal physical wrote, so a number in a table can
always be traced to the rows that produced it. If a table looks wrong, the
rows are on disk.

TWO REFUSALS ARE BUILT IN AND NEITHER IS OPTIONAL
-------------------------------------------------
An ASR over an empty attackable set is printed as "undefined", never as 0.000.
A row whose victim could not see the vehicle without the patch is not a row
about the patch.

A printability number is not printed at all while INK_MEASURED is False. The
shipped ink set is a colour cube, a cube is not any printer's gamut, and an NPS
computed against it is a number about a cube.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from physdecal import config as N


UNDEF = "undefined"


# ------------------------------------------------------------------- reading

def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(r, k, default=float("nan")):
    v = r.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def result_files(arm, patch=None):
    pat = os.path.join(N.RESULT_DIR, "%s_*.csv" % arm)
    out = []
    for p in sorted(glob.glob(pat)):
        base = os.path.basename(p)[len(arm) + 1:-4]
        if "__" not in base:
            continue
        tag, victim = base.rsplit("__", 1)
        if patch and not tag.startswith(patch):
            continue
        out.append((tag, victim, p))
    return out


def summarise(rows, arm):
    """ASR and its companions, or undefined with the reason."""
    att = [r for r in rows if r.get("attackable") == "1"]
    n = len(rows)
    if not att:
        return {"n": n, "n_attackable": 0, "asr": None,
                "note": "attackable set empty"}
    suc = sum(1 for r in att if r.get("success") == "1")
    d = {"n": n, "n_attackable": len(att), "asr": suc / len(att), "note": ""}
    if arm == "seg":
        d["mean_drop"] = float(np.nanmean([fnum(r, "iou_drop") for r in att]))
        d["clean"] = float(np.nanmean([fnum(r, "clean_iou") for r in att]))
        r50 = [fnum(r, "r50_m") for r in att]
        r90 = [fnum(r, "r90_m") for r in att]
        d["r50"] = float(np.nanmedian(r50)) if np.any(~np.isnan(r50)) else float("nan")
        d["r90"] = float(np.nanmedian(r90)) if np.any(~np.isnan(r90)) else float("nan")
        under = [fnum(r, "flip_under") for r in att]
        away = [fnum(r, "flip_away") for r in att]
        tot = np.nansum(under) + np.nansum(away)
        d["away_frac"] = float(np.nansum(away) / tot) if tot > 0 else float("nan")
    else:
        d["mean_drop"] = float(np.nanmean([fnum(r, "score_drop") for r in att]))
        d["clean"] = float(np.nanmean([fnum(r, "clean_score") for r in att]))
    return d


def asr_str(d, prec=3):
    return UNDEF if d["asr"] is None else ("%.*f" % (prec, d["asr"]))


# -------------------------------------------------------------------- tables

def table_main(patch=None):
    """T1: every patch against every victim, both arms."""
    lines = []
    for arm, label in (("seg", "Segmentation"), ("det", "Detection")):
        files = result_files(arm, patch)
        if not files:
            continue
        lines.append("")
        lines.append("%s  (ASR over the attackable set only)" % label)
        lines.append("%-34s %-20s %7s %9s %7s %9s"
                     % ("patch", "victim", "n", "attack.", "ASR", "drop"))
        lines.append("-" * 92)
        for tag, victim, p in files:
            d = summarise(read_rows(p), arm)
            lines.append("%-34s %-20s %7d %9d %7s %9s"
                         % (tag[:34], victim[:20], d["n"], d["n_attackable"],
                            asr_str(d),
                            "%.3f" % d["mean_drop"] if d["asr"] is not None
                            else "-"))
            if d["note"]:
                lines.append("%-34s %-20s   %s -- ASR is UNDEFINED, not zero"
                             % ("", "", d["note"]))
    return "\n".join(lines)


def table_realism(patch=None):
    """T2: the realism curve. ASR against what the patch gave up to get it."""
    rows = []
    for tag in sorted(os.listdir(N.PATCH_DIR)) if os.path.isdir(N.PATCH_DIR) else []:
        meta_p = os.path.join(N.PATCH_DIR, tag, "run.json")
        if not os.path.isfile(meta_p):
            continue
        if patch and not tag.startswith(patch):
            continue
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
        m = meta.get("realism_metrics", {})

        best = None
        for arm in ("seg", "det"):
            for t, victim, p in result_files(arm, tag):
                if t != tag:
                    continue
                d = summarise(read_rows(p), arm)
                if d["asr"] is not None and (best is None or d["asr"] > best[1]):
                    best = ("%s/%s" % (arm, victim), d["asr"])

        rows.append({
            "tag": tag, "level": meta.get("realism_level"),
            "dof": meta.get("degrees_of_freedom"),
            "inks": m.get("ink_count"), "hf": m.get("hf_ratio"),
            "edge": m.get("edge_iou"), "de": m.get("delta_e_med"),
            "best": best,
        })
    if not rows:
        return ""

    out = ["", "Realism against attack strength",
           "%-30s %6s %8s %6s %6s %7s %8s %-22s"
           % ("patch", "level", "dof", "inks", "hf", "edgeIoU", "ASR", "best victim"),
           "-" * 104]
    for r in sorted(rows, key=lambda x: (x["level"] is None, x["level"])):
        out.append("%-30s %6s %8s %6s %6s %7s %8s %-22s"
                   % (r["tag"][:30], r["level"], r["dof"], r["inks"],
                      "%.3f" % r["hf"] if r["hf"] is not None else "-",
                      "%.2f" % r["edge"] if r["edge"] is not None else "-",
                      "%.3f" % r["best"][1] if r["best"] else UNDEF,
                      r["best"][0] if r["best"] else ""))
    out.append("")
    out.append("dof is how many numbers the attacker controls. An ASR is only")
    out.append("interesting beside the size of the search space that produced it.")
    return "\n".join(out)


def table_binned(patch, arm, victim, field, bins, label):
    """ASR binned on a viewpoint axis. The transfer envelope, as a table."""
    path = os.path.join(N.RESULT_DIR, "%s_%s__%s.csv" % (arm, patch, victim))
    if not os.path.isfile(path):
        return ""
    rows = read_rows(path)
    out = ["", "%s: %s on %s" % (label, patch, victim),
           "%-14s %7s %9s %8s %9s" % (label, "n", "attack.", "ASR", "drop"),
           "-" * 52]
    for lo, hi in bins:
        sel = [r for r in rows if lo <= fnum(r, field) < hi]
        if not sel:
            out.append("%-14s %7d %9s %8s %9s"
                       % ("%g-%g" % (lo, hi), 0, "-", "-", "-"))
            continue
        d = summarise(sel, arm)
        out.append("%-14s %7d %9d %8s %9s"
                   % ("%g-%g" % (lo, hi), d["n"], d["n_attackable"], asr_str(d),
                      "%.3f" % d["mean_drop"] if d["asr"] is not None else "-"))
    return "\n".join(out)


def table_physical():
    """The physical sessions, beside nothing -- the comparison is the reader's
    job and the digital table is directly above it."""
    sessions = sorted(glob.glob(os.path.join(N.PHYSICAL_DIR, "*",
                                             "physical_results.csv")))
    if not sessions:
        return ""
    out = ["", "Physical capture",
           "%-30s %7s %8s %9s %8s %9s"
           % ("session", "pairs", "arm", "attack.", "ASR", "drop"),
           "-" * 78]
    for p in sessions:
        name = os.path.basename(os.path.dirname(p))
        rows = read_rows(p)
        for arm, att_k, suc_k, drop_k in (
                ("seg", "seg_attackable", "seg_success", "iou_drop"),
                ("det", "det_attackable", "det_success", "score_drop")):
            have = [r for r in rows if r.get(att_k) not in (None, "")]
            if not have:
                continue
            att = [r for r in have if r.get(att_k) == "1"]
            if not att:
                out.append("%-30s %7d %8s %9d %8s %9s"
                           % (name[:30], len(have), arm, 0, UNDEF, "-"))
                continue
            asr = sum(1 for r in att if r.get(suc_k) == "1") / len(att)
            drop = float(np.nanmean([fnum(r, drop_k) for r in att]))
            out.append("%-30s %7d %8s %9d %8.3f %9.3f"
                       % (name[:30], len(have), arm, len(att), asr, drop))
    out.append("")
    out.append("Physical n is small by nature. Report it as a paired")
    out.append("measurement with n stated, not as a rate with implied precision.")
    return "\n".join(out)


def table_provenance(patch=None):
    """What produced these numbers. Without it a table is unreproducible."""
    out = ["", "Provenance"]
    tags = []
    if os.path.isdir(N.PATCH_DIR):
        tags = [t for t in sorted(os.listdir(N.PATCH_DIR))
                if (not patch or t.startswith(patch))
                and os.path.isfile(os.path.join(N.PATCH_DIR, t, "run.json"))]
    for t in tags:
        with open(os.path.join(N.PATCH_DIR, t, "run.json"), encoding="utf-8") as f:
            m = json.load(f)
        out.append("  %-28s task=%-6s seg=%-18s det=%-10s level=%s chain=%s"
                   % (t[:28], m.get("task"), m.get("seg_model"),
                      m.get("det_model"), m.get("realism_level"),
                      m.get("print_chain")))
        # size_mode "roof" sizes the decal per vehicle, so patch_size_m is
        # present and null rather than absent: .get's default never fires.
        size = m.get("patch_size_m")
        out.append("  %-28s size=%-16s res=%d  steps=%s  gsd=%s  anchor=%s"
                   % ("", "%.2f m" % size if size is not None
                      else "per-roof (%s)" % m.get("patch_size_mode", "roof"),
                      m.get("patch_res") or 0,
                      m.get("steps"), m.get("band_gsd_m_per_px"),
                      m.get("anchor_design")))

    out.append("")
    if N.INK_MEASURED:
        out.append("  ink set: MEASURED (%s)" % N.PRINTABLE_COLOURS_CSV)
    else:
        out.append("  ink set: PLACEHOLDER. No printability number is reported")
        out.append("           here, and none should be reported in the paper,")
        out.append("           until PRINTABLE_COLOURS_CSV is a scan of the")
        out.append("           actual printer and INK_MEASURED is True.")
    return "\n".join(out)


# --------------------------------------------------------------------- plots

def _ink_label(meta):
    """
    "10 of 11 inks", never a bare "10 inks".

    ink_count is how many colours the patch USED; the palette it could draw
    from is larger at level 4, where the anchor's own spot colours join the
    ink set. Printing the first without the second invites the reader to
    compare it against INK_COUNT and conclude the constraint leaked.
    """
    from physdecal.core import decal as DEC
    used = (meta.get("realism_metrics") or {}).get("ink_count", "?")
    pal = DEC.palette_size(meta)
    return "%s of %s inks" % (used, pal) if pal else "%s colours used" % used


def plot_realism_curve(patch=None, victim=None, arm=None, out=None):
    """
    ASR against realism level, for ONE victim held fixed across every point.

    The victim is not a detail. This curve used to plot max(ASR) over whatever
    victims each run happened to have been evaluated against, and coverage is
    not uniform: sweep_r4 has been scored on twelve victims and sweep_r0..r3 on
    two. A maximum over twelve draws beats a maximum over two by construction,
    so that version measured evaluation coverage and drew it as a realism
    effect. Holding the victim fixed is the whole fix.

    If no single victim covers every run in the series there is no honest curve
    to draw, and this returns None with the reason rather than drawing one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {}
    for tag in sorted(os.listdir(N.PATCH_DIR)) if os.path.isdir(N.PATCH_DIR) else []:
        mp = os.path.join(N.PATCH_DIR, tag, "run.json")
        if not os.path.isfile(mp) or (patch and not tag.startswith(patch)):
            continue
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("realism_level") is None:
            continue
        cells = {}
        for a in ("seg", "det"):
            for t, v, rp in result_files(a, tag):
                if t != tag:
                    continue
                d = summarise(read_rows(rp), a)
                if d["asr"] is not None:
                    cells[(a, v)] = d
        if cells:
            runs[tag] = (meta, cells)

    if len(runs) < 2:
        return None

    common = set.intersection(*[set(c) for _, c in runs.values()])
    if arm or victim:
        common = {k for k in common
                  if (not arm or k[0] == arm) and (not victim or k[1] == victim)}
    if not common:
        print("  realism curve: no victim covers all of %s -- not drawn, "
              "because a curve over different victims per point is a curve "
              "about evaluation coverage" % ", ".join(sorted(runs)))
        return None

    prefer = [("seg", N.ATTACK_MODELS[0]), ("det", N.DET_ATTACK_MODELS[0])]
    key = next((k for k in prefer if k in common), sorted(common)[0])

    # One y per x or it is not a curve. Several runs can share a realism level
    # while differing in size, steps and supercell -- level 4 holds task_joint,
    # L_dof, roof_r4 and more -- and joining those with a line draws a shape
    # that is really about those other axes. Name them and refuse.
    by_level = {}
    for tag, (m, _) in runs.items():
        by_level.setdefault(m["realism_level"], []).append(tag)
    clashes = {k: v for k, v in by_level.items() if len(v) > 1}
    if clashes:
        print("  realism curve: not drawn -- %s share a realism level, and "
              "they differ in size, steps and supercell, so a line through "
              "them is not a realism curve. Pass a single series (patch="
              "'sweep' or 'full')."
              % "; ".join("level %s: %s" % (k, ", ".join(sorted(v)))
                          for k, v in sorted(clashes.items())))
        return None

    pts = sorted(((m["realism_level"], c[key]["asr"], c[key]["n_attackable"],
                   _ink_label(m))
                  for m, c in runs.values()), key=lambda q: q[0])

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot([q[0] for q in pts], [q[1] for q in pts], "o-", color="#2b6cb0")
    for lvl, asr, _, label in pts:
        ax.annotate(label, (lvl, asr),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5)
    ns = sorted({q[2] for q in pts})
    ax.set_xlabel("realism level (nested constraint sets, not loss weights)")
    ax.set_ylabel("attack success rate")
    # the victim and the attackable count belong ON the figure: they are the
    # two things that make the points comparable, and a reader cannot check
    # them from the axes. As the axes title, under the real one.
    fig.suptitle("What the attack gives up to become printable", fontsize=11)
    ax.set_title("%s/%s, attackable n=%s" % (key[0], key[1],
                 ns[0] if len(ns) == 1 else "%d-%d (UNEQUAL)" % (ns[0], ns[-1])),
                 fontsize=8, color="#666666")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out or os.path.join(N.FIGURE_DIR, "realism_curve.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=N.PANEL_DPI)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    return out


def plot_envelope(patch, arm, victim, out=None):
    """ASR against nadir angle: the transfer envelope."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(N.RESULT_DIR, "%s_%s__%s.csv" % (arm, patch, victim))
    if not os.path.isfile(path):
        return None
    rows = read_rows(path)
    xs, ys, ns = [], [], []
    for lo, hi in N.REPORT_ANGLE_BINS:
        sel = [r for r in rows if lo <= fnum(r, "theta_deg") < hi]
        d = summarise(sel, arm) if sel else {"asr": None, "n_attackable": 0}
        if d["asr"] is not None:
            xs.append(0.5 * (lo + hi))
            ys.append(d["asr"])
            ns.append(d["n_attackable"])
    if len(xs) < 2:
        return None

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(xs, ys, "o-", color="#c05621")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate("n=%d" % n, (x, y), textcoords="offset points",
                    xytext=(0, 7), fontsize=7, ha="center")
    ax.axvspan(0, N.NADIR_MAX_THETA_DEG, color="#2b6cb0", alpha=0.10)
    ax.text(N.NADIR_MAX_THETA_DEG / 2, ax.get_ylim()[1] * 0.96, "optimised\nhere",
            fontsize=7.5, ha="center", va="top", color="#2b6cb0")
    ax.set_xlabel("nadir angle (degrees)")
    ax.set_ylabel("attack success rate")
    ax.set_title("Transfer envelope: %s on %s" % (patch, victim))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out or os.path.join(N.FIGURE_DIR,
                              "envelope_%s_%s__%s.png" % (arm, patch, victim))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=N.PANEL_DPI)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch", default=None, help="restrict to tags with this prefix")
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()

    os.makedirs(N.FIGURE_DIR, exist_ok=True)
    parts = [
        "=" * 92,
        "PhysDecal results",
        "=" * 92,
        table_main(a.patch),
        table_realism(a.patch),
    ]

    files = result_files("seg", a.patch)
    if files:
        tag, victim, _ = files[0]
        parts.append(table_binned(tag, "seg", victim, "theta_deg",
                                  N.REPORT_ANGLE_BINS, "nadir angle"))
        parts.append(table_binned(tag, "seg", victim, "altitude_m",
                                  N.REPORT_ALT_BINS, "altitude m"))
    parts.append(table_physical())
    parts.append(table_provenance(a.patch))

    text = "\n".join(p for p in parts if p)
    print(text)
    out = os.path.join(N.FIGURE_DIR, "report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\nwrote %s" % out)

    if not a.tables_only:
        # One file per controlled series, never one shared name: a single
        # realism_curve.png silently meant a different thing depending on
        # --patch, which is how a two-point curve at level 4 got published.
        for sname in ([a.patch] if a.patch else ["sweep", "full"]):
            p = plot_realism_curve(sname, out=os.path.join(
                N.FIGURE_DIR, "realism_curve_%s.png" % sname))
            if p:
                print("wrote %s" % p)
        for tag, victim, _ in result_files("seg", a.patch)[:3]:
            p = plot_envelope(tag, "seg", victim)
            if p:
                print("wrote %s" % p)
        for tag, victim, _ in result_files("det", a.patch)[:3]:
            p = plot_envelope(tag, "det", victim)
            if p:
                print("wrote %s" % p)


if __name__ == "__main__":
    main()
