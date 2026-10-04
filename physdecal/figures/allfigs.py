r"""
physdecal allfigs
Generates the complete artefact set for one patch: an inference panel for every
victim that was evaluated, plus the consolidated roster table.

    physdecal allfigs --patch L_dof

WHY A DRIVER RATHER THAN A LOOP IN THE SHELL
--------------------------------------------
Two things have to be handled per victim and neither is a one-liner.

A victim with no successes cannot have a success panel, and that is not an
error -- it is the result. Those victims are collected and listed rather than
crashing the run, and an "attackable" panel is drawn for them instead so the
paper still has a picture of what the patch does there.

The open-vocabulary arm appears once per prompt, so its panels have to carry
the prompt into the title and the filename or three figures become
indistinguishable.
"""

import argparse
import csv
import glob
import os

import numpy as np

from physdecal import config as N
from physdecal.figures import panels as FG


def victims_for(patch_tag, arm):
    """Every victim this patch was actually scored against, from the files."""
    out = []
    pre = "%s_%s__" % (arm, patch_tag)
    for p in sorted(glob.glob(os.path.join(N.RESULT_DIR, "%s*.csv" % pre))):
        key = os.path.basename(p)[len(pre):-4]
        if key.startswith("anchor__") or key.startswith("grey__") \
                or key.startswith("random__"):
            continue
        out.append(key)
    return out


def stats(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    att = [r for r in rows if r["attackable"] == "1"]
    suc = [r for r in att if r["success"] == "1"]
    return rows, att, suc


def make_panels(patch_tag, n=3, device=None):
    made, empty, failed = [], [], []
    for arm, fn in (("seg", FG.seg_panel), ("det", FG.det_panel)):
        for victim in victims_for(patch_tag, arm):
            path = os.path.join(N.RESULT_DIR,
                                "%s_%s__%s.csv" % (arm, patch_tag, victim))
            rows, att, suc = stats(path)
            if not att:
                empty.append((arm, victim, "attackable set empty"))
                continue
            select = "success" if suc else "attackable"
            try:
                out = fn(patch_tag, victim, n=n, select=select, device=device)
                made.append((arm, victim, len(suc), len(att), out))
                if not suc:
                    empty.append((arm, victim,
                                  "no successes: panel shows attackable "
                                  "instances the patch did NOT break"))
            except SystemExit as e:
                failed.append((arm, victim, str(e).splitlines()[0]))
    return made, empty, failed


def roster_table(patch_tag, control="anchor"):
    """
    The consolidated table: every victim, its control, and the net.

    Written as text and as LaTeX, because the number that goes in the paper
    should be generated from the rows rather than retyped from a terminal.
    """
    lines, tex = [], []
    hdr = ("%-5s %-34s %7s %8s %8s %8s %8s"
           % ("arm", "victim / prompt", "n_att", "clean", "ASR", control, "net"))
    lines += [hdr, "-" * len(hdr)]
    tex += [r"\begin{tabular}{llrrrrr}", r"\toprule",
            r"Arm & Victim & $n$ & Clean & ASR & Control & Net \\",
            r"\midrule"]

    for arm in ("seg", "det"):
        for victim in victims_for(patch_tag, arm):
            p = os.path.join(N.RESULT_DIR,
                             "%s_%s__%s.csv" % (arm, patch_tag, victim))
            rows, att, suc = stats(p)
            cp = os.path.join(N.RESULT_DIR, "%s_%s__%s__%s.csv"
                              % (arm, patch_tag, control, victim))
            if not att:
                lines.append("%-5s %-34s %7d %8s %8s %8s %8s"
                             % (arm, victim, 0, "-", "UNDEF", "-", "-"))
                tex.append(r"%s & \texttt{%s} & 0 & -- & undefined & -- & -- \\"
                           % (arm, victim.replace("_", r"\_")))
                continue
            ck = "clean_iou" if arm == "seg" else "clean_score"
            clean = float(np.mean([float(r[ck]) for r in att]))
            asr = len(suc) / len(att)
            if os.path.isfile(cp):
                _, catt, csuc = stats(cp)
                c = len(csuc) / len(catt) if catt else float("nan")
                net = asr - c
                cs, ns = "%.3f" % c, "%+.3f" % net
            else:
                cs, ns = "-", "-"
            lines.append("%-5s %-34s %7d %8.3f %8.3f %8s %8s"
                         % (arm, victim, len(att), clean, asr, cs, ns))
            tex.append(r"%s & \texttt{%s} & %d & %.3f & %.3f & %s & %s \\"
                       % (arm, victim.replace("_", r"\_"), len(att), clean,
                          asr, cs, ns))
    tex += [r"\bottomrule", r"\end{tabular}"]

    os.makedirs(N.FIGURE_DIR, exist_ok=True)
    txt_p = os.path.join(N.FIGURE_DIR, "roster_%s.txt" % patch_tag)
    tex_p = os.path.join(N.FIGURE_DIR, "roster_%s.tex" % patch_tag)
    head = ("Full victim roster for patch '%s'\n"
            "control = %s (the unoptimised reference decal at the same size)\n"
            "net = ASR minus control: what the OPTIMISATION is worth\n\n"
            % (patch_tag, control))
    with open(txt_p, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(lines) + "\n")
    with open(tex_p, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(head + "\n".join(lines))
    return txt_p, tex_p


def prompt_table(patch_tag):
    """The open-vocabulary arm on its own, because the prompt is the variable."""
    rowsout = []
    for victim in victims_for(patch_tag, "seg"):
        if not victim.startswith("clipseg"):
            continue
        p = os.path.join(N.RESULT_DIR, "seg_%s__%s.csv" % (patch_tag, victim))
        rows, att, suc = stats(p)
        prompt = next((r.get("prompt") for r in rows if r.get("prompt")), "?")
        n_all = len(rows)
        if att:
            rowsout.append((prompt, n_all, len(att), len(att) / n_all,
                            float(np.mean([float(r["clean_iou"]) for r in att])),
                            len(suc) / len(att)))
        else:
            rowsout.append((prompt, n_all, 0, 0.0, float("nan"), None))
    if not rowsout:
        return None
    lines = ["Open-vocabulary arm: CLIPSeg, one row per prompt",
             "published weights, never finetuned (its class is a sentence)",
             "",
             "%-34s %7s %7s %9s %8s %8s"
             % ("prompt", "n", "n_att", "attack.%", "clean", "ASR"),
             "-" * 78]
    for pr, n, na, fr, cl, a in rowsout:
        lines.append("%-34s %7d %7d %8.1f%% %8s %8s"
                     % ('"%s"' % pr, n, na, 100 * fr,
                        "-" if cl != cl else "%.3f" % cl,
                        "UNDEFINED" if a is None else "%.3f" % a))
    lines += ["",
              "An empty attackable set is UNDEFINED, not zero: the victim could",
              "not segment the vehicle without the patch, so there was nothing",
              "to attack. Prompt phrasing decides whether this model is",
              "measurable at all."]
    p = os.path.join(N.FIGURE_DIR, "clipseg_prompts_%s.txt" % patch_tag)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--control", default="anchor")
    ap.add_argument("--no-panels", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    os.makedirs(N.FIGURE_DIR, exist_ok=True)
    if not a.no_panels:
        made, empty, failed = make_panels(a.patch, a.n, a.device)
        print("\n%d panels written" % len(made))
        for arm, v, ns, na, out in made:
            print("   %-4s %-34s %d successes of %d attackable"
                  % (arm, v, ns, na))
        if empty:
            print("\nvictims with nothing to show, and why:")
            for arm, v, why in empty:
                print("   %-4s %-34s %s" % (arm, v, why))
        if failed:
            print("\nfailed:")
            for arm, v, why in failed:
                print("   %-4s %-34s %s" % (arm, v, why))

    print()
    roster_table(a.patch, a.control)
    prompt_table(a.patch)


if __name__ == "__main__":
    main()
