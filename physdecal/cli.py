"""
The `physdecal` command: one entry point over every stage of the workflow.

    physdecal <command> [options]
    physdecal <command> --help

Each command is a module with its own argparse main(); this file only routes
to it, so `python -m physdecal.attack.optimize ...` and
`physdecal optimize ...` are the same program.
"""

import importlib
import sys

# command -> (module, one-line summary), in workflow order
COMMANDS = {
    # setup and preflight
    "check":        ("physdecal.evaluation.check",
                     "preflight: environment, dataset, split, manifold, weights; --gate"),
    "finetune":     ("physdecal.victims.finetune",
                     "decoder-only / head-only finetuning, backbones frozen"),
    "yolo":         ("physdecal.victims.yolo",
                     "YOLO arm: setup the dataset mirror, train, check"),
    "decal":        ("physdecal.core.decal",
                     "decal manifold self-check at every realism level"),
    "printchain":   ("physdecal.core.printchain",
                     "render print-chain draws of the anchor design"),
    "inkchart":     ("physdecal.physical.inkchart",
                     "make a printable ink chart / read its scan into printable.csv"),
    "palettes":     ("physdecal.physical.palettes",
                     "build the H1 ink sets (chroma, cardinality, random hue)"),
    # attack
    "optimize":     ("physdecal.attack.optimize",
                     "optimise a universal patch (seg / det / joint)"),
    "objcheck":     ("physdecal.attack.objcheck",
                     "per-frame vs universal patches: is the objective the limit?"),
    # evaluation and analysis
    "evaluate":     ("physdecal.evaluation.evaluate",
                     "score a patch or control on the holdout, both arms"),
    "report":       ("physdecal.evaluation.report",
                     "tables and summary plots from every result CSV"),
    "stats":        ("physdecal.evaluation.stats",
                     "vehicle bootstrap CIs, leave-one-vehicle-out, paired diffs"),
    "asr":          ("physdecal.evaluation.asr",
                     "one-line ASR per tag, nadir vs all viewpoints"),
    "viewpoint":    ("physdecal.evaluation.viewpoint",
                     "transfer envelope: angle, altitude, lighting, surface"),
    "lighting":     ("physdecal.evaluation.lighting",
                     "verify the three lighting labels are distinct conditions"),
    "manifest":     ("physdecal.evaluation.manifest",
                     "write out/EXPERIMENTS.md from everything on disk"),
    # figures
    "figures":      ("physdecal.figures.panels",
                     "inference panels: seg, det, compare, gallery"),
    "allfigs":      ("physdecal.figures.allfigs",
                     "every panel for one patch plus the roster table"),
    "gradcam":      ("physdecal.figures.gradcam",
                     "Grad-CAM of the vehicle evidence, clean vs patched"),
    "fig-ce":       ("physdecal.figures.paper_ce",
                     "paper figure: targeted-CE simple-patch results (seg, det)"),
    "fig-h1":       ("physdecal.figures.paper_h1",
                     "paper figure: H1 palette ablation"),
    "fig-loss":     ("physdecal.figures.paper_loss",
                     "paper figure: loss curves of a run against its re-run"),
    # physical
    "export-print": ("physdecal.physical.export_print",
                     "print-ready sheets with ArUco fiducials; --scale-table"),
    "physical":     ("physdecal.physical.capture",
                     "plan / ingest / annotate / track / score / panel photographs"),
    "automask":     ("physdecal.physical.automask",
                     "draft vehicle masks for a physical session"),
    "altitude":     ("physdecal.physical.altitude",
                     "fiducial-free altitude recovery for physical sessions"),
}


def usage():
    lines = ["usage: physdecal <command> [options]", "",
             "commands:"]
    for name, (_, summary) in COMMANDS.items():
        lines.append("  %-13s %s" % (name, summary))
    lines += ["", "Run  physdecal <command> --help  for a command's options."]
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        raise SystemExit("unknown command '%s'\n\n%s" % (cmd, usage()))
    module = importlib.import_module(COMMANDS[cmd][0])
    # Each module parses sys.argv itself, so hand it the remaining arguments
    # under a program name that reads correctly in its --help.
    sys.argv = ["physdecal %s" % cmd] + rest
    module.main()


if __name__ == "__main__":
    main()
