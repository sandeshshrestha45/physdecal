r"""
physdecal fig-ce
The summary figures for paper Secs. res-ce and res-ce-det: every simple-patch
condition on the white-box victim, and transfer of the two printed variants to
the other victims of the same task.

    physdecal fig-ce            # both
    physdecal fig-ce --arm det  # one

Read straight from the result CSVs, three seeds per condition, so the figures
cannot drift from the tables. Dot = three-seed mean, line = seed min to max,
small marks = the individual seeds.

Colour follows the patch, the same in both panels and both figures: blue is the
printed patch optimised through the printer response, orange the free-pixel
patch printed, aqua the untargeted -CE patch printed, grey every other variant
(context). Slots 1-3 of the reference categorical palette, which validate
all-pairs.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physdecal import config as N  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CONTEXT, INK, INK2, GRID = "#8f8e8a", "#0b0b0b", "#52514e", "#e4e3df"
REF = "#b5b4af"                 # reference lines: visible, quieter than data
SEEDS = ("", "_s101", "_s202")

# Per arm: white-box victim and its label, the tag of every condition, the
# controls, the transfer victims and the x range of panel (a).
ARMS = {
    "seg": {
        "victim": "segformer_b0", "name": "SegFormer-B0",
        "tags": {"hinge": "dig_segformer_b0", "ce": "ce_dig_segformer_b0",
                 "chain": "ce_simple_segformer_b0",
                 "ink": "ce_ink_segformer_b0", "snap": "ce_pal_snap_segformer_b0",
                 "logit": "ce_pal_logit_segformer_b0",
                 "free": "ce_simple_segformer_b0", "unt": "ce_unt_lut_segformer_b0",
                 "lut": "ce_lut_segformer_b0"},
        "grey": "ce_ink_segformer_b0__grey",
        "grey_p": "ce_lut_segformer_b0__grey__printed",
        "transfer": [("segformer_b0", "SegFormer-B0 (white box)"),
                     ("fcn_r50", "FCN-R50"), ("deeplabv3_r101", "DeepLabv3-R101"),
                     ("upernet_convnext_t", "UPerNet-ConvNeXt-T"),
                     ("upernet_swin_t", "UPerNet-Swin-T"),
                     ("segformer_b2", "SegFormer-B2"),
                     ("clipseg__a_car", "CLIPSeg, “a car”"),
                     ("clipseg__a_parked_car", "CLIPSeg, “a parked car”")],
        "xmax": 0.32, "out": "ce_results.png"},
    "det": {
        "victim": "retinanet", "name": "RetinaNet",
        "tags": {"hinge": "ddig_retinanet", "ce": "dce_dig_retinanet",
                 "chain": "dce_simple_retinanet",
                 "ink": "dce_ink_retinanet", "snap": "dce_pal_snap_retinanet",
                 "logit": "dce_pal_logit_retinanet",
                 "free": "dce_simple_retinanet", "unt": "dce_unt_lut_retinanet",
                 "lut": "dce_lut_retinanet"},
        "grey": "ddig_retinanet__grey",
        "grey_p": "dce_lut_retinanet__grey__printed",
        "transfer": [("retinanet", "RetinaNet (white box)"),
                     ("fasterrcnn", "Faster R-CNN"), ("fcos", "FCOS"),
                     ("yolo", "YOLOv8n")],
        "xmax": 0.36, "out": "ce_results_det.png"},
}


def asr(arm, tag, victim):
    with open(os.path.join(N.RESULT_DIR, "%s_%s__%s.csv" % (arm, tag, victim)),
              newline="", encoding="utf-8") as f:
        att = [r for r in csv.DictReader(f) if r["attackable"] == "1"]
    return float(np.mean([int(r["success"]) for r in att]))


def seeds(arm, base, suffix, victim):
    return np.array([asr(arm, base + s + suffix, victim) for s in SEEDS])


def dot(ax, y, vals, colour, offset=0.0):
    y = y + offset
    ax.plot([vals.min(), vals.max()], [y, y], color=colour, lw=2,
            solid_capstyle="round", zorder=2)
    ax.scatter(vals, [y] * len(vals), s=10, color=colour, alpha=0.45,
               linewidths=0, zorder=3)
    ax.scatter([vals.mean()], [y], s=46, color=colour, edgecolors="white",
               linewidths=1.5, zorder=4)


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(axis="y", length=0, labelsize=8, labelcolor=INK)
    ax.tick_params(axis="x", labelsize=8, colors=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def build(arm):
    c = ARMS[arm]
    v, t = c["victim"], c["tags"]
    S = lambda key, suffix="": seeds(arm, t[key], suffix, v)  # noqa: E731
    # (label, values, colour), top to bottom, with group headers as None
    rows = [
        ("Digital (not printed)", None, None),
        ("Hinge, free pixels", S("hinge"), CONTEXT),
        ("Targeted CE, free pixels", S("ce"), CONTEXT),
        ("Targeted CE, free pixels + print chain", S("chain"), CONTEXT),
        ("32 measured inks", None, None),
        ("Level 3 (band + flat + inks)", S("ink"), CONTEXT),
        ("Snap to nearest ink", S("snap"), CONTEXT),
        ("Per-pixel ink logits", S("logit"), CONTEXT),
        ("Printed (scored through P)", None, None),
        ("Free pixels, targeted CE", S("free", "__printed"), ORANGE),
        ("Optimised through P, untargeted $-$CE", S("unt", "__printed"), AQUA),
        ("Optimised through P, targeted CE", S("lut", "__printed"), BLUE),
    ]
    grey = asr(arm, c["grey"], v)
    grey_p = asr(arm, c["grey_p"], v)
    decal = asr(arm, "task_joint_ink", v)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.6),
                               gridspec_kw={"width_ratios": [1.15, 1]})

    # ---------------------------------------------------------------- (a)
    ys, labels = [], []
    y = 0
    for label, vals, colour in rows:
        if vals is None:
            y -= 0.35
            a.text(-0.005, y, label, transform=a.get_yaxis_transform(),
                   ha="right", va="center", fontsize=8, color=INK2,
                   style="italic")
            y -= 1
            continue
        dot(a, y, vals, colour)
        ys.append(y)
        labels.append(label)
        y -= 1
    # the reference lines can sit close together, so their labels go on
    # opposite sides of their lines
    refs = sorted([(grey_p, "grey square,\nprinted"), (grey, "grey\nsquare"),
                   (decal, "measured-gamut\ndecal")])
    for i, (x, name) in enumerate(refs):
        ha, dx = ("right", -0.003) if i == 0 else ("left", 0.003)
        a.axvline(x, color=REF, lw=1, zorder=1)
        # the third label rides a second row, clear of the first two
        a.text(x + dx, 1.0 if i < 2 else 1.95, name, ha=ha, va="bottom",
               fontsize=7, color=INK2,
               bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    a.set_yticks(ys)
    a.set_yticklabels(labels)
    a.set_ylim(y + 0.4, 2.8)
    a.set_xlim(0, c["xmax"])
    a.set_xlabel("attack success rate, %s (white box)" % c["name"],
                 fontsize=8.5, color=INK)
    a.set_title("(a) Every condition, three seeds", fontsize=9.5,
                loc="left", color=INK)
    style(a)

    # ---------------------------------------------------------------- (b)
    victims = c["transfer"]
    for i, (key, _) in enumerate(victims):
        ctrl = asr(arm, c["grey_p"], key)
        dot(b, -i, seeds(arm, t["lut"], "__printed", key) - ctrl,
            BLUE, offset=0.14)
        dot(b, -i, seeds(arm, t["free"], "__printed", key) - ctrl,
            ORANGE, offset=-0.14)
    b.axvline(0, color=REF, lw=1, zorder=1)
    b.set_yticks([-i for i in range(len(victims))])
    b.set_yticklabels([name for _, name in victims])
    b.set_ylim(-len(victims) + 0.4, 0.6)
    b.set_xlabel("net attack success over a printed grey square",
                 fontsize=8.5, color=INK)
    b.set_title("(b) Transfer of the printed patches, three seeds",
                fontsize=9.5, loc="left", color=INK)
    style(b)

    handles = [plt.Line2D([], [], marker="o", ls="-", lw=2, ms=6, color=col,
                          markeredgecolor="white", label=lab)
               for col, lab in ((BLUE, "optimised through P, targeted CE, printed"),
                                (AQUA, "optimised through P, untargeted $-$CE, printed"),
                                (ORANGE, "free pixels, targeted CE, printed"),
                                (CONTEXT, "other variants"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(N.FIGURE_DIR, c["out"])
    fig.savefig(out, dpi=N.PANEL_DPI * 2, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default=None, choices=list(ARMS))
    a = ap.parse_args()
    for arm in ([a.arm] if a.arm else list(ARMS)):
        build(arm)


if __name__ == "__main__":
    main()
