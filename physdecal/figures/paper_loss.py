r"""
physdecal fig-loss
Loss curves of an original run against its re-run, one panel per victim.

    physdecal fig-loss \
        --pair UPerNet-ConvNeXt-T wb_upernet_convnext_t wb_upernet_convnext_t_fixed \
        --pair UPerNet-Swin-T wb_upernet_swin_t wb_upernet_swin_t_fixed \
        --out wb_losscurves

The two runs of a pair optimise different attack terms (posterior hinge vs
log-odds hinge), whose values are not on a common scale. Each curve is
therefore indexed to its own first block: 1.0 is where the run started, below
1.0 is descending. Only the direction and relative size of the change are
comparable across the two, which is all the figure is used to show.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physdecal import config as N

ORIGINAL = "#eb6834"      # categorical slot 2
FIXED = "#2a78d6"         # categorical slot 1
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


def blocks(tag, width):
    with open(os.path.join(N.PATCH_DIR, tag, "history.csv"), newline="",
              encoding="utf-8") as f:
        seg = np.array([float(r["seg"]) for r in csv.DictReader(f)])
    n = len(seg) // width
    means = seg[:n * width].reshape(n, width).mean(1)
    return (np.arange(n) + 0.5) * width, means / means[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=3, action="append", required=True,
                    metavar=("TITLE", "ORIGINAL_TAG", "FIXED_TAG"))
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--out", default="wb_losscurves")
    a = ap.parse_args()

    fig, axes = plt.subplots(1, len(a.pair), figsize=(4.2 * len(a.pair), 3.1),
                             sharey=True, squeeze=False)
    lo, hi = 1.0, 1.0
    for ax, (title, orig, fixed) in zip(axes[0], a.pair):
        for tag, colour, style, name in ((orig, ORIGINAL, "--", "original"),
                                         (fixed, FIXED, "-", "fixed")):
            x, y = blocks(tag, a.block)
            lo, hi = min(lo, y.min()), max(hi, y.max())
            ax.plot(x, y, color=colour, lw=2, ls=style, label=name,
                    solid_capstyle="round")
            ax.annotate("%s %+.0f%%" % (name, 100 * (y[-1] - 1)),
                        (x[-1], y[-1]), xytext=(5, 0),
                        textcoords="offset points", va="center",
                        fontsize=8, color=INK)
        ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("optimisation step", fontsize=8.5, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlim(0, None)
        ax.margins(x=0.22)
    axes[0][0].set_ylabel("attack loss, relative to start", fontsize=8.5,
                          color=MUTED)
    pad = 0.08 * (hi - lo)
    axes[0][0].set_ylim(lo - pad, hi + pad)
    axes[0][-1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()

    out = os.path.join(N.FIGURE_DIR, a.out + ".png")
    fig.savefig(out, dpi=N.PANEL_DPI, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
