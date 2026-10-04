"""
The H1 palette ablation as one figure.

Three panels, one shared y axis, because all three show the same quantity and
the comparison between them is the point. The grey band on every panel is the
measured-gamut run plus or minus the between-seed standard deviation of the
optimiser on this victim, so a point inside the band is not separable from
run-to-run variation and the reader can see that without arithmetic.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from physdecal import config as N
from physdecal.core import decal as D

R = N.RESULT_DIR
P = os.path.join(N.REFERENCE_DIR, "palettes")
SEED_SD = 0.0178          # pooled between-seed sd, segformer_b0, Table 5
CUBE_C = 54.6             # cube ink set's own mean chroma at k = 8

BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"


def asr(tag):
    p = os.path.join(R, "seg_%s__segformer_b0.csv" % tag)
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    att = [x for x in rows if x["attackable"] == "1"]
    return sum(x["success"] == "1" for x in att) / len(att)


def palette(name):
    a = np.loadtxt(os.path.join(P, name + ".csv"), delimiter=",",
                   dtype=np.float32).reshape(-1, 3)
    lab = D.rgb_to_lab(torch.from_numpy(a).T.reshape(1, 3, -1, 1)
                       ).reshape(3, -1).T.numpy()
    return a, len(a), float(np.hypot(lab[:, 1], lab[:, 2]).mean())


def dress(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)


def band(ax, base):
    ax.axhspan(base - SEED_SD, base + SEED_SD, color=GRID, alpha=0.55, lw=0,
               zorder=0)
    ax.axhline(base, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)


def main():
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    base = asr("h1_chroma_g100")
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.05, 1.05, 1.0],
                                          "wspace": 0.12})

    # ---------------------------------------------------- A. chroma sweep
    ax = axes[0]
    names = ["chroma_g050", "chroma_g100", "chroma_g150", "chroma_g200"]
    xs = [palette(n)[2] for n in names]
    ys = [asr("h1_" + n) for n in names]
    band(ax, base)
    ax.axvline(CUBE_C, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=1)
    ax.text(CUBE_C - 1.6, 0.075, "cube ink set", rotation=90, ha="right",
            va="center", fontsize=7.5, color=MUTED)
    ax.plot(xs, ys, "-o", color=BLUE, lw=2.0, ms=8, mec="white", mew=1.4,
            zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate("%.3f" % y, (x, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=7.5, color=INK)
    ax.set_xlabel("palette mean chroma $C^*$", fontsize=9, color=INK)
    ax.set_title("(a)  chroma varied, $k=8$", fontsize=9.5, color=INK,
                 loc="left", pad=8)
    ax.set_xlim(8, 66)
    dress(ax)

    # ----------------------------------------------- B. cardinality sweep
    ax = axes[1]
    names = ["card_k03", "card_k05", "chroma_g100", "card_k16", "card_k32"]
    xs = [palette(n)[1] for n in names]
    ys = [asr("h1_" + n) for n in names]
    band(ax, base)
    ax.plot(xs, ys, "-o", color=BLUE, lw=2.0, ms=8, mec="white", mew=1.4,
            zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate("%.3f" % y, (x, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=7.5, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("palette cardinality $k$", fontsize=9, color=INK)
    ax.set_title("(b)  cardinality varied, $C^*\\approx 33$", fontsize=9.5,
                 color=INK, loc="left", pad=8)
    ax.set_xlim(2.4, 42)
    dress(ax)

    # ------------------------------------------------- C. random palettes
    ax = axes[2]
    names = ["chroma_g100", "rand_s11", "rand_s22", "rand_s33"]
    labels = ["measured", "random A", "random B", "random C"]
    ys = [asr("h1_" + n) for n in names]
    band(ax, base)
    for i, (n, y) in enumerate(zip(names, ys)):
        c = BLUE if i == 0 else ORANGE
        ax.plot([i], [y], "o", color=c, ms=9, mec="white", mew=1.4, zorder=3)
        ax.annotate("%.3f" % y, (i, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=7.5, color=INK)
        # the palette itself, drawn under the axis: this panel's claim is
        # about which colours were used, so show them
        cols = palette(n)[0]
        for j, col in enumerate(cols):
            ax.add_patch(plt.Rectangle(
                (i - 0.34 + j * 0.68 / len(cols), -0.019),
                0.68 / len(cols), 0.013, facecolor=tuple(col), lw=0,
                clip_on=False, zorder=4))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis="x", pad=26)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_title("(c)  $k$, $L^*$, $C^*$ matched; hues redrawn", fontsize=9.5,
                 color=INK, loc="left", pad=8)
    dress(ax)

    axes[0].set_ylabel("attack success rate", fontsize=9, color=INK)
    axes[0].set_ylim(0.0, 0.19)
    # No footnote here: the LaTeX caption carries it, and a fig.text at this
    # width collided with panel (c)'s swatches.
    fig.savefig(os.path.join(N.FIGURE_DIR, "h1_palette_ablation.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(N.FIGURE_DIR, "h1_palette_ablation.pdf"),
                bbox_inches="tight", facecolor="white")
    print("wrote %s/h1_palette_ablation.{png,pdf}" % N.FIGURE_DIR)


if __name__ == "__main__":
    main()
