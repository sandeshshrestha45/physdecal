r"""
physdecal objcheck
Verifies the attack objective by separating two things a weak patch result
conflates: whether the OBJECTIVE can suppress the vehicle beyond the decal,
and whether ONE universal decal can do so across a fleet.

    physdecal objcheck --victim segformer_b0 \
        --universal task_joint_ink dig_segformer_b0

For a fixed, outcome-blind set of attackable nadir holdout instances, a patch
is optimised against each frame ALONE, with the same objective (paper Eq.
segloss, SEG_OBJECTIVE "vanish_prob"), the same 2.0 m footprint and the same
roof placement as the universal patches, at two realism levels:

    per-frame digital    level 0, free pixels
    per-frame printable  level 4, one ink per anchor region

No EOT and no print chain: this is the upper bound on what the objective can
do on that frame, not a physical attack. The universal patches are scored on
the same frames. If the per-frame patches remove the vehicle well beyond the
footprint and the universal ones do not, the objective is doing its job and
the limit is universality.

Writes results/objcheck_<victim>.csv, figures/objcheck_<victim>.{png,pdf},
figures/objcheck_<victim>.txt and figures/objcheck_<victim>_caption.txt.
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
from physdecal.core import data as D
from physdecal.core import decal as DEC
from physdecal.evaluation import evaluate as E
from physdecal.figures import panels as FG
from physdecal.core import losses as L
from physdecal.victims import segmentation as M
from physdecal.core import patch as P


def select_instances(model, n, device):
    """
    Evenly spaced over the nadir holdout, keeping the first n that the victim
    segments correctly clean. The rule never looks at an attack outcome, so
    the set is not chosen in anyone's favour.
    """
    index = [it for it in D.build_index("holdout", nadir_only=False)
             if it["theta_deg"] <= N.NADIR_MAX_THETA_DEG]
    picked = []
    for i in np.linspace(0, len(index) - 1, 4 * n).astype(int):
        b, meta = FG.load_one(index[i], device)
        x0, y0, x1, y1 = E.window_of(meta["bbox"], b["image"].shape[-2:])
        g = b["gt"][0, 0, y0:y1, x0:x1].cpu().numpy() > 0.5
        with torch.no_grad():
            pc = model(b["image"])["p_veh"][0, 0, y0:y1, x0:x1].cpu().numpy() > 0.5
        if E.iou(pc, g) >= N.ATTACKABLE_IOU:
            picked.append(index[i])
        if len(picked) == n:
            break
    return picked


def per_frame_patch(model, b, level, size_m, steps, device):
    """Optimise one patch against this frame only. Returns the patch."""
    man = DEC.DecalManifold(level=level, device=device, patch_size_m=size_m,
                            supercell=16)
    if level >= 4:
        tex = DEC.InkRegionTexture(man).to(device)
        make, lr = tex, N.INK_LOGIT_LR
    else:
        tex = P.PatchTexture(init="grey").to(device)
        make, lr = (lambda: man.forward(tex().unsqueeze(0))[0]), 0.05
    opt = torch.optim.Adam(tex.parameters(), lr=lr)
    for _ in range(steps):
        adv, alpha, _ = P.composite(b["image"], b["quad"], b["meta"], make(),
                                    b["inst"], train=False, device=device,
                                    size_m=size_m)
        out = model(adv)
        loss = L.seg_attack_loss(out["m_veh"], b["gt"], alpha=alpha,
                                 objective="vanish_prob", p_veh=out["p_veh"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return make().detach().clamp(0, 1)


def display_crop(meta, shape, aspect=1.75, expand=2.6):
    """Same landscape shape for every instance, so figure rows line up."""
    H, W = shape
    x0, y0, x1, y1 = meta["bbox"]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    h = max((y1 - y0) * expand, (x1 - x0) * expand / aspect)
    h = min(h, H, W / aspect)
    w = h * aspect
    cx = min(max(cx, w / 2), W - w / 2)
    cy = min(max(cy, h / 2), H - h / 2)
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


@torch.no_grad()
def score(model, b, meta, patch, size_m, device):
    """Metrics in the evaluation window (evaluate.window_of); the returned
    prediction is full frame, so the figure never draws unscored pixels as
    background."""
    img, gt = b["image"], b["gt"]
    x0, y0, x1, y1 = E.window_of(meta["bbox"], img.shape[-2:])
    cut = lambda t: t[y0:y1, x0:x1]
    g = cut(gt[0, 0].cpu().numpy()) > 0.5
    pc_full = model(img)["p_veh"][0, 0].cpu().numpy() > 0.5
    pc = cut(pc_full)
    if patch is None:
        return {"iou": E.iou(pc, g)}, pc_full, img, None
    adv, alpha, _ = P.composite(img, b["quad"], b["meta"], patch, b["inst"],
                                train=False, device=device, size_m=size_m)
    pa_full = model(adv)["p_veh"][0, 0].cpu().numpy() > 0.5
    pa = cut(pa_full)
    al = cut(alpha[0, 0].cpu().numpy()) > 0.5
    lost = g & pc & ~pa
    iou_a = E.iou(pa, g)
    return ({"iou": iou_a, "iou_drop": E.iou(pc, g) - iou_a,
             "lost_under": (lost & al).sum() / max(g.sum(), 1),
             "lost_away": (lost & ~al).sum() / max(g.sum(), 1),
             "success": int(iou_a < N.ASR_IOU_THRESHOLD)}, pa_full, adv, alpha)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--victim", default="segformer_b0")
    ap.add_argument("--universal", nargs="+",
                    default=["task_joint_ink", "dig_segformer_b0"])
    ap.add_argument("--n-stats", type=int, default=24)
    ap.add_argument("--n-show", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--labels", nargs="+", default=None,
                    help="column titles for the universal patches, in order")
    ap.add_argument("--size", type=float, default=2.0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = a.device or N.DEVICE

    model = M.load_model(a.victim, device=device)
    universal = [(t, E.load_patch(t, device), E.patch_size_of(t))
                 for t in a.universal]
    conds = ["universal: %s" % t for t, _, _ in universal] + \
            ["per-frame printable", "per-frame digital"]

    items = select_instances(model, a.n_stats, device)
    # Per-frame patches are cached, so a re-render reuses them rather than
    # re-optimising, which on a GPU is not bit-reproducible.
    cache_p = os.path.join(N.RESULT_DIR, "objcheck_%s_patches.pt" % a.victim)
    cache = torch.load(cache_p) if os.path.isfile(cache_p) else {}
    show = set(np.linspace(0, len(items) - 1, a.n_show).astype(int).tolist())
    rows, drawn = [], []
    for k, it in enumerate(items):
        b, meta = FG.load_one(it, device)
        patches = [(p, s) for _, p, s in universal]
        key = (meta["frame_id"], meta["target"], a.steps, a.size)
        if key not in cache:
            with torch.enable_grad():
                cache[key] = [per_frame_patch(model, b, lvl, a.size, a.steps,
                                              device).cpu() for lvl in (4, 0)]
        patches += [(p.to(device), a.size) for p in cache[key]]
        clean, pc, img, _ = score(model, b, meta, None, None, device)
        tiles = [(img, pc, None, clean)]
        for cond, (p, s) in zip(conds, patches):
            m, pa, adv, alpha = score(model, b, meta, p, s, device)
            rows.append(dict(frame_id=meta["frame_id"], target=meta["target"],
                             altitude_m=round(meta["altitude_m"], 1),
                             condition=cond, clean_iou=round(clean["iou"], 4),
                             **{k2: round(float(v), 4) for k2, v in m.items()}))
            tiles.append((adv, pa, alpha, m))
        if k in show:
            drawn.append((meta, b, tiles))
        print("%2d/%d %-40s done" % (k + 1, len(items), meta["frame_id"]))

    os.makedirs(N.RESULT_DIR, exist_ok=True)
    torch.save(cache, cache_p)
    rp = os.path.join(N.RESULT_DIR, "objcheck_%s.csv" % a.victim)
    with open(rp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["Objective check, victim %s, %d attackable nadir holdout "
             "instances, %d per-frame steps, vanish_prob objective"
             % (a.victim, len(items), a.steps), "",
             "%-34s %8s %8s %11s %10s %8s"
             % ("condition", "ASR", "IoU drop", "lost under", "lost away",
                "n")]
    for c in conds:
        r = [x for x in rows if x["condition"] == c]
        lines.append("%-34s %8.3f %8.3f %10.1f%% %9.1f%% %8d"
                     % (c, np.mean([x["success"] for x in r]),
                        np.mean([x["iou_drop"] for x in r]),
                        100 * np.mean([x["lost_under"] for x in r]),
                        100 * np.mean([x["lost_away"] for x in r]), len(r)))
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(N.FIGURE_DIR, "objcheck_%s.txt" % a.victim), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")

    # --- the figure: reference layout, images above the maps they produce
    labels = a.labels or [t for t, _, _ in universal]
    titles = ["Clean"] + ["Universal\n%s" % l for l in labels] + \
             ["Per-frame\nprintable", "Per-frame\ndigital"]
    cols = len(titles)
    tile_w, aspect = 2.75, 1.75
    fig, axes = plt.subplots(2 * len(drawn), cols, squeeze=False,
                             figsize=(tile_w * cols,
                                      2 * len(drawn) * (tile_w / aspect + 0.42)))
    for r, (meta, b, tiles) in enumerate(drawn):
        x0, y0, x1, y1 = display_crop(meta, b["image"].shape[-2:], aspect)
        rgb = lambda t: t[0].permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]
        for c, (img, pred, alpha, m) in enumerate(tiles):
            full = pred[y0:y1, x0:x1]
            im = rgb(img)
            if alpha is not None:
                im = FG.outline(im, alpha[0, 0].cpu().numpy()[y0:y1, x0:x1] > 0.5,
                                FG.AMBER)
            sub_top = ("%s, %.0f m" % (meta["target"], meta["altitude_m"])
                       if c == 0 else "")
            FG.tile(axes[2 * r][c], im, titles[c] if r == 0 else "", sub_top)
            sub = ("IoU %.3f" % m["iou"] if alpha is None else
                   "IoU %.2f  away %.0f%%" % (m["iou"], 100 * m["lost_away"]))
            g = b["gt"][0, 0].cpu().numpy()[y0:y1, x0:x1] > 0.5
            FG.tile(axes[2 * r + 1][c], FG._seg_map(full, g), "", sub)
    FG._seg_legend(fig)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.subplots_adjust(hspace=0.30, wspace=0.05)
    out = os.path.join(N.FIGURE_DIR, "objcheck_%s.png" % a.victim)
    FG._save(fig, out)

    cap = ("Segmentation of the same holdout instances under universal and "
           "per-frame patches, %s. Universal patches were optimised once over "
           "855 training instances; per-frame patches were optimised against "
           "the frame shown alone, for %d steps, with the same objective "
           "(Eq. segloss), 2.0 m footprint and roof placement, and without "
           "EOT or the print chain. The per-frame printable patch is held to "
           "the level-4 constraint set. Rows alternate input and predicted "
           "vehicle map; 'away' is the share of vehicle pixels lost outside "
           "the decal footprint (outlined). The %d instances shown are evenly "
           "spaced over a fixed set of %d attackable nadir holdout instances "
           "chosen without reference to any attack outcome; the rates over "
           "all %d are in the accompanying table."
           % (a.victim, a.steps, len(drawn), len(items), len(items)))
    with open(out.replace(".png", "_caption.txt"), "w", encoding="utf-8") as f:
        f.write(cap + "\n")


if __name__ == "__main__":
    main()
