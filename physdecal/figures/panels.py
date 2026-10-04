r"""
physdecal figures
The inference figures: what the victim actually predicts, clean against
patched, for segmentation and for detection.

    physdecal figures --patch segformer_b0_joint_r4 --panel seg
    physdecal figures --patch segformer_b0_joint_r4 --panel det
    physdecal figures --patch segformer_b0_joint_r4 --panel both
    physdecal figures --panel gallery --patches a b c
    physdecal figures --patch X --panel seg --frames <frame_id> ...

THESE ARE THE FIGURES THE PAPER IS JUDGED ON
--------------------------------------------
A table of ASR tells a reader the attack worked. It does not show them the
failure, and a physical-attack paper that cannot show the failure is asking to
be taken on faith. Each panel is built so that everything needed to check the
claim is inside the picture: the ground truth, the clean prediction, the
patched frame, the adversarial prediction, and the per-instance IoU or score
printed on the tile it belongs to.

SELECTION IS STATED, NOT HIDDEN
-------------------------------
By default the panel shows SUCCESSES -- instances the victim got right clean
and wrong under the patch -- because that is what the figure is for. That is a
selected sample and the caption emitted alongside says so, in those words,
together with how many instances were available to select from. Use
--select attackable for an unselected draw from the attackable set, or
--frames to name exact frames and remove the choice entirely.
"""

import argparse
import csv
import os
import textwrap

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from physdecal import config as N
from physdecal.core import data as D
from physdecal.core import patch as P
from physdecal.evaluation import evaluate as E


GREEN = (0.20, 0.80, 0.35)
RED = (0.90, 0.20, 0.25)
BLUE = (0.25, 0.55, 0.95)
AMBER = (0.98, 0.70, 0.15)


# --------------------------------------------------------------- selection

CLEAR_DROP = 0.20              # select="clear": minimum drop to draw
# Viewpoints already drawn in another figure, set by --exclude-frames. Keyed
# on the frame id without its lighting prefix, so the same vehicle from the
# same viewpoint under different light is not drawn twice either.
EXCLUDE_VIEWS = set()


def _view(frame_id):
    return frame_id.split("_", 1)[1]


def _load_patch(tag, device):
    """
    (patch, size_m, size_mode) for a results tag. A tag ending "__printed" is
    the patch as physdecal evaluate --printed scored it: through the measured
    printer colour response, so the figure shows what the table counted.
    """
    base = tag[:-len("__printed")] if tag.endswith("__printed") else tag
    patch = E.load_patch(base, device)
    if base != tag:
        from physdecal.core import printchain as PC
        patch = PC.PrinterLUT(device=device)(patch.unsqueeze(0))[0]
    return patch, E.patch_size_of(base), E.size_mode_of(base)


def pick_instances(patch_tag, victim, arm, n, select, frames=None):
    """
    Choose which instances to draw, from a results CSV that already exists.

    Reading the CSV rather than re-scoring means the figure shows the same
    instances the table counted, which is the only way the two can be checked
    against each other.
    """
    path = os.path.join(N.RESULT_DIR, "%s_%s__%s.csv" % (arm, patch_tag, victim))
    if not os.path.isfile(path):
        raise SystemExit(
            "No %s.\nRun physdecal evaluate --patch %s --%s %s first: the figure "
            "draws the instances the table scored, so the table has to exist."
            % (path, patch_tag, "models" if arm == "seg" else "detectors",
               victim))
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if frames:
        want = set(frames)
        chosen = [r for r in rows if r["frame_id"] in want]
        return chosen, len(rows), "named frames"

    att = [r for r in rows if r["attackable"] == "1"]
    if select == "success":
        pool = [r for r in att if r["success"] == "1"]
        note = "successes"
    elif select == "clear":
        # Successes whose loss is visible, dropping the ones that only just
        # cross the ASR threshold (a box truck at IoU 0.503 -> 0.493 looks
        # unchanged). The share of successes kept goes in the caption.
        succ = [r for r in att if r["success"] == "1"]
        key = "iou_drop" if arm == "seg" else "score_drop"
        pool = [r for r in succ if float(r[key]) >= CLEAR_DROP]
        note = ("successes with %s >= %.2f (%d of %d successes)"
                % (key.replace("_", " "), CLEAR_DROP, len(pool), len(succ)))
    else:
        pool = att
        note = "attackable instances"
    if EXCLUDE_VIEWS:
        n_before = len(pool)
        pool = [r for r in pool if _view(r["frame_id"]) not in EXCLUDE_VIEWS]
        note += ", excluding %d viewpoints drawn in other figures" % (
            n_before - len(pool))
    if not pool:
        raise SystemExit(
            "No %s to draw for %s/%s. The attack did not succeed on this "
            "victim, which is a result -- report it rather than hunting for a "
            "frame." % (note, patch_tag, victim))

    # Spread the choice across the viewpoint range instead of taking the first
    # n, which would all come from one altitude and one azimuth and would make
    # the attack look more uniform than it is.
    pool.sort(key=lambda r: float(r["theta_deg"]))
    idx = np.linspace(0, len(pool) - 1, min(n, len(pool))).round().astype(int)
    return [pool[i] for i in idx], len(att), note


def resolve_victim(victim, rows):
    """
    Split a results-file victim key into (model key, prompt or None).

    The open-vocabulary arm is scored once per prompt and its results are
    written as "clipseg__a_car", so the figure has to undo that to know which
    model to load. The prompt is taken from the CSV's own prompt column rather
    than by un-slugging the filename, because a slug is lossy and the column
    is exact.
    """
    if victim in N.MODELS:
        return victim, None
    base = victim.split("__")[0]
    if base not in N.MODELS:
        raise SystemExit("Unknown victim '%s'" % victim)
    prompt = next((r.get("prompt") for r in rows if r.get("prompt")), None)
    return base, prompt


def instance_for(index, frame_id, target):
    for it in index:
        if it["frame_id"] == frame_id and it["target"] == target:
            return it
    return None


def load_one(item, device):
    """One instance as the batch of 1 the rest of the pipeline expects."""
    ds = D.NadirPatchDataset([item])
    b = D.collate([ds[0]])
    return ({k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in b.items()}, b["meta"][0])


# ------------------------------------------------------------------ drawing

def crop_of(meta, shape, expand=2.6):
    return E.window_of(meta["bbox"], shape, expand)


def _window_cut(meta, shape):
    """Slice a B x 1 x H x W tensor to the EVALUATION window. Tiles are drawn
    on the wider crop_of, but every number printed on them is scored here, so
    it is the number in the results CSV. Scoring the display crop instead
    counts any other vehicle in the extra margin and disagrees with the
    tables."""
    x0, y0, x1, y1 = E.window_of(meta["bbox"], shape)
    return lambda t: t[0, 0].cpu().numpy()[y0:y1, x0:x1]


def overlay(rgb, mask, colour, alpha=0.45):
    out = rgb.copy()
    m = mask.astype(bool)
    for c in range(3):
        out[..., c][m] = (1 - alpha) * out[..., c][m] + alpha * colour[c]
    return out


def outline(rgb, mask, colour, width=1):
    m = (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = (rgb * 255).astype(np.uint8).copy()
    cv2.drawContours(out, cnts, -1, tuple(int(255 * c) for c in colour), width)
    return out.astype(np.float32) / 255.0


def tile(ax, img, title, sub=None):
    ax.imshow(np.clip(img, 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8.5, pad=3)
    if sub:
        ax.set_xlabel(sub, fontsize=7.5, labelpad=2)


# ------------------------------------------------------------- the seg panel

@torch.no_grad()
def seg_panel(patch_tag, victim, n=None, select="success", frames=None,
              device=None, out=None):
    from physdecal.victims import segmentation as M

    device = device or N.DEVICE
    n = n or N.PANEL_FRAMES
    rows, n_pool, note = pick_instances(patch_tag, victim, "seg", n, select,
                                        frames)
    model_key, prompt = resolve_victim(victim, rows)
    index = D.build_index("holdout", nadir_only=False)
    model = M.load_model(model_key, device=device)
    if prompt:
        model.set_prompt(prompt)
    patch, size_m, size_mode = _load_patch(patch_tag, device)

    fig, axes = plt.subplots(len(rows), 5,
                             figsize=(14.0, 2.85 * len(rows)), squeeze=False)

    for r, rec in enumerate(rows):
        item = instance_for(index, rec["frame_id"], rec["target"])
        if item is None:
            raise SystemExit("instance %s/%s is in the CSV but not the index"
                             % (rec["frame_id"], rec["target"]))
        b, meta = load_one(item, device)
        img, gt, inst = b["image"], b["gt"], b["inst"]

        adv, alpha, _ = P.composite(img, b["quad"], b["meta"], patch, inst,
                                    train=False, device=device, size_m=size_m,
                                    chain=None, size_mode=size_mode)
        p_clean = model(img)["p_veh"]
        p_adv = model(adv)["p_veh"]

        x0, y0, x1, y1 = crop_of(meta, img.shape[-2:])
        cut = lambda t: t[0].permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1] \
            if t.shape[1] == 3 else t[0, 0].cpu().numpy()[y0:y1, x0:x1]

        rgb_c, rgb_a = cut(img), cut(adv)
        g = cut(gt) > 0.5
        pc = cut(p_clean) > 0.5
        pa = cut(p_adv) > 0.5
        al = cut(alpha) > 0.5

        flipped = g & pc & (~pa)
        under = flipped & al
        away = flipped & (~al)
        # Numbers printed on the tiles are scored in the evaluation window,
        # not the wider display crop, so they equal the results CSV.
        wcut = _window_cut(meta, img.shape[-2:])
        wg, wpc, wpa = wcut(gt) > 0.5, wcut(p_clean) > 0.5, wcut(p_adv) > 0.5
        wal, wfl = wcut(alpha) > 0.5, wg & wpc & ~wpa
        iou_c, iou_a = E.iou(wpc, wg), E.iou(wpa, wg)
        n_under, n_away = int((wfl & wal).sum()), int((wfl & ~wal).sum())

        vp = "%.0f deg, %.0f m, %s" % (float(rec["theta_deg"]),
                                       float(rec["altitude_m"]),
                                       rec["lighting"])

        tile(axes[r][0], outline(rgb_c, g, BLUE), "clean frame" if r == 0 else "",
             "%s  %s" % (rec["target"], vp))
        tile(axes[r][1], overlay(rgb_c, pc, GREEN),
             "clean prediction" if r == 0 else "", "IoU %.3f" % iou_c)
        # The decal's SIZE AND COVERAGE, not just "decal on roof". The same
        # patch is a large square on a 4.3 m pickup and a speck on a 7.3 m box
        # truck, and a panel that does not say so reads as two different
        # patches. Taken from the results row, so it is the size actually used.
        cov = rec.get("coverage_frac")
        psz = rec.get("patch_size_m")
        sub = ("%.2f m decal, %.0f%% of vehicle"
               % (float(psz), 100 * float(cov))) if (cov and psz)             else "decal on roof"
        tile(axes[r][2], outline(rgb_a, al, AMBER),
             "patched frame" if r == 0 else "", sub)
        tile(axes[r][3], overlay(rgb_a, pa, RED),
             "prediction under attack" if r == 0 else "",
             "IoU %.3f  (drop %.3f)" % (iou_a, iou_c - iou_a))

        vis = np.stack([rgb_c.mean(-1)] * 3, -1) * 0.45
        vis = overlay(vis, under, AMBER, 0.95)
        vis = overlay(vis, away, RED, 0.95)
        tile(axes[r][4], vis, "vehicle pixels lost" if r == 0 else "",
             "under %d  away %d" % (n_under, n_away))

    handles = [mpatches.Patch(color=BLUE, label="ground truth outline"),
               mpatches.Patch(color=GREEN, label="predicted vehicle, clean"),
               mpatches.Patch(color=RED, label="predicted vehicle, attacked / lost away from patch"),
               mpatches.Patch(color=AMBER, label="patch footprint / lost under patch")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.002))

    title = ("Segmentation under a printed roof decal: %s on %s"
             % (patch_tag, model_key))
    if prompt:
        title += '   prompt: "%s"' % prompt
    fig.suptitle(title, fontsize=11, y=0.997)
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))

    out = out or os.path.join(N.FIGURE_DIR,
                              "panel_seg_%s__%s.png" % (patch_tag, victim))
    _save(fig, out)
    _caption(out, patch_tag, victim, "segmentation", note, len(rows), n_pool,
             select, frames)
    return out


# ------------------------------------------------------------- the det panel

def draw_boxes(ax, img, pred, ids, gt_box, off, score_thresh):
    """
    Draw the detections that fall inside this crop.

    Boxes outside the crop are SKIPPED rather than drawn off-axis. Matplotlib
    will happily place an artist outside the data limits and then expand the
    axes to contain it, which puts orphaned score labels in the figure margin
    and silently rescales the image tile. The axis limits are pinned below for
    the same reason.

    n counts only the boxes actually shown, so the "vehicle lost" verdict
    refers to the crop the reader is looking at.
    """
    H, W = img.shape[:2]
    ax.imshow(np.clip(img, 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    ox, oy = off

    gx0, gy0, gx1, gy1 = gt_box
    ax.add_patch(plt.Rectangle((gx0 - ox, gy0 - oy), gx1 - gx0, gy1 - gy0,
                               fill=False, ec=BLUE, lw=1.4, ls=":"))

    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    n = 0
    for bx, sc, lb in zip(boxes, scores, labels):
        if lb not in ids or sc < score_thresh:
            continue
        x0, y0, x1, y1 = bx[0] - ox, bx[1] - oy, bx[2] - ox, bx[3] - oy
        if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
            continue                      # entirely outside this crop
        n += 1
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, ec=GREEN, lw=1.8))
        ax.text(min(max(x0 + 2, 2), W - 40), min(max(y0 - 3, 8), H - 4),
                "vehicle %.2f" % sc, fontsize=7, color="white",
                bbox=dict(fc=GREEN, ec="none", pad=1.0, alpha=0.9))

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    return n


@torch.no_grad()
def det_panel(patch_tag, victim, n=None, select="success", frames=None,
              device=None, out=None):
    from physdecal.victims import detection as DET

    device = device or N.DEVICE
    n = n or N.PANEL_FRAMES
    rows, n_pool, note = pick_instances(patch_tag, victim, "det", n, select,
                                        frames)
    index = D.build_index("holdout", nadir_only=False)
    model = DET.load_detector(victim, device=device)
    patch, size_m, size_mode = _load_patch(patch_tag, device)

    fig, axes = plt.subplots(len(rows), 3,
                             figsize=(11.0, 3.0 * len(rows)), squeeze=False)

    for r, rec in enumerate(rows):
        item = instance_for(index, rec["frame_id"], rec["target"])
        b, meta = load_one(item, device)
        img, inst = b["image"], b["inst"]
        adv, alpha, _ = P.composite(img, b["quad"], b["meta"], patch, inst,
                                    train=False, device=device, size_m=size_m,
                                    chain=None, size_mode=size_mode)

        ids = model.vehicle_ids(meta["cls"])
        pc = model.predict(img)[0]
        pa = model.predict(adv)[0]

        x0, y0, x1, y1 = crop_of(meta, img.shape[-2:], expand=3.0)
        rgb_c = img[0].permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]
        rgb_a = adv[0].permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]
        al = alpha[0, 0].cpu().numpy()[y0:y1, x0:x1] > 0.5

        s_c = DET.max_vehicle_score(pc, meta["bbox"], ids)
        s_a = DET.max_vehicle_score(pa, meta["bbox"], ids)

        nc = draw_boxes(axes[r][0], rgb_c, pc, ids, meta["bbox"], (x0, y0),
                        N.DET_SCORE)
        axes[r][0].set_title("clean frame" if r == 0 else "", fontsize=8.5)
        axes[r][0].set_xlabel("%s  %.0f deg  %.0f m\nbest vehicle score %.3f"
                              % (rec["target"], float(rec["theta_deg"]),
                                 float(rec["altitude_m"]), s_c), fontsize=7.5)

        na = draw_boxes(axes[r][1], rgb_a, pa, ids, meta["bbox"], (x0, y0),
                        N.DET_SCORE)
        axes[r][1].set_title("under attack" if r == 0 else "", fontsize=8.5)
        lost, verdict = _det_verdict(pa, meta["bbox"], ids, na)
        axes[r][1].set_xlabel("best vehicle score %.3f  (drop %.3f)\n%s"
                              % (s_a, s_c - s_a, verdict), fontsize=7.5,
                              color=(RED if lost else "black"))

        vis = np.stack([rgb_a.mean(-1)] * 3, -1) * 0.5
        vis = overlay(vis, al, AMBER, 0.9)
        tile(axes[r][2], vis, "decal footprint" if r == 0 else "",
             "%.1f%% of the crop" % (100.0 * al.mean()))

    handles = [mpatches.Patch(color=BLUE, label="ground truth box"),
               mpatches.Patch(color=GREEN, label="detection above threshold"),
               mpatches.Patch(color=AMBER, label="patch footprint")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.002))
    fig.suptitle("Detection under a printed roof decal: %s on %s (threshold "
                 "%.2f)" % (patch_tag, victim, N.DET_SCORE), fontsize=11,
                 y=0.997)
    fig.tight_layout(rect=(0, 0.04, 1, 0.985))

    out = out or os.path.join(N.FIGURE_DIR,
                              "panel_det_%s__%s.png" % (patch_tag, victim))
    _save(fig, out)
    _caption(out, patch_tag, victim, "detection", note, len(rows), n_pool,
             select, frames)
    return out


# --------------------------------------------- the multi-patch comparison

SEG_BG = (0.914, 0.910, 0.894)   # background class, #e9e8e4, near-neutral
SEG_VEH = (0.165, 0.471, 0.839)  # vehicle class, #2a78d6, 3.6:1 on SEG_BG
SEG_GT = (0.12, 0.12, 0.11)      # ground-truth outline, near-black


def _seg_map(mask, gt=None):
    """A prediction as a flat two-class map, the way the patch-attack
    literature draws them: the figure is read by comparing coloured regions
    between columns, so the prediction is shown as a map rather than as an
    overlay that the underlying photograph can bleed through."""
    out = np.empty(mask.shape + (3,), np.float32)
    out[...] = SEG_BG
    out[mask] = SEG_VEH
    if gt is not None:
        # the true vehicle outline, so what the attack removed reads as the
        # gap between the outline and the fill
        out = outline(out, gt, SEG_GT, width=1)
    return out


def _det_verdict(pred, gt_box, ids, n_boxes):
    """The verdict under the scoring rule of evaluate.py: the vehicle is lost
    when no box matches it at DET_IOU_MATCH and DET_SCORE, even if boxes that
    cover part of it remain (a fragmented detection is a miss)."""
    from physdecal.victims import detection as DET
    if DET.best_hit(pred, gt_box, ids) is not None:
        return False, "detected"
    if n_boxes:
        return True, "VEHICLE LOST (%d partial box%s)" % (
            n_boxes, "" if n_boxes == 1 else "es")
    return True, "VEHICLE LOST"


def _seg_legend(fig):
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    handles = [mpatches.Patch(color=SEG_VEH, label="predicted vehicle"),
               mpatches.Patch(facecolor=SEG_BG, edgecolor="0.6",
                              label="predicted background"),
               mlines.Line2D([], [], color=SEG_GT, lw=1.2,
                             label="ground-truth vehicle outline")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))


@torch.no_grad()
def _compare_panel_det(tags, victim, n=None, select="success", frames=None,
                       device=None, out=None, labels=None):
    """
    The detection counterpart: one row per instance, clean and every patch,
    with the boxes the detector returns drawn on each.

    There is no second row here. A detector's output IS the overlay, so a
    separate map row would repeat the same picture; the score and the surviving
    box count carry what the segmentation map carries in the other variant.
    """
    from physdecal.victims import detection as DET

    device = device or N.DEVICE
    n = n or N.PANEL_FRAMES
    rows, n_pool, note = pick_instances(tags[0], victim, "det", n, select,
                                        frames)
    index = D.build_index("holdout", nadir_only=False)
    model = DET.load_detector(victim, device=device)
    loaded = [(t,) + _load_patch(t, device) for t in tags]
    cols = 1 + len(loaded)
    fig, axes = plt.subplots(len(rows), cols,
                             figsize=(3.4 * cols, 3.0 * len(rows)),
                             squeeze=False)

    for r, rec in enumerate(rows):
        item = instance_for(index, rec["frame_id"], rec["target"])
        if item is None:
            raise SystemExit("instance %s/%s is in the CSV but not the index"
                             % (rec["frame_id"], rec["target"]))
        b, meta = load_one(item, device)
        img, inst = b["image"], b["inst"]
        ids = model.vehicle_ids(meta["cls"])
        x0, y0, x1, y1 = crop_of(meta, img.shape[-2:], expand=3.0)
        rgb_c = img[0].permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]

        pc = model.predict(img)[0]
        s_c = DET.max_vehicle_score(pc, meta["bbox"], ids)
        draw_boxes(axes[r][0], rgb_c, pc, ids, meta["bbox"], (x0, y0),
                   N.DET_SCORE)
        axes[r][0].set_title("Clean" if r == 0 else "", fontsize=8.5)
        axes[r][0].set_xlabel("%s, %.0f m\nbest vehicle score %.3f"
                              % (rec["target"], float(rec["altitude_m"]), s_c),
                              fontsize=7.5)

        for c, (tag, patch, size_m, size_mode) in enumerate(loaded, start=1):
            adv, alpha, _ = P.composite(img, b["quad"], b["meta"], patch, inst,
                                        train=False, device=device,
                                        size_m=size_m, chain=None,
                                        size_mode=size_mode)
            rgb_a = adv[0].permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]
            pa = model.predict(adv)[0]
            s_a = DET.max_vehicle_score(pa, meta["bbox"], ids)
            na = draw_boxes(axes[r][c], rgb_a, pa, ids, meta["bbox"], (x0, y0),
                            N.DET_SCORE)
            axes[r][c].set_title((labels[c - 1] if labels else tag)
                                 if r == 0 else "", fontsize=8.5)
            lost, verdict = _det_verdict(pa, meta["bbox"], ids, na)
            axes[r][c].set_xlabel("score %.3f  (drop %.3f)\n%s"
                                  % (s_a, s_c - s_a, verdict), fontsize=7.5,
                                  color=(RED if lost else "black"))

    handles = [mpatches.Patch(color=BLUE, label="ground truth box"),
               mpatches.Patch(color=GREEN, label="detection above threshold")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Detection under each patch, %s" % victim, fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    out = out or os.path.join(N.FIGURE_DIR,
                              "compare_%s__%s.png" % ("_vs_".join(tags), victim))
    _save(fig, out)
    _caption(out, " vs ".join(tags), victim, "det", note, len(rows), n_pool,
             select, frames)
    return out


@torch.no_grad()
def compare_panel(tags, victim, n=None, select="success", frames=None,
                  device=None, out=None, arm="seg", labels=None):
    """
    One frame per block: the clean image and every patch applied to it, each
    above the segmentation it produces.

    Columns are attacks, rows alternate image and prediction. The comparison
    only means anything if every column is the SAME instance under the SAME
    victim, so instances are chosen once from the first tag's results and the
    remaining tags are composited onto those same frames rather than each
    choosing its own.
    """
    if arm == "det":
        return _compare_panel_det(tags, victim, n, select, frames, device, out,
                                  labels)
    from physdecal.victims import segmentation as M

    device = device or N.DEVICE
    n = n or N.PANEL_FRAMES
    rows, n_pool, note = pick_instances(tags[0], victim, "seg", n, select,
                                        frames)
    model_key, prompt = resolve_victim(victim, rows)
    index = D.build_index("holdout", nadir_only=False)
    model = M.load_model(model_key, device=device)
    if prompt:
        model.set_prompt(prompt)

    loaded = [(t,) + _load_patch(t, device) for t in tags]
    cols = 1 + len(loaded)
    # Two short rows per instance, not two tall ones: the crops are wide, so
    # height budgeted per tile leaves a band of dead space between the image
    # and the map it produced, and that band is exactly where the reader's eye
    # has to travel to make the comparison.
    fig, axes = plt.subplots(2 * len(rows), cols,
                             figsize=(3.05 * cols, 2.15 * 2 * len(rows)),
                             squeeze=False)

    for r, rec in enumerate(rows):
        item = instance_for(index, rec["frame_id"], rec["target"])
        if item is None:
            raise SystemExit("instance %s/%s is in the CSV but not the index"
                             % (rec["frame_id"], rec["target"]))
        b, meta = load_one(item, device)
        img, gt, inst = b["image"], b["gt"], b["inst"]
        x0, y0, x1, y1 = crop_of(meta, img.shape[-2:])
        cut = lambda t: (t[0].permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]
                         if t.shape[1] == 3
                         else t[0, 0].cpu().numpy()[y0:y1, x0:x1])

        wcut = _window_cut(meta, img.shape[-2:])
        p_clean = model(img)["p_veh"]
        pc, wg = cut(p_clean) > 0.5, wcut(gt) > 0.5
        gcut = cut(gt) > 0.5
        iou_c = E.iou(wcut(p_clean) > 0.5, wg)
        top, bot = axes[2 * r], axes[2 * r + 1]

        tile(top[0], cut(img), "Clean" if r == 0 else "",
             "%s, %.0f m" % (rec["target"], float(rec["altitude_m"])))
        tile(bot[0], _seg_map(pc, gcut), "", "IoU %.3f" % iou_c)

        for c, (tag, patch, size_m, size_mode) in enumerate(loaded, start=1):
            adv, alpha, _ = P.composite(img, b["quad"], b["meta"], patch, inst,
                                        train=False, device=device,
                                        size_m=size_m, chain=None,
                                        size_mode=size_mode)
            p_adv = model(adv)["p_veh"]
            pa = cut(p_adv) > 0.5
            iou_a = E.iou(wcut(p_adv) > 0.5, wg)
            tile(top[c], outline(cut(adv), cut(alpha) > 0.5, AMBER),
                 (labels[c - 1] if labels else tag) if r == 0 else "",
                 "%.1f m decal" % size_m)
            tile(bot[c], _seg_map(pa, gcut), "",
                 "IoU %.3f  (drop %.3f)" % (iou_a, iou_c - iou_a))

    fig.suptitle("Segmentation under each patch, %s" % victim, fontsize=11)
    _seg_legend(fig)
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    fig.subplots_adjust(hspace=0.30, wspace=0.06)
    out = out or os.path.join(N.FIGURE_DIR,
                              "compare_%s__%s.png" % ("_vs_".join(tags), victim))
    _save(fig, out)
    _caption(out, " vs ".join(tags), victim, "seg", note, len(rows), n_pool,
             select, frames)
    return out


# ---------------------------------------------------------------- gallery

def gallery(tags, out=None, title=None, labels=None, device=None):
    """
    The patches themselves, side by side, with their realism numbers.

    This is the figure that answers "does it look like a sticker" without
    asking the reader to take a word for it, and it is why the realism metrics
    are printed under each tile rather than in a separate table.
    """
    from physdecal.core import decal as DEC

    cols = min(len(tags), 4)
    nrows = -(-len(tags) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(2.6 * cols, 3.4 * nrows),
                             squeeze=False)
    for ax in axes.flat[len(tags):]:
        ax.axis("off")
    for i, tag in enumerate(tags):
        base = tag[:-len("__printed")] if tag.endswith("__printed") else tag
        patch = _load_patch(tag, device or N.DEVICE)[0]
        img = patch.permute(1, 2, 0).cpu().numpy()
        meta = E.run_meta(base)
        m = meta.get("realism_metrics", {})
        lvl = meta.get("realism_level", "?")
        pal = DEC.palette_size(meta)
        used = m.get("ink_count", "?")
        sub = "level %s   %s\nhf %.2f" % (
            lvl,
            "%s of %s inks" % (used, pal) if pal else "%s colours used" % used,
            m.get("hf_ratio", float("nan")))
        if "edge_iou" in m:
            sub += "   edge IoU %.2f" % m["edge_iou"]
        if base != tag:
            # the metrics above describe the requested colours, not these
            sub = "as printed, P(requested)"
        tile(axes[i // cols][i % cols], img, labels[i] if labels else tag, sub)
    # the caption has to say what the row actually varies: this figure is
    # used for the realism ladder, for the ink pair and for the lever
    # comparison, and those are three different claims
    fig.suptitle(title or "Patches at each realism level", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.06 / nrows))
    out = out or os.path.join(N.FIGURE_DIR, "gallery_patches.png")
    _save(fig, out)
    return out


# ------------------------------------------------------------------ helpers

def _save(fig, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=N.PANEL_DPI, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % out)


def _caption(out, tag, victim, arm, note, n_shown, n_pool, select, frames):
    """
    Write the caption next to the figure, including the selection rule.

    A figure of successes with no statement that they are successes is the
    easiest way to mislead in this literature, and it is usually accidental.
    Emitting the sentence automatically means the paper cannot forget it.
    """
    if frames:
        sel = ("The %d instances shown were named explicitly on the command "
               "line, so no selection was performed." % n_shown)
    else:
        sel = ("The %d instances shown are %s, drawn evenly across the nadir "
               "angle range from a pool of %d attackable instances. THIS IS A "
               "SELECTED SAMPLE, shown to illustrate the failure mode; the "
               "rate at which it occurs is in the results table, not here."
               % (n_shown, note, n_pool))
    txt = ("Figure: %s under the patch '%s', victim %s. Left to right: the "
           "clean frame with the ground truth outlined, the victim's clean "
           "prediction, the frame with the printed decal composited onto the "
           "vehicle roof through the metric roof homography, the victim's "
           "prediction under attack, and the vehicle pixels lost, separated "
           "into those under the patch footprint (occlusion) and those away "
           "from it (the adversarial effect). %s"
           % (arm.capitalize(), tag, victim, sel))
    path = out.replace(".png", "_caption.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(textwrap.wrap(txt, 78)) + "\n")
    print("       caption -> %s" % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch", default=None)
    ap.add_argument("--panel", default="both",
                    choices=["seg", "det", "both", "gallery", "compare"])
    ap.add_argument("--victim", default=None,
                    help="defaults to the first attack model / detector")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--select", default="success",
                    choices=["success", "clear", "attackable"])
    ap.add_argument("--frames", nargs="*", default=None,
                    help="draw these exact frame ids, removing the selection")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="compare/gallery: column titles, one per patch")
    ap.add_argument("--exclude-frames", nargs="*", default=None,
                    help="do not draw these frames' viewpoints (any "
                         "lighting), so figures do not repeat instances")
    ap.add_argument("--patches", nargs="*", default=None,
                    help="gallery only: the tags to show")
    ap.add_argument("--title", default=None,
                    help="gallery only: caption saying what the row varies")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", default="seg", choices=["seg", "det"],
                    help="compare only: which victim family --victim names")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    os.makedirs(N.FIGURE_DIR, exist_ok=True)
    EXCLUDE_VIEWS.update(_view(f) for f in (a.exclude_frames or []))
    if a.panel == "compare":
        if not a.patches or len(a.patches) < 2:
            raise SystemExit("--panel compare needs --patches tag tag [tag ...]")
        compare_panel(a.patches, a.victim or N.ATTACK_MODELS[0], a.n,
                      a.select, a.frames, a.device, a.out, a.arm, a.labels)
        return
    if a.panel == "gallery":
        if not a.patches:
            raise SystemExit("--panel gallery needs --patches tag [tag ...]")
        gallery(a.patches, a.out, a.title, a.labels, a.device)
        return

    if not a.patch:
        raise SystemExit("--patch is required for the seg and det panels")
    if a.panel in ("seg", "both"):
        seg_panel(a.patch, a.victim or N.ATTACK_MODELS[0], a.n, a.select,
                  a.frames, a.device)
    if a.panel in ("det", "both"):
        det_panel(a.patch, a.victim or N.DET_ATTACK_MODELS[0], a.n, a.select,
                  a.frames, a.device)


if __name__ == "__main__":
    main()
