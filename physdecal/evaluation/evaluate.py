r"""
physdecal evaluate
Scores a patch on held-out data, for segmentation and for detection, and
writes one row per instance.

    physdecal evaluate --patch segformer_b0_joint_r4 --models all
    physdecal evaluate --patch segformer_b0_joint_r4 --detectors all
    physdecal evaluate --patch segformer_b0_joint_r4 --baseline grey

WHAT IS HELD OUT
----------------
Everything scored here is unseen by the optimisation in two independent ways.
The vehicles are holdout vehicles, split by instance rather than at random,
because two nadir frames of the same parked car from adjacent azimuths are
near duplicates and a random split leaks them across the boundary. And the
viewpoints run the full grid, 0 to 70 degrees off nadir, while the patch was
optimised at nadir only. Every oblique row is therefore a generalisation
measurement and not a fit.

THE ATTACKABLE SET IS THE WHOLE BALLGAME
----------------------------------------
ASR is computed ONLY over instances the victim got right without the patch:

    attackable   clean IoU >= ATTACKABLE_IOU          (segmentation)
                 detected clean at DET_SCORE          (detection)
    success      attackable AND the patched frame fails the same test

An instance the model could not see anyway is not a victory, and counting it
is the single most common way aerial attack numbers get inflated. The size of
the attackable set is written into every row, so a reader can always recover
what the denominator was.

UNDER AND AWAY
--------------
A vehicle pixel lost beneath the patch is occlusion: a sheet of paper covers
the roof and the roof stops being visible. A vehicle pixel lost OUTSIDE the
patch alpha is the adversarial effect proper. They are counted separately and
reported separately, and the propagation radii R50 and R90 are computed from
the away-flips alone, in metres on the ground via the per-frame GSD.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from physdecal import config as N
from physdecal.core import data as D
from physdecal.core import patch as P
from physdecal.core import decal as DEC


# ------------------------------------------------------------------- loading

def run_meta(tag):
    path = os.path.join(N.PATCH_DIR, tag, "run.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def patch_size_of(tag):
    """
    The metric size the patch was OPTIMISED at, read back from its run record.

    A patch tensor carries no scale, so scoring a 1.0 m patch under the current
    PATCH_SIZE_M of 1.2 m would paste it 44 percent larger by area and inflate
    every number it produced. The size travels with the run, not the config.
    """
    m = run_meta(tag)
    v = m.get("patch_size_m")
    if v:
        return float(v)
    if str(m.get("patch_size_mode") or "fixed") == "roof":
        # Uncapped roof mode: there is no single size, and returning one would
        # be read as a cap by patch.size_for and would silently shrink the
        # decal on every large-roofed vehicle. None means "each vehicle gets
        # its own roof maximum", which is what the run did.
        return None
    print("   no patch_size_m for '%s', falling back to PATCH_SIZE_M = %.2f m"
          % (tag, N.PATCH_SIZE_M))
    return float(N.PATCH_SIZE_M)


def size_mode_of(tag):
    """
    The sizing mode the patch was OPTIMISED under, from its run record.

    Carried with the run for the same reason the size is: scoring a
    roof-proportional patch under the fixed protocol, or the reverse, changes
    the decal's footprint on every instance and silently changes every number.
    Runs made before this option existed have no field and are "fixed".
    """
    return str(run_meta(tag).get("patch_size_mode") or "fixed")


def load_patch(tag, device):
    """The PROJECTED patch, which is the one that would be printed."""
    path = os.path.join(N.PATCH_DIR, tag, "patch.pt")
    if not os.path.isfile(path):
        raise SystemExit("No %s. Run physdecal optimize first." % path)
    blob = torch.load(path, map_location=device)
    return blob["patch"].to(device).clamp(0, 1)


def baseline_patch(kind, device, res=None):
    """
    The controls. Every ASR is reported next to these, and whatever survives
    the subtraction is the attack rather than the presence of a square.

      grey    a uniform mid-grey square
      random  uniform noise, unoptimised
      anchor  the reference decal, UNOPTIMISED

    The third one is new here and it is the control the realism claim needs.
    Grey and noise answer "does a square on the roof do this by itself".
    Anchor answers the harder question: does a plausible printed decal do this
    by itself, before any optimisation? If a plain cargo marking already
    suppresses the vehicle, the attack is not what is doing the work.
    """
    res = res or N.PATCH_RES
    if kind == "grey":
        return torch.full((3, res, res), 0.5, device=device)
    if kind == "random":
        g = torch.Generator(device="cpu").manual_seed(N.SEED)
        return torch.rand(3, res, res, generator=g).to(device)
    if kind == "anchor":
        return DEC.anchor_image(size=res, device=device)[0]
    raise SystemExit("Unknown baseline '%s'. Use grey, random or anchor." % kind)


# ------------------------------------------------------------------- metrics

def iou(pred, gt):
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    return inter / union if union > 0 else 0.0


def window_of(bbox, shape, expand=2.0):
    """A window expand times the vehicle box, clipped to the frame. Scoring in
    this window rather than the whole frame stops a 1280x720 sea of correct
    background from drowning a 60x40 vehicle in every IoU."""
    H, W = shape
    x0, y0, x1, y1 = bbox
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w, h = (x1 - x0) * expand, (y1 - y0) * expand
    return (max(int(cx - w / 2), 0), max(int(cy - h / 2), 0),
            min(int(cx + w / 2), W), min(int(cy + h / 2), H))


def propagation_radii(flip_mask, centre_xy, gsd):
    """
    Ground radii in metres containing 50 and 90 percent of the away-flips.

    Metres, not pixels, and that is the point: a 20 m frame and a 60 m frame
    have GSDs that differ threefold, so a radius in pixels is not comparable
    between them and a radius in metres is.
    """
    ys, xs = np.nonzero(flip_mask)
    if len(xs) == 0:
        return float("nan"), float("nan"), 0
    d = np.sqrt((xs - centre_xy[0]) ** 2 + (ys - centre_xy[1]) ** 2) * gsd
    d.sort()
    n = len(d)
    return (float(d[min(int(0.50 * n), n - 1)]),
            float(d[min(int(0.90 * n), n - 1)]), n)


# ---------------------------------------------------------------- evaluation

def evaluate(patch_tag, model_keys=(), det_keys=(), split="holdout",
             baseline=None, stride=1, max_items=0, device=None, size_m=None,
             size_mode=None, prompts=None, printed=False):
    device = device or N.DEVICE
    D.seed_everything()
    size_mode = size_mode or size_mode_of(patch_tag)

    if baseline:
        patch = baseline_patch(baseline, device)
        # THE CONTROL MUST BE THE SIZE OF THE PATCH IT CONTROLS FOR.
        # This read PATCH_SIZE_M, which is wrong and wrong silently: a control
        # for a 2.0 m patch was being pasted at the config's 1.2 m, so the
        # subtraction "attack minus square on a roof" compared a 2.0 m attack
        # against a 1.2 m square and credited the attack with the extra area.
        # The size comes from the patch's own run record, exactly as it does
        # for the patch itself.
        size_m = size_m or patch_size_of(patch_tag)
        out_tag = "%s__%s" % (patch_tag, baseline)
    else:
        patch = load_patch(patch_tag, device)
        size_m = size_m or patch_size_of(patch_tag)
        out_tag = patch_tag
    if printed:
        # Score what the printer makes of the patch (measured ink chart),
        # rather than the requested colours.
        from physdecal.core import printchain as PC
        patch = PC.PrinterLUT(device=device)(patch.unsqueeze(0))[0]
        out_tag += "__printed"
    print("patch '%s' at %s%s"
          % (out_tag,
             "per-vehicle roof fill" if size_mode == "roof"
             else "%.2f m" % size_m,
             "  (baseline)" if baseline else ""))

    index = D.build_index(split, nadir_only=False)
    if stride > 1:
        index = index[::stride]
    if max_items:
        index = index[:max_items]
    loader = D.make_loader(index, 1, shuffle=False, workers=2)
    os.makedirs(N.RESULT_DIR, exist_ok=True)

    for key in model_keys:
        if N.MODELS.get(key, (None,))[0] == "hf_clipseg":
            # THE OPEN-VOCABULARY ARM IS SCORED ONCE PER PROMPT.
            # Its "class" is a sentence, so re-wording the prompt is a
            # generalisation test no closed-set model can be given: the patch
            # was optimised against one phrasing and is re-scored against
            # others with nothing retrained. Results are tagged with the
            # prompt so the rows never merge.
            for prompt in (prompts or N.CLIPSEG_PROMPTS):
                _eval_seg(out_tag, key, loader, patch, size_m, device,
                          size_mode, prompt=prompt)
        else:
            _eval_seg(out_tag, key, loader, patch, size_m, device, size_mode)
    for key in det_keys:
        _eval_det(out_tag, key, loader, patch, size_m, device, size_mode)


def _size_cols(meta, size_m):
    """
    The decal's footprint on THIS instance.

    coverage_frac is the column that makes results comparable across vehicle
    classes: a 2.0 m decal is 49 percent of a pickup and 18 percent of a box
    truck, and it is the fraction, not the metres, that predicts whether the
    attack works.
    """
    return {"patch_size_m": round(float(size_m), 4),
            "coverage_frac": round(P.coverage_fraction(meta, size_m), 4),
            "roof_max_m": round(P.max_patch_size_m(meta), 4),
            "fits_roof": int(size_m <= P.max_patch_size_m(meta) + 1e-6)}


def _base_row(meta):
    return {
        "frame_id": meta["frame_id"], "target": meta["target"],
        "cls": meta["cls"], "theta_deg": round(meta["theta_deg"], 2),
        "altitude_m": round(meta["altitude_m"], 2),
        "lighting": meta["lighting"],
        "gsd_m_per_px": round(meta["gsd_m_per_px"], 6),
        "visible_px": meta["visible_px"],
    }


@torch.no_grad()
def _eval_seg(tag, key, loader, patch, size_m, device, size_mode=None,
              prompt=None):
    from physdecal.victims import segmentation as M

    model = M.load_model(key, device=device)
    out_key = key
    if prompt is not None:
        if not hasattr(model, "set_prompt"):
            raise SystemExit("%s takes no prompt; only the open-vocabulary "
                             "arm does." % key)
        model.set_prompt(prompt)
        # The prompt is part of the victim's identity, so it goes in the
        # filename. A slug, because a prompt is a sentence and a sentence is
        # not a filename.
        slug = "".join(c if c.isalnum() else "_" for c in prompt).strip("_")
        out_key = "%s__%s" % (key, slug)
        print("    prompt: %r" % prompt)
    rows = []
    for b in loader:
        img = b["image"].to(device)
        gt = b["gt"].to(device)
        inst = b["inst"].to(device)
        meta = b["meta"][0]

        # EVALUATION COMPOSITES WITHOUT EOT AND WITHOUT THE PRINT CHAIN.
        # The chain is a distribution the patch is optimised OVER, not a
        # transform applied at test time; sampling it here would add variance
        # that has nothing to do with the victim and make two runs of the same
        # evaluation disagree.
        adv, alpha, sizes = P.composite(img, b["quad"], b["meta"], patch,
                                        inst, train=False, device=device,
                                        size_m=size_m, chain=None,
                                        size_mode=size_mode)

        p_clean = model(img)["p_veh"]
        p_adv = model(adv)["p_veh"]

        x0, y0, x1, y1 = window_of(meta["bbox"], img.shape[-2:])
        g = gt[0, 0, y0:y1, x0:x1].cpu().numpy() > 0.5
        pc = p_clean[0, 0, y0:y1, x0:x1].cpu().numpy() > 0.5
        pa = p_adv[0, 0, y0:y1, x0:x1].cpu().numpy() > 0.5
        al = alpha[0, 0, y0:y1, x0:x1].cpu().numpy() > 0.5

        iou_c, iou_a = iou(pc, g), iou(pa, g)
        flipped = g & pc & (~pa)                 # was right, now wrong
        under = flipped & al
        away = flipped & (~al)

        ys, xs = np.nonzero(al)
        centre = (xs.mean(), ys.mean()) if len(xs) else \
                 (0.5 * (g.shape[1]), 0.5 * (g.shape[0]))
        r50, r90, n_away = propagation_radii(away, centre, meta["gsd_m_per_px"])

        attackable = iou_c >= N.ATTACKABLE_IOU
        row = _base_row(meta)
        row["victim"] = key
        row["prompt"] = prompt or ""
        row.update(_size_cols(meta, sizes[0]))
        row.update({
            "clean_iou": round(iou_c, 4), "adv_iou": round(iou_a, 4),
            "iou_drop": round(iou_c - iou_a, 4),
            "attackable": int(attackable),
            "success": int(attackable and iou_a < N.ASR_IOU_THRESHOLD),
            "flip_under": int(under.sum()), "flip_away": int(n_away),
            "patch_px": int(al.sum()),
            "r50_m": round(r50, 3) if r50 == r50 else "",
            "r90_m": round(r90, 3) if r90 == r90 else "",
        })
        rows.append(row)

    path = os.path.join(N.RESULT_DIR, "seg_%s__%s.csv" % (tag, out_key))
    _write(path, rows)
    _summarise(rows, "seg", out_key, path)


@torch.no_grad()
def _eval_det(tag, key, loader, patch, size_m, device, size_mode=None):
    from physdecal.victims import detection as DET

    model = DET.load_detector(key, device=device)
    rows = []
    for b in loader:
        img = b["image"].to(device)
        inst = b["inst"].to(device)
        meta = b["meta"][0]

        adv, _, sizes = P.composite(img, b["quad"], b["meta"], patch, inst,
                                    train=False, device=device, size_m=size_m,
                                    chain=None, size_mode=size_mode)

        ids = model.vehicle_ids(meta["cls"])
        pc = model.predict(img)[0]
        pa = model.predict(adv)[0]
        gt_box = meta["bbox"]

        hit_c = DET.best_hit(pc, gt_box, ids)
        hit_a = DET.best_hit(pa, gt_box, ids)
        s_c = DET.max_vehicle_score(pc, gt_box, ids)
        s_a = DET.max_vehicle_score(pa, gt_box, ids)

        attackable = hit_c is not None
        row = _base_row(meta)
        row.update(_size_cols(meta, sizes[0]))
        row.update({
            "clean_detected": int(attackable),
            "adv_detected": int(hit_a is not None),
            "clean_score": round(s_c, 4), "adv_score": round(s_a, 4),
            "score_drop": round(s_c - s_a, 4),
            "clean_iou_box": round(hit_c[1], 4) if hit_c else 0.0,
            "adv_iou_box": round(hit_a[1], 4) if hit_a else 0.0,
            "attackable": int(attackable),
            "success": int(attackable and hit_a is None),
        })
        rows.append(row)

    path = os.path.join(N.RESULT_DIR, "det_%s__%s.csv" % (tag, key))
    _write(path, rows)
    _summarise(rows, "det", key, path)


def _write(path, rows):
    if not rows:
        print("  nothing to write for %s" % path)
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _summarise(rows, arm, key, path):
    n = len(rows)
    att = [r for r in rows if r["attackable"]]
    if not att:
        print("  %-6s %-20s  attackable set EMPTY of %d instances. ASR is "
              "UNDEFINED, not zero: nothing here was visible to the victim "
              "without the patch. Finetune this arm before reporting it."
              % (arm, key, n))
        return
    asr = sum(r["success"] for r in att) / len(att)
    extra = ""
    if arm == "seg":
        extra = "  mean IoU drop %.3f" % np.mean([r["iou_drop"] for r in att])
    else:
        extra = "  mean score drop %.3f" % np.mean([r["score_drop"] for r in att])
    print("  %-6s %-20s  attackable %d/%d (%.0f%%)  ASR %.3f%s"
          % (arm, key, len(att), n, 100.0 * len(att) / n, asr, extra))
    print("         -> %s" % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch", required=True, help="a tag under out/patches")
    ap.add_argument("--models", nargs="*", default=None,
                    help="segmentation victims, or 'all'")
    ap.add_argument("--detectors", nargs="*", default=None,
                    help="detection victims, or 'all'")
    ap.add_argument("--split", default="holdout", choices=["holdout", "train", "all"])
    ap.add_argument("--baseline", default=None,
                    choices=["grey", "random", "anchor"])
    ap.add_argument("--stride", type=int, default=1,
                    help="score every Nth instance; the grid is dense and a "
                         "stride of 4 changes no conclusion at a quarter the "
                         "time. State it in the paper if you use it.")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--prompts", nargs="*", default=None,
                    help="prompts for the open-vocabulary arm. Defaults to "
                         "CLIPSEG_PROMPTS; clipseg is scored once per prompt.")
    ap.add_argument("--size-mode", default=None, choices=["fixed", "roof"],
                    help="override the mode recorded in the patch's run.json. "
                         "Only for deliberate cross-protocol comparisons; the "
                         "default reads the run record.")
    ap.add_argument("--printed", action="store_true",
                    help="pass the patch (or baseline) through the measured "
                         "printer colour response first; results get the "
                         "suffix __printed")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    mk = (list(N.EVAL_MODELS) if a.models == ["all"] else (a.models or []))
    dk = (list(N.DET_EVAL_MODELS) if a.detectors == ["all"] else (a.detectors or []))
    if not mk and not dk:
        mk = [N.ATTACK_MODELS[0]]

    D.start_log("evaluate")
    evaluate(a.patch, mk, dk, a.split, a.baseline, a.stride, a.max_items,
             a.device, size_mode=a.size_mode, prompts=a.prompts,
             printed=a.printed)


if __name__ == "__main__":
    main()
