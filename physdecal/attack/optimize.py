r"""
physdecal optimize
Optimises a printable decal that suppresses the vehicle, at nadir, on the
roof.

    physdecal optimize --model segformer_b0 --task seg
    physdecal optimize --model segformer_b0 --task joint --det retinanet
    physdecal optimize --model segformer_b0 --realism 0 --tag baseline
    physdecal optimize --model segformer_b0 --sweep-realism

THE LOOP, AND THE ONE LINE THAT MATTERS
---------------------------------------
This is projected gradient descent, not penalised gradient descent:

    raw            <- Adam step on the free variable
    patch          <- manifold.forward(texture(raw))     <-- this line
    printed        <- print_chain(patch)                 sampled per draw
    frame          <- composite(frame, printed, roof homography)
    loss           <- attack(frame) + realism(patch)

manifold.forward is a straight-through projection. The forward value is a
patch that is band limited, flat, made of real inks and laid out on a real
design; the backward value is the identity, so the free variable keeps a
gradient everywhere. The consequence is that EVERY iterate is printable, not
just the last one, and the realism of the result is a property of the search
space rather than an outcome the weights had to win.

With LEVEL4_PARAM = "ink_logits" the level-4 texture is
decal.InkRegionTexture instead, which is on the manifold by construction,
and the manifold.forward line is skipped.

WHAT IS OPTIMISED, AND WHAT IS HELD OUT
---------------------------------------
Optimisation sees nadir frames from the TRAIN split only. Every oblique frame
and every holdout vehicle is untouched here and is a genuine generalisation
test in physdecal evaluate. Holding the split is the whole basis for the transfer
claim, so the loader is built from N.SPLIT_TRAIN with nadir_only=True and
there is no flag to change it.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from physdecal import config as N
from physdecal.core import data as D
from physdecal.core import patch as P
from physdecal.core import decal as DEC
from physdecal.core import losses as L
from physdecal.core import printchain as PC


# Parametrisations that are on the manifold by construction, so they skip
# manifold.forward and manifold.project.
INK_TEXTURES = (DEC.InkRegionTexture, DEC.InkPixelTexture)


def _run_dir(tag):
    return os.path.join(N.PATCH_DIR, tag)


def save_run(tag, texture, manifold, meta, history):
    """
    Write the patch and everything needed to score it later.

    patch.pt holds the PROJECTED patch, which is the one that gets printed and
    the one an evaluation must use. The raw parameter is saved beside it only
    so a run can be resumed; nothing downstream should ever read it, because
    the raw variable is outside the feasible set by construction.
    """
    d = _run_dir(tag)
    os.makedirs(d, exist_ok=True)
    with torch.no_grad():
        if isinstance(texture, INK_TEXTURES):
            patch = texture().clamp(0, 1)      # already on the manifold
        else:
            patch = manifold.project(texture().unsqueeze(0))[0].clamp(0, 1)

    torch.save({"patch": patch.cpu(), "raw": texture.raw.detach().cpu(),
                "meta": meta}, os.path.join(d, "patch.pt"))

    import cv2
    img = (patch.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(d, "patch.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # An 8x nearest-neighbour copy, because a 256 px PNG in a paper figure is
    # resampled by the viewer and the ink boundaries stop looking crisp.
    big = cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(d, "patch_large.png"),
                cv2.cvtColor(big, cv2.COLOR_RGB2BGR))

    metrics = DEC.realism_metrics(patch.unsqueeze(0), manifold.anchor,
                                  manifold.inks, meta.get("patch_size_m"))
    meta = dict(meta)
    meta["realism_metrics"] = metrics
    meta["manifold"] = manifold.describe()
    meta["degrees_of_freedom"] = manifold.degrees_of_freedom()
    with open(os.path.join(d, "run.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if history:
        import csv
        with open(os.path.join(d, "history.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
    return d, metrics


def build_victims(model_key, det_key, task, device):
    """Load only what the task needs. Loading both on a 6 GB card when only
    one is used is how a run dies at step 900 instead of step 0."""
    seg = det = None
    if task in ("seg", "joint"):
        from physdecal.victims import segmentation as M
        seg = M.load_model(model_key, device=device)
    if task in ("det", "joint"):
        from physdecal.victims import detection as DET
        det = DET.load_detector(det_key, device=device)
        if det.density != "dense":
            raise SystemExit(
                "Detector '%s' is not dense, so it cannot be the white-box "
                "arm. Use retinanet or fcos." % det_key)
    return seg, det


def optimise(model_key="segformer_b0", det_key="retinanet", task=None,
             realism=None, design=None, steps=None, batch=None, lr=None,
             size_m=None, tag=None, print_chain=None, max_items=0,
             device=None, log_every=25, supercell=None, size_mode=None, seed=None,
             inks_csv=None, ink_count=None, seg_objective=None,
             det_objective="hinge",
             level4_param=None, palette_only=False, printer_lut=False,
             epochs=None):
    device = device or N.DEVICE
    task = task or N.ATTACK_TASK
    steps = steps or N.ATTACK_STEPS
    batch = batch or N.ATTACK_BATCH
    size_mode = size_mode or N.PATCH_SIZE_MODE
    seg_objective = seg_objective or N.SEG_OBJECTIVE
    # In mode "roof", size_m is an upper CAP and None means uncapped, so it
    # must NOT be defaulted to PATCH_SIZE_M -- doing that caps every vehicle at
    # the fixed-mode size and quietly undoes the whole point of the mode.
    if size_mode == "fixed":
        size_m = size_m or N.PATCH_SIZE_M
    realism = N.REALISM_LEVEL if realism is None else realism
    level4_param = level4_param or N.LEVEL4_PARAM
    # At level 3 the ink logits are per pixel (InkPixelTexture).
    ink_logits = realism >= 3 and level4_param == "ink_logits"
    lr = lr or (N.INK_LOGIT_LR if ink_logits else N.ATTACK_LR)
    design = design or N.ANCHOR_DESIGN
    tag = tag or "%s_%s_r%d" % (model_key if task != "det" else det_key,
                                task, realism)

    D.seed_everything(seed)
    index = D.build_index("train", nadir_only=True)
    if not index:
        raise SystemExit(
            "The nadir train index is empty. Either NADIR_MAX_THETA_DEG is too "
            "tight or make_splits.py has not run.")
    loader = D.make_loader(index, batch, shuffle=True, workers=2,
                           max_items=max_items)
    # A step budget wins when one is given; otherwise the run is a whole
    # number of passes over the data. The loop below restarts the loader when
    # it runs out, so epochs * len(loader) steps is exactly that many passes.
    if steps:
        epochs = steps / float(len(loader))
    else:
        epochs = epochs or N.ATTACK_EPOCHS
        steps = int(epochs * len(loader))

    seg, det = build_victims(model_key, det_key, task, device)

    if size_mode == "roof":
        # The band limit is a function of the decal's metric size, which under
        # mode "roof" differs per vehicle. The manifold is one object shared by
        # every instance, so it is designed at the MEDIAN size over the
        # optimisation set. Designing at the smallest would over-blur the decal
        # on large roofs; at the largest it would leave texture the camera
        # cannot resolve on small ones.
        band_size = float(np.median([P.size_for(it, None, "roof")
                                     for it in index]))
        print("size mode 'roof': decal spans %.2f to %.2f m over the "
              "optimisation set, band designed at the median %.2f m"
              % (min(P.size_for(it, None, "roof") for it in index),
                 max(P.size_for(it, None, "roof") for it in index), band_size))
    else:
        band_size = size_m
    # An explicit ink set is only for the H1 palette experiments; the
    # default is the measured gamut read from PRINTABLE_COLOURS_CSV.
    inks = (DEC.load_inks(path=inks_csv, device=device, k=ink_count)
            if inks_csv else None)
    manifold = DEC.DecalManifold(level=realism, design=design, device=device,
                                 patch_size_m=band_size, supercell=supercell,
                                 inks=inks, palette_only=palette_only)
    chain = PC.PrintChain(enabled=print_chain, device=device)
    # The victim sees what the printer makes of the patch, not the patch.
    lut = PC.PrinterLUT(device=device) if printer_lut else None
    objective = L.Objective(manifold, task=task, device=device,
                            seg_objective=seg_objective,
                            det_objective=det_objective)

    if ink_logits and realism >= 4:
        texture = DEC.InkRegionTexture(manifold).to(device)
    elif ink_logits:
        texture = DEC.InkPixelTexture(manifold).to(device)
    else:
        texture = P.PatchTexture(init="grey").to(device)
    opt = torch.optim.Adam(texture.parameters(), lr=lr)

    print("\n" + "=" * 72)
    print("tag        : %s" % tag)
    print("task       : %s   seg=%s det=%s   seg objective=%s"
          % (task, model_key if seg else "-", det_key if det else "-",
             seg_objective))
    print("patch      : %s, %d px, %s"
          % (("per-vehicle roof fill%s"
              % ("" if not size_m else ", capped at %.2f m" % size_m))
             if size_mode == "roof" else "%.2f m" % size_m,
             N.PATCH_RES, manifold.describe()))
    print("free vars  : %d of %d raw pixels"
          % (manifold.degrees_of_freedom(), N.PATCH_RES ** 2 * 3))
    print("channel    : %s" % chain.describe())
    print("data       : %d nadir train instances, batch %d, %d steps "
          "(%.1f epochs of %d steps)"
          % (len(index), batch, steps, epochs, len(loader)))
    print("=" * 72 + "\n")

    history = []
    it = iter(loader)
    t0 = time.time()
    best = None

    for step in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)

        img = b["image"].to(device, non_blocking=True)
        gt = b["gt"].to(device, non_blocking=True)
        inst = b["inst"].to(device, non_blocking=True)
        quad = b["quad"]

        if isinstance(texture, INK_TEXTURES):
            patch = texture()
        else:
            patch = manifold.forward(texture().unsqueeze(0))[0]
        shown = lut(patch.unsqueeze(0))[0] if lut is not None else patch
        adv, alpha, _ = P.composite(img, quad, b["meta"], shown, inst,
                                    train=True, device=device, size_m=size_m,
                                    chain=chain, size_mode=size_mode)

        m_veh = p_veh = dense = boxes = None
        if seg is not None:
            out = seg(adv)
            m_veh, p_veh = out["m_veh"], out["p_veh"]
        if det is not None:
            dense = det.dense_vehicle_map(adv)

        loss, parts = objective(patch.unsqueeze(0), m_veh=m_veh, gt_mask=gt,
                                dense_maps=dense, boxes=boxes, alpha=alpha,
                                p_veh=p_veh)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        parts["step"] = step
        parts["seconds"] = round(time.time() - t0, 1)
        history.append(parts)

        if best is None or parts["total"] < best:
            best = parts["total"]

        if step % log_every == 0 or step == steps - 1:
            msg = "  ".join("%s %.4f" % (k, parts[k]) for k in
                            ("total", "seg", "det", "realism_total")
                            if k in parts)
            print("step %5d/%d  %s  [%.0fs]"
                  % (step, steps, msg, time.time() - t0))

    meta = {
        "tag": tag, "task": task, "seg_model": model_key if seg else None,
        "det_model": det_key if det else None, "realism_level": realism,
        "seg_objective": seg_objective if seg else None,
        "det_objective": det_objective if det else None,
        "anchor_design": design if realism >= 4 else None,
        "patch_size_m": size_m, "patch_size_mode": size_mode,
        "band_design_size_m": band_size,
        "patch_res": N.PATCH_RES, "steps": steps,
        "epochs": round(epochs, 2), "steps_per_epoch": len(loader),
        "batch": batch, "lr": lr, "print_chain": chain.enabled,
        "parametrisation": type(texture).__name__,
        "band_gsd_m_per_px": N.BAND_GSD_M_PER_PX,
        "supercell": manifold.supercell,
        "inks_csv": inks_csv, "ink_count": ink_count,
        "palette_only": palette_only, "printer_lut": printer_lut,
        "weights": {"ink": N.W_INK, "flat": N.W_FLAT, "anchor": N.W_ANCHOR,
                    "gamut": N.W_GAMUT, "nps": N.W_NPS,
                    "det": N.JOINT_DET_WEIGHT},
        "n_train_instances": len(index),
        "final_loss": history[-1]["total"] if history else None,
        "seconds": round(time.time() - t0, 1),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    d, metrics = save_run(tag, texture, manifold, meta, history)
    print("\nwrote %s" % d)
    print("realism: " + "  ".join("%s=%s" % (k, round(v, 3) if isinstance(v, float) else v)
                                  for k, v in metrics.items()))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="segformer_b0",
                    help="segmentation victim (white box)")
    ap.add_argument("--det", default="retinanet",
                    help="detection victim (white box, dense families only)")
    ap.add_argument("--task", default=None, choices=["seg", "det", "joint"])
    ap.add_argument("--realism", type=int, default=None,
                    help="0..4, see config.py REALISM_LEVEL")
    ap.add_argument("--design", default=None, help="anchor design name")
    ap.add_argument("--supercell", type=int, default=None,
                    help="anchor region subdivision at level 4. Higher means "
                         "more flat areas and more attacker degrees of "
                         "freedom, at some cost in how unmistakably the decal "
                         "reads as the reference design.")
    ap.add_argument("--epochs", type=float, default=None,
                    help="passes over the nadir train set (default "
                         "ATTACK_EPOCHS = %d)" % N.ATTACK_EPOCHS)
    ap.add_argument("--steps", type=int, default=None,
                    help="fixed step budget; overrides --epochs")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--size", type=float, default=None, help="patch metres")
    ap.add_argument("--size-mode", default=None, choices=["fixed", "roof"],
                    help="'roof' sizes the decal per vehicle to the largest "
                         "square its own roof panel admits, so no instance "
                         "carries a decal that could not be applied")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-print-chain", action="store_true",
                    help="ablation: digital EOT only")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--sweep-realism", action="store_true",
                    help="run every level in REALISM_SWEEP, for the RQ4 curve")
    ap.add_argument("--inks", default=None,
                    help="CSV of sRGB triples to use as the ink set, for "
                         "the H1 palette experiments. Default: the "
                         "measured gamut.")
    ap.add_argument("--ink-count", type=int, default=None,
                    help="colours to reduce the ink CSV to. Omit to use "
                         "every row of it.")
    ap.add_argument("--seg-objective", default=None,
                    choices=["vanish_prob", "vanish", "vanish_away",
                             "untargeted", "ce_targeted", "ce_untargeted"],
                    help="which pixels the segmentation hinge covers, see "
                         "config.py SEG_OBJECTIVE")
    ap.add_argument("--det-objective", default="hinge",
                    choices=["hinge", "ce_targeted", "ce_untargeted"],
                    help="detection attack term, see losses.det_attack_loss")
    ap.add_argument("--level4-param", default=None,
                    choices=["rgb_snap", "ink_logits"],
                    help="level-4 parametrisation, see config.py LEVEL4_PARAM. "
                         "At level 3, ink_logits is one logit per pixel and ink")
    ap.add_argument("--printer-lut", action="store_true",
                    help="pass the patch through the measured printer colour "
                         "response (ink chart, printchain.PrinterLUT) "
                         "before the victim sees it")
    ap.add_argument("--palette-only", action="store_true",
                    help="level 3 without the band and flat steps: the patch "
                         "is only constrained to the measured inks")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config.SEED, for variance runs")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    D.start_log("optimise")
    common = dict(model_key=a.model, det_key=a.det, task=a.task,
                  design=a.design, steps=a.steps, epochs=a.epochs,
                  batch=a.batch, lr=a.lr,
                  size_m=a.size, max_items=a.max_items, device=a.device,
                  supercell=a.supercell, size_mode=a.size_mode,
                  print_chain=False if a.no_print_chain else None,
                  seed=a.seed, inks_csv=a.inks, ink_count=a.ink_count,
                  seg_objective=a.seg_objective,
                  det_objective=a.det_objective,
                  level4_param=a.level4_param, palette_only=a.palette_only,
                  printer_lut=a.printer_lut)

    if a.sweep_realism:
        for lvl in N.REALISM_SWEEP:
            optimise(realism=lvl,
                     tag=a.tag and ("%s_r%d" % (a.tag, lvl)), **common)
    else:
        optimise(realism=a.realism, tag=a.tag, **common)


if __name__ == "__main__":
    main()
