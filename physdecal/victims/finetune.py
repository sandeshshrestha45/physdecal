r"""
physdecal finetune
Decoder-only finetuning, so the victims can actually see a vehicle from
directly overhead.

    physdecal finetune --model segformer_b0
    physdecal finetune --model all
    physdecal finetune --model upernet_swin_t --epochs 10 --batch 2

WHY THIS STEP EXISTS
--------------------
Measured on this dataset with physdecal check --clean, every published checkpoint
in the roster scores 0.000 clean vehicle IoU at nadir, under the full-frame
protocol AND under ROI crops. An overhead car is returned as "wall" and a
truck cab as "windowpane". ADE20K and PASCAL VOC contain essentially no
straight-down vehicle imagery, so this is out of distribution rather than a
bug: the capture is correct, the masks are correct, the class indices are
correct, and the models simply cannot do it.

That blocks the whole experiment. ASR is defined only over instances the model
segmented correctly WITHOUT the patch, so at nadir the attackable set is empty
and ASR is undefined, not low. The patch is optimised at nadir, so this is
precisely the cell that has to work.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------
Trains the DECODE HEAD only. The backbone is frozen and never receives a
gradient. That is not a shortcut, it is the experimental control: the
independent variable is the backbone's attention scope, and a trained backbone
would mean the ConvNeXt-against-Swin comparison was partly measuring this
training run instead of the published architectures. Every arm gets the same
treatment, the same schedule and the same data.

The head becomes two classes, background and vehicle, because the attack only
ever consumes p_veh. That also removes the class-name lookup, so there is no
way left to attack "wall" by accident. It does mean ATTACK_OBJECTIVE =
"targeted" is unavailable on finetuned weights, since there is no road class
to push toward. The default objective is "vanish".

NO NEW DATA IS NEEDED. The capture already carries pixel-accurate per-vehicle
instance masks in segmentation/ keyed by palette.json.

THE SPLIT, WHICH IS THE PART THAT IS EASY TO GET WRONG
------------------------------------------------------
Finetuning happens on the TRAIN split only, and validation is carved out of
that same train split. The holdout vehicles are never trained on and never
validated on, or every attack number downstream is leaked.

About a tenth of the rows inside train frames are holdout vehicles standing in
the background. Those pixels are marked IGNORE and excluded from the loss.
Calling them background would teach the model that a Pickup is not a vehicle,
which would corrupt the holdout evaluation in the direction that flatters the
attack. Calling them vehicle would leak held-out appearance into training.
Ignoring them is the only honest option.

ALL VIEWPOINTS, NOT ONLY NADIR
------------------------------
Training uses every viewpoint in the train split, not just the nadir frames.
The experiment measures how a NADIR-OPTIMISED PATCH generalises across
viewpoint, so the victim has to be uniformly competent across that grid. A
victim finetuned only at nadir would decay with angle by itself, and that
decay would be indistinguishable from the patch failing to generalise, which
is the one thing the paper is trying to measure.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from physdecal import config as N
from physdecal.core import data as D
from physdecal.victims import segmentation as M


def split_targets():
    """(train vehicles, holdout vehicles) by primary target, from the CSVs."""
    out = {}
    for name, path in (("train", N.SPLIT_TRAIN), ("holdout", N.SPLIT_HOLDOUT)):
        if not os.path.isfile(path):
            raise SystemExit(
                "No %s. Run make_splits.py from the GeoPatchCity capture pipeline first." % path)
        with open(path, newline="", encoding="utf-8") as f:
            out[name] = sorted({r["target"] for r in csv.DictReader(f)
                                if r["is_primary"] == "1"})
    return out["train"], out["holdout"]


class FrameSegDataset(Dataset):
    """
    One item is one FRAME, not one instance, because this is semantic
    segmentation and a frame with three vehicles is one training example.

    target is HxW int64 with three values:
        0                background
        1                a TRAIN vehicle
        IGNORE_INDEX     a HOLDOUT vehicle, excluded from the loss
    """

    def __init__(self, frame_ids, train_colours, holdout_colours, size_hw):
        self.frame_ids = frame_ids
        self.train_colours = train_colours
        self.holdout_colours = holdout_colours
        self.size_hw = size_hw

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, i):
        fid = self.frame_ids[i]
        bgr = cv2.imread(os.path.join(N.IMAGES_DIR, fid + ".png"),
                         cv2.IMREAD_COLOR)
        seg = cv2.imread(os.path.join(N.SEG_DIR, fid + ".png"),
                         cv2.IMREAD_COLOR)
        if bgr is None or seg is None:
            raise RuntimeError("missing frame %s" % fid)

        tgt = np.zeros(seg.shape[:2], dtype=np.int64)
        for col in self.holdout_colours:
            tgt[np.all(seg == np.array(col, np.uint8), axis=-1)] = N.IGNORE_INDEX
        for col in self.train_colours:
            tgt[np.all(seg == np.array(col, np.uint8), axis=-1)] = 1

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = self.size_hw
        # The image is resampled bilinearly and the target with NEAREST, so a
        # label is never invented between two classes.
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        tgt = cv2.resize(tgt.astype(np.int32), (w, h),
                         interpolation=cv2.INTER_NEAREST).astype(np.int64)
        return (torch.from_numpy(rgb).permute(2, 0, 1),
                torch.from_numpy(tgt))


def model_grid(model):
    """The HxW the victim actually consumes, so the loss is computed there."""
    probe = torch.zeros(1, 3, N.IMAGE_HEIGHT, N.IMAGE_WIDTH)
    return tuple(model.resize_to_model(probe).shape[-2:])


def forward_logits(model, x01, out_hw):
    """Raw 2-class logits at out_hw. Bypasses SegVictim.forward, which
    reduces to p_veh and is the inference path, not the training one."""
    xm = F.interpolate(x01, size=out_hw, mode="bilinear", align_corners=False)
    xn = (xm - model.mean) / model.std
    logits, _ = model._raw_logits(xn, False)
    if logits.shape[-2:] != out_hw:
        logits = F.interpolate(logits, size=out_hw, mode="bilinear",
                               align_corners=False)
    return logits


@torch.no_grad()
def evaluate(model, loader, dev, grid):
    """Vehicle-class IoU over the validation frames, ignore label excluded."""
    model.eval()
    inter = union = 0.0
    for img, tgt in loader:
        img, tgt = img.to(dev), tgt.to(dev)
        with torch.amp.autocast("cuda", enabled=(N.AMP and dev == "cuda")):
            logits = forward_logits(model, img, grid)
        pred = logits.argmax(1)
        valid = tgt != N.IGNORE_INDEX
        p = (pred == 1) & valid
        g = (tgt == 1) & valid
        inter += float((p & g).sum())
        union += float((p | g).sum())
    return inter / union if union > 0 else 0.0


def run(key, epochs, batch, lr, max_frames):
    N.ensure_dirs()
    os.makedirs(N.FINETUNE_DIR, exist_ok=True)
    D.start_log("finetune_%s" % key)
    D.seed_everything()
    dev = N.DEVICE

    if N.MODELS[key][0] == "hf_clipseg":
        raise SystemExit(
            "clipseg is the open-vocabulary arm. Its 'class' is a sentence, so "
            "replacing\nits head with a fixed two-class one would delete the "
            "only reason it is in\nthe roster. Leave it on published weights "
            "and report it separately.")

    train_names, holdout_names = split_targets()
    palette = {k: tuple(int(v) for v in c) for k, c in
               json.load(open(N.PALETTE_JSON))["colours_bgr"].items()}
    train_colours = [palette[t] for t in train_names if t in palette]
    holdout_colours = [palette[t] for t in holdout_names if t in palette]
    print("train vehicles   : %s" % ", ".join(train_names))
    print("holdout vehicles : %s   (pixels ignored, never background)"
          % ", ".join(holdout_names))

    # Frames come from the TRAIN split index, every viewpoint, deduplicated.
    index = D.build_index("train", nadir_only=False)
    frames = sorted({it["frame_id"] for it in index})
    rng = np.random.RandomState(N.SEED)
    rng.shuffle(frames)
    if max_frames:
        frames = frames[:max_frames]
    n_val = max(1, int(len(frames) * N.FINETUNE_VAL_FRACTION))
    val_ids, train_ids = frames[:n_val], frames[n_val:]
    print("frames: %d train, %d validation (validation is carved out of the "
          "TRAIN split,\n        never from the holdout)"
          % (len(train_ids), len(val_ids)))

    model = M.load_model(key, dev, cache=False, finetuned=False)
    M.make_binary(model)
    model = model.to(dev)
    grid = model_grid(model)
    print("loss computed on the %dx%d model grid" % grid)

    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for mod in M.decoder_modules(model):
        for p in mod.parameters():
            p.requires_grad_(True)
            params.append(p)
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print("training %s decoder parameters of %s total (%.1f%%). Backbone "
          "frozen." % (f"{n_train:,}", f"{n_total:,}",
                       100.0 * n_train / max(n_total, 1)))

    mk = lambda ids, sh: torch.utils.data.DataLoader(
        FrameSegDataset(ids, train_colours, holdout_colours, grid),
        batch_size=batch, shuffle=sh, num_workers=N.FINETUNE_WORKERS,
        pin_memory=True, drop_last=sh, persistent_workers=True)
    tl, vl = mk(train_ids, True), mk(val_ids, False)

    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs * len(tl), 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(N.AMP and dev == "cuda"))
    # Vehicles are well under one percent of the pixels, so an unweighted
    # cross entropy is minimised by predicting background everywhere. The
    # weight is what stops that trivial solution.
    weight = torch.tensor([1.0, 20.0], device=dev)

    hist = os.path.join(N.FINETUNE_DIR, "%s_history.csv" % key)
    hf = open(hist, "w", newline="")
    hw = csv.writer(hf)
    hw.writerow(["epoch", "train_loss", "val_vehicle_iou", "sec"])

    best = -1.0
    t0 = time.time()
    base_iou = evaluate(model, vl, dev, grid)
    print("\nvehicle IoU before any training: %.4f  (a fresh 2-class head is "
          "random,\n  so this is the floor, not the published model's score)"
          % base_iou)

    for ep in range(epochs):
        model.train()
        # The frozen backbone stays in eval mode so its BatchNorm running
        # statistics are not silently rewritten by our batches.
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        tot = nb = 0
        for img, tgt in tl:
            img, tgt = img.to(dev, non_blocking=True), tgt.to(dev,
                                                              non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(N.AMP and dev == "cuda")):
                logits = forward_logits(model, img, grid)
                loss = F.cross_entropy(logits, tgt, weight=weight,
                                       ignore_index=N.IGNORE_INDEX)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss)
            nb += 1
            if nb % 50 == 0:
                print("  epoch %d  batch %4d/%d  loss %.4f  %.0fs"
                      % (ep, nb, len(tl), tot / nb, time.time() - t0))

        iou = evaluate(model, vl, dev, grid)
        hw.writerow([ep, round(tot / max(nb, 1), 5), round(iou, 5),
                     round(time.time() - t0, 1)])
        hf.flush()
        print("epoch %d/%d  train loss %.4f  val vehicle IoU %.4f  %s"
              % (ep + 1, epochs, tot / max(nb, 1), iou,
                 "  <- best" if iou > best else ""))
        if iou > best:
            best = iou
            torch.save({
                "decoder": {k: v.cpu() for k, v in model.state_dict().items()
                            if any(k.startswith(p) for p in _decoder_prefixes(model))},
                "val_iou": iou, "epoch": ep, "key": key,
                "labels": M.BINARY_LABELS,
                "train_vehicles": train_names,
                "holdout_vehicles": holdout_names,
                "frames_train": len(train_ids), "frames_val": len(val_ids),
                "epochs": epochs, "lr": lr, "batch": batch,
                "input_grid": list(grid), "seed": N.SEED,
                "backbone": "frozen",
            }, M.finetuned_path(key))
    hf.close()

    print("\nbest val vehicle IoU %.4f  ->  %s" % (best, M.finetuned_path(key)))
    if best < N.CLEAN_IOU_GATE:
        print("STILL UNDER THE %.2f GATE. More epochs may help, but consider "
              "that a frozen\nbackbone which never saw a nadir vehicle may not "
              "carry features a decoder can\nread. The next lever is an "
              "aerial-pretrained backbone, not more epochs."
              % N.CLEAN_IOU_GATE)
    else:
        print("Above the %.2f gate. Set USE_FINETUNED = True in config.py, "
              "then re-run\n  physdecal check --clean --models %s\n"
              "Every number from these weights is labelled weights=finetuned."
              % (N.CLEAN_IOU_GATE, key))


def _decoder_prefixes(model):
    """state_dict key prefixes belonging to the decoder, for a small save."""
    names = []
    for mod in M.decoder_modules(model):
        for n, m in model.named_modules():
            if m is mod:
                names.append(n + ".")
    return names or [""]



# ===========================================================================
#                         THE DETECTION ARM
# ===========================================================================
#
# Identical protocol to the segmentation arm above, and it has to be identical
# or the two columns of the results table are not comparable:
#
#   backbone FROZEN, head trained          same as decoder-only
#   train split only, val carved from it   same split, same vehicles
#   every viewpoint, not only nadir        same reason: a victim finetuned at
#                                          nadir decays with angle by itself,
#                                          and that decay is indistinguishable
#                                          from the patch failing to transfer
#
# The one place the two arms genuinely differ is what an ignored instance
# means. Segmentation can mark a holdout vehicle's pixels IGNORE_INDEX and
# drop them from the loss. A detector has no such label: a box is present or
# it is not, and an unlabelled vehicle is trained as background, which teaches
# the detector that a Pickup is not a vehicle and corrupts the holdout
# evaluation in the direction that flatters the attack.
#
# So frames containing a holdout vehicle are DROPPED from detector training
# entirely rather than half-labelled. It costs frames and it is the only
# honest option available. The count is printed, because a reader should know
# how much data that cost.


class FrameDetDataset(Dataset):
    """
    One item is one FRAME with all of its train-vehicle boxes.

    Returns (image, target) in the torchvision detection convention:
        image   3 x H x W float in [0,1]
        target  dict(boxes = M x 4 xyxy float, labels = M int64 all ones)
    """

    def __init__(self, rows_by_frame, size_hw):
        self.frames = sorted(rows_by_frame)
        self.rows = rows_by_frame
        self.size_hw = size_hw

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        fid = self.frames[i]
        bgr = cv2.imread(os.path.join(N.IMAGES_DIR, fid + ".png"),
                         cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("missing frame %s" % fid)
        H0, W0 = bgr.shape[:2]
        h, w = self.size_hw
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        sx, sy = w / float(W0), h / float(H0)
        boxes = []
        for r in self.rows[fid]:
            x0, y0, x1, y1 = r
            boxes.append([x0 * sx, y0 * sy, x1 * sx, y1 * sy])
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        # A degenerate box makes the detection loss NaN rather than raising,
        # so they are dropped here where the cause is still visible.
        keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
        boxes = boxes[keep]

        return (torch.from_numpy(rgb).permute(2, 0, 1),
                {"boxes": torch.from_numpy(boxes),
                 "labels": torch.ones(len(boxes), dtype=torch.int64)})


def det_collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def det_rows(split_csv, holdout_targets):
    """
    Frame -> list of boxes, dropping any frame that contains a holdout vehicle.

    Returns (rows_by_frame, n_dropped_frames).
    """
    rows, dropped = {}, set()
    with open(split_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("label_status") == "ignored":
                continue
            fid = r["frame_id"]
            if r["target"] in holdout_targets:
                dropped.add(fid)
                continue
            if int(r.get("visible_px") or 0) < N.MIN_INSTANCE_PX:
                continue
            rows.setdefault(fid, []).append(
                (float(r["bbox_x0"]), float(r["bbox_y0"]),
                 float(r["bbox_x1"]), float(r["bbox_y1"])))
    for fid in dropped:
        rows.pop(fid, None)
    return rows, len(dropped)


@torch.no_grad()
def det_evaluate(model, loader, dev, score=0.05):
    """
    Recall at DET_IOU_MATCH, and AP50.

    AP50 is computed the plain way -- sort predictions by score, greedily match
    to ground truth at IoU 0.5, integrate precision over 101 recall points --
    rather than pulling in pycocotools for one number. There is one class here,
    so the COCO machinery would add a dependency and change nothing.
    """
    from physdecal.victims import detection as DET

    model.net.eval()
    per_frame_recall, all_pred = [], []
    n_gt = 0
    for imgs, tgts in loader:
        imgs = [im.to(dev) for im in imgs]
        preds = model.net(imgs)
        for pred, tgt in zip(preds, tgts):
            gtb = tgt["boxes"].numpy()
            n_gt += len(gtb)
            pb = pred["boxes"].cpu().numpy()
            ps = pred["scores"].cpu().numpy()
            keep = ps >= score
            pb, ps = pb[keep], ps[keep]
            taken = np.zeros(len(gtb), bool)
            for j in np.argsort(-ps):
                if len(gtb) == 0:
                    all_pred.append((float(ps[j]), 0))
                    continue
                ious = DET.box_iou(pb[j], gtb)
                k = int(np.argmax(ious))
                hit = ious[k] >= N.DET_IOU_MATCH and not taken[k]
                if hit:
                    taken[k] = True
                all_pred.append((float(ps[j]), 1 if hit else 0))
            if len(gtb):
                per_frame_recall.append(taken.mean())

    if not all_pred or n_gt == 0:
        return {"recall": 0.0, "ap50": 0.0, "n_gt": n_gt}
    all_pred.sort(key=lambda t: -t[0])
    tp = np.cumsum([p[1] for p in all_pred])
    fp = np.cumsum([1 - p[1] for p in all_pred])
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / max(n_gt, 1)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        m = prec[rec >= t]
        ap += (m.max() if len(m) else 0.0) / 101.0
    return {"recall": float(np.mean(per_frame_recall)) if per_frame_recall else 0.0,
            "ap50": float(ap), "n_gt": n_gt}


def run_det(key, epochs, batch, lr, max_frames):
    from physdecal.victims import detection as DET

    dev = N.DEVICE
    D.seed_everything()
    os.makedirs(N.FINETUNE_DIR, exist_ok=True)

    train_t, holdout_t = split_targets()
    rows, n_dropped = det_rows(N.SPLIT_TRAIN, set(holdout_t))
    frames = sorted(rows)
    if max_frames:
        frames = frames[:max_frames]
        rows = {f: rows[f] for f in frames}
    if not frames:
        raise SystemExit("No trainable frames left for the detector.")

    rng = np.random.RandomState(N.SEED)
    idx = rng.permutation(len(frames))
    n_val = max(1, int(len(frames) * N.FINETUNE_VAL_FRACTION))
    val_f = [frames[i] for i in idx[:n_val]]
    trn_f = [frames[i] for i in idx[n_val:]]

    long_side = N.DET_INPUT_LONG_SIDE
    size_hw = (int(round(N.IMAGE_HEIGHT * long_side / float(N.IMAGE_WIDTH))),
               long_side)
    print("  %d frames, %d train / %d val; %d frames dropped for containing a "
          "holdout vehicle" % (len(frames), len(trn_f), len(val_f), n_dropped))
    print("  grid %dx%d" % (size_hw[1], size_hw[0]))

    def mk(fl, sh):
        return torch.utils.data.DataLoader(
            FrameDetDataset({f: rows[f] for f in fl}, size_hw),
            batch_size=batch, shuffle=sh, num_workers=N.FINETUNE_WORKERS,
            collate_fn=det_collate, drop_last=sh, pin_memory=True)

    trn, val = mk(trn_f, True), mk(val_f, False)

    model = DET.load_detector(key, device=dev, finetuned=False, cache=False)
    DET._binary_head(model.net, key)
    model.finetuned = True
    model = model.to(dev)

    for p in model.net.parameters():
        p.requires_grad_(False)
    head_params = []
    for mod in DET.detector_head_modules(model.net, key):
        for p in mod.parameters():
            p.requires_grad_(True)
            head_params.append(p)
    n_train = sum(p.numel() for p in head_params)
    n_all = sum(p.numel() for p in model.net.parameters())
    print("  trainable %d of %d parameters (%.1f%%), backbone frozen"
          % (n_train, n_all, 100.0 * n_train / max(n_all, 1)))

    opt = torch.optim.Adam(head_params, lr=lr)
    hist, best = [], -1.0
    for ep in range(epochs):
        model.net.train()
        t0, tot, n = time.time(), 0.0, 0
        for imgs, tgts in trn:
            imgs = [im.to(dev) for im in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            if sum(len(t["boxes"]) for t in tgts) == 0:
                continue
            losses = model.net(imgs, tgts)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 10.0)
            opt.step()
            tot += float(loss)
            n += 1
        m = det_evaluate(model, val, dev)
        row = {"epoch": ep, "train_loss": tot / max(n, 1),
               "val_recall": m["recall"], "val_ap50": m["ap50"],
               "seconds": round(time.time() - t0, 1)}
        hist.append(row)
        print("  epoch %d  loss %.4f  val recall %.3f  AP50 %.3f  [%.0fs]"
              % (ep, row["train_loss"], m["recall"], m["ap50"], row["seconds"]))

        if m["ap50"] > best:
            best = m["ap50"]
            head_state = {k: v.cpu() for k, v in model.net.state_dict().items()
                          if k.startswith(("head.", "roi_heads."))}
            torch.save({"head": head_state, "epoch": ep, "val_ap50": best,
                        "val_recall": m["recall"], "key": key},
                       DET.finetuned_path(key))

    with open(os.path.join(N.FINETUNE_DIR, "det_%s_history.csv" % key), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    print("  best val AP50 %.3f -> %s" % (best, DET.finetuned_path(key)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="*", default=None,
                    help="one or more keys from MODELS, or 'all'. 'all' skips "
                         "clipseg, whose class is a sentence.")
    # Defaults are resolved after we know which arm is being trained, because
    # the two arms have different sensible defaults and argparse cannot know
    # which one --det selected.
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=N.FINETUNE_MAX_FRAMES)
    ap.add_argument("--det", nargs="*", default=None,
                    help="one or more keys from DET_MODELS, or 'all'. "
                         "Finetunes the DETECTION head instead of a decoder.")
    args = ap.parse_args()

    if args.det:
        keys = (list(N.DET_MODELS) if "all" in args.det else args.det)
        unknown = [k for k in keys if k not in N.DET_MODELS]
        if unknown:
            raise SystemExit("Not in DET_MODELS: %s" % ", ".join(unknown))
        for k in keys:
            print("\n" + "=" * 70)
            print("finetuning detector %s" % k)
            print("=" * 70)
            run_det(k, args.epochs or N.DET_FINETUNE_EPOCHS,
                    args.batch or N.DET_FINETUNE_BATCH,
                    args.lr or N.DET_FINETUNE_LR, args.max_frames)
        return

    asked = args.model or [N.ATTACK_MODELS[0]]
    if "all" in asked:
        keys = [k for k in N.MODELS if N.MODELS[k][0] != "hf_clipseg"]
    else:
        keys = asked
    unknown = [k for k in keys if k not in N.MODELS]
    if unknown:
        raise SystemExit("Not in MODELS: %s" % ", ".join(unknown))
    for k in keys:
        print("\n" + "=" * 70)
        print("finetuning %s" % k)
        print("=" * 70)
        run(k, args.epochs or N.FINETUNE_EPOCHS,
            args.batch or N.FINETUNE_BATCH,
            args.lr or N.FINETUNE_LR, args.max_frames)


if __name__ == "__main__":
    main()
