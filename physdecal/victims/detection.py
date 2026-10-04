"""
detection.py
The detection arm: one interface over three torchvision detectors, exposing
both what an attack needs (dense, differentiable, per-location vehicle scores)
and what an evaluation needs (boxes and scores after NMS).

WHY THIS IS A FIRST-CLASS ARM HERE
----------------------------------
paper2 scored detectors but never attacked them, and the patch it scored had
been optimised entirely against a segmentation posterior. That answers "does a
segmentation patch happen to also break a detector", which is a fine question
but not the one a reader of a physical-attack paper asks. The question they
ask is whether the printed thing on the roof defeats the perception stack an
aerial pipeline actually runs, and aerial pipelines run detectors.

So here the detector is optimised against directly (`physdecal optimize --task det`),
jointly with segmentation (--task joint), and scored either way.

THE SAME GATE APPLIES, AND IT WILL FIRE
---------------------------------------
COCO detectors have the identical nadir domain gap the COCO/ADE20K
segmentation models had: a car from 60 m straight down is not in COCO. Clean
recall at nadir is near zero on published weights, so the attackable set is
empty and vanish rate is UNDEFINED, not low. physdecal finetune therefore
finetunes the detection head the same way it finetunes a decode head, with the
backbone frozen, and every arm gets the same treatment. Published-weight
numbers are reported too, as the measurement that justifies the finetuning.

TWO DETECTOR FAMILIES, AND ONLY ONE IS WHITE BOX
------------------------------------------------
retinanet and fcos are dense: every feature cell carries a class logit, so the
attack has a gradient at every location the vehicle covers. fasterrcnn scores
sampled proposals, so its gradient reaches only the handful of boxes the RPN
happened to propose. Calling that white box would be generous, so it is a
transfer target only and the paper says so.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from physdecal import config as N


# COCO 2017 class ids as torchvision emits them.
COCO_VEHICLES = {2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus",
                 7: "train", 8: "truck"}
# Narrow on purpose: the dataset's classes mapped onto COCO's. A box truck
# scored against COCO "truck" is already a generous reading, and widening this
# further would be scoring the attack against a class the vehicle is not.
DATASET_TO_COCO = {
    "car": [3],
    "truck": [8, 6],
    "bus": [6],
    "van": [3, 8],
}

BINARY_VEHICLE_ID = 1          # after make_binary_det: 0 background, 1 vehicle


def _binary_head(model, family):
    """Replace the classification head with a two-class one."""
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
    from torchvision.models.detection.fcos import FCOSClassificationHead
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if family == "retinanet":
        old = model.head.classification_head
        new = RetinaNetClassificationHead(
            in_channels=256, num_anchors=old.num_anchors, num_classes=2,
            norm_layer=None)
        model.head.classification_head = new
    elif family == "fcos":
        old = model.head.classification_head
        new = FCOSClassificationHead(
            in_channels=256, num_anchors=old.num_anchors, num_classes=2,
            norm_layer=None)
        model.head.classification_head = new
    elif family == "fasterrcnn":
        inf = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(inf, 2)
    else:
        raise SystemExit("no binary head rule for family '%s'" % family)
    return model


def detector_head_modules(model, family):
    """The part finetuning is allowed to move. Everything else is frozen."""
    if family in ("retinanet", "fcos"):
        return [model.head]
    if family == "fasterrcnn":
        return [model.roi_heads]
    raise SystemExit("no head rule for family '%s'" % family)


def finetuned_path(key):
    import os
    if N.DET_MODELS.get(key, (None,))[0] == "ultralytics":
        from physdecal.victims import yolo
        return yolo.WEIGHTS
    return os.path.join(N.FINETUNE_DIR, "det_%s_head.pt" % key)


class DetVictim(torch.nn.Module):
    """
    Wrapper over a torchvision detector.

    Two entry points and they are deliberately different:

      predict()      the ordinary inference path, through the model's own
                     postprocessing, returning boxes and scores after NMS.
                     Used for every number in a results table.

      dense_scores() the attack path. Returns the per-location vehicle
                     probability map for each FPN level, BEFORE NMS, with the
                     image-space centre of every location, so the attack can
                     hinge exactly the locations the vehicle covers.

    NMS is the reason these cannot be the same function. It is an index
    selection, so a gradient does reach the boxes that survive it, but only
    those: once the attack has pushed a box below the NMS threshold the
    gradient to it vanishes and the optimiser stops improving a location it
    has only just barely beaten. Attacking the dense map instead keeps every
    location in the objective for the whole run.
    """

    def __init__(self, key, arch, family, finetuned):
        super().__init__()
        self.key = key
        self.family = family
        self.finetuned = finetuned
        fn = getattr(torchvision.models.detection, arch)
        self.net = fn(weights="COCO_V1")
        if finetuned:
            _binary_head(self.net, family)
        self.net.eval()

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, images):
        """
        images  B x 3 x H x W in [0,1]
        returns list of B dicts with boxes (M x 4), scores (M,), labels (M,)
        """
        self.net.eval()
        return self.net([im for im in images])

    def vehicle_ids(self, dataset_class="car"):
        if self.finetuned:
            return [BINARY_VEHICLE_ID]
        return DATASET_TO_COCO.get(dataset_class, [3])

    # --------------------------------------------------------------- attack
    def dense_scores(self, images):
        """
        images  B x 3 x H x W in [0,1], WITH grad

        returns list over FPN levels of
            (probs, centres)
            probs    B x A x K x h x w   sigmoid class scores
            centres  2 x h x w           location centres in ORIGINAL image px

        Only for the dense families. fasterrcnn raises, by design: see the
        module docstring.
        """
        if self.family == "fasterrcnn":
            raise SystemExit(
                "fasterrcnn has no dense class map, so it cannot be a white "
                "box arm. Attack 'retinanet' or 'fcos' and score fasterrcnn "
                "as transfer.")

        H0, W0 = images.shape[-2:]
        imgs, _ = self.net.transform([im for im in images])
        feats = self.net.backbone(imgs.tensors)
        head = self.net.head.classification_head

        Ht, Wt = imgs.tensors.shape[-2:]
        out = []
        for f in feats.values():
            logits = head.cls_logits(head.conv(f))          # B x A*K x h x w
            B, AK, h, w = logits.shape
            K = head.num_classes
            A = AK // K
            probs = torch.sigmoid(logits.view(B, A, K, h, w))

            # Location centres, mapped back to ORIGINAL image pixels. The
            # model's transform resized the frame, so a centre computed on the
            # feature grid is in transformed pixels and must be divided back,
            # or every hinge lands on the wrong part of the vehicle.
            sy, sx = Ht / h, Wt / w
            ys = (torch.arange(h, device=logits.device, dtype=logits.dtype)
                  + 0.5) * sy * (H0 / Ht)
            xs = (torch.arange(w, device=logits.device, dtype=logits.dtype)
                  + 0.5) * sx * (W0 / Wt)
            centres = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), 0)
            out.append((probs, centres))
        return out

    def dense_vehicle_map(self, images, dataset_class="car"):
        """
        The vehicle probability at every location of every level, reduced over
        anchors and classes, each resampled to the original frame grid.

        Returned as a list of B x 1 x H x W maps, one per FPN level. Keeping
        the levels separate rather than summing them is deliberate: a vehicle
        at 20 m and one at 60 m are found at different levels, and collapsing
        them would let a strong response at one scale mask a weak one at
        another.
        """
        H0, W0 = images.shape[-2:]
        ids = self.vehicle_ids(dataset_class)
        maps = []
        for probs, _ in self.dense_scores(images):
            veh = probs[:, :, ids].amax(dim=2).amax(dim=1, keepdim=True)
            maps.append(F.interpolate(veh, size=(H0, W0), mode="bilinear",
                                      align_corners=False))
        return maps


_CACHE = {}


def load_detector(key, device=None, finetuned=None, cache=True):
    import os

    if key not in N.DET_MODELS:
        raise SystemExit("Unknown detector '%s'. Known: %s"
                         % (key, list(N.DET_MODELS)))
    finetuned = N.USE_FINETUNED if finetuned is None else finetuned
    ck = (key, bool(finetuned))
    if cache and ck in _CACHE:
        return _CACHE[ck]

    fam0, arch, scope = N.DET_MODELS[key]
    if fam0 == "ultralytics":
        # A different framework entirely, so it gets its own wrapper rather
        # than a branch inside DetVictim. It satisfies the same predict()
        # contract, which is all evaluate.py consumes.
        from physdecal.victims import yolo
        print("loading detector %-12s %-28s density=%s" % (key, arch, scope))
        m = yolo.YoloVictim(device=device)
        m.density = scope
        if cache:
            _CACHE[ck] = m
        return m

    family = key
    print("loading detector %-12s %-28s density=%s" % (key, arch, scope))
    m = DetVictim(key, arch, family, finetuned)

    if finetuned:
        path = finetuned_path(key)
        if not os.path.isfile(path):
            raise SystemExit(
                "No %s.\nUSE_FINETUNED is on but this detector has no "
                "finetuned head. Run\n  physdecal finetune --det %s"
                % (path, key))
        state = torch.load(path, map_location="cpu")
        missing = m.net.load_state_dict(state["head"], strict=False)
        print("    finetuned head loaded, val AP50 %.3f at epoch %d"
              % (state.get("val_ap50", float("nan")), state.get("epoch", -1)))
        del missing

    m = m.to(device or N.DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.density = scope
    if cache:
        _CACHE[ck] = m
    return m


def unload_all():
    _CACHE.clear()
    torch.cuda.empty_cache()


# ------------------------------------------------------------------- scoring

def box_iou(a, b):
    """IoU of one box against an M x 4 array. Both xyxy."""
    if len(b) == 0:
        return np.zeros(0, np.float32)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    x0 = np.maximum(a[0], b[:, 0])
    y0 = np.maximum(a[1], b[:, 1])
    x1 = np.minimum(a[2], b[:, 2])
    y1 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    area_b = np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-6, None)
    return inter / (area_a + area_b - inter)


def best_hit(pred, gt_box, wanted_ids, score_thresh=None, iou_thresh=None):
    """
    The best detection of this vehicle, or None.

    Returns (score, iou, label) of the highest-scoring prediction of a wanted
    class whose box overlaps the ground truth at iou_thresh or better.
    """
    score_thresh = N.DET_SCORE if score_thresh is None else score_thresh
    iou_thresh = N.DET_IOU_MATCH if iou_thresh is None else iou_thresh

    boxes = pred["boxes"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    labels = pred["labels"].detach().cpu().numpy()

    keep = np.isin(labels, list(wanted_ids)) & (scores >= score_thresh)
    if not keep.any():
        return None
    ious = box_iou(np.asarray(gt_box, np.float32), boxes[keep])
    ok = ious >= iou_thresh
    if not ok.any():
        return None
    s = scores[keep][ok]
    i = int(np.argmax(s))
    return float(s[i]), float(ious[ok][i]), int(labels[keep][ok][i])


def max_vehicle_score(pred, gt_box, wanted_ids, iou_thresh=None):
    """
    Highest vehicle-class score overlapping the ground truth at ANY score.

    Distinct from best_hit and both are reported. best_hit answers "was it
    detected", which is a threshold decision and therefore a cliff: a box at
    0.49 and a box at 0.01 are both misses and the table cannot tell them
    apart. This answers "how close did it come", which is the continuous
    quantity, and it is what makes a detection failure figure legible.
    """
    iou_thresh = N.DET_IOU_MATCH if iou_thresh is None else iou_thresh
    boxes = pred["boxes"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    labels = pred["labels"].detach().cpu().numpy()
    keep = np.isin(labels, list(wanted_ids))
    if not keep.any():
        return 0.0
    ious = box_iou(np.asarray(gt_box, np.float32), boxes[keep])
    ok = ious >= iou_thresh
    return float(scores[keep][ok].max()) if ok.any() else 0.0
