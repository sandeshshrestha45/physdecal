r"""
physdecal gradcam
Where the victim's evidence for "vehicle" comes from, and what the decal does
to it.

    physdecal gradcam --patches L_dof task_joint_ink --victim segformer_b0
    physdecal gradcam --patches L_dof task_joint_ink --victim retinanet --arm det

WHY A CAM AND NOT JUST THE PREDICTION
-------------------------------------
The segmentation panels already show that a prediction changes. They cannot
show WHY, and the two candidate whys make different predictions about how the
attack will behave off-distribution:

  the decal SUPPRESSES vehicle evidence under itself, leaving the rest of the
      vehicle still supporting the class. Then coverage is what matters and
      the attack scales with the fraction of the roof it occupies.

  the decal CREATES a competing region that pulls the model's attention off
      the vehicle entirely. Then the attack is about salience, not coverage,
      and a small decal in the right place would do as well as a large one.

Finding 8 argues the first from coverage numbers alone. A CAM is the direct
evidence, and it is cheap: one backward pass per image.

THE TARGET LAYER IS CHOSEN, NOT HARD CODED
------------------------------------------
Grad-CAM needs the deepest feature map that still has spatial extent. Three
model families are in this study -- torchvision, HuggingFace segformer/upernet,
and CLIPSeg -- and their module trees share no naming convention, so a dotted
path per family is four things to keep in sync and to get silently wrong when
a checkpoint changes. Instead the tree is run once and the last 4D activation
with spatial extent above a floor is taken. The chosen module's name is
recorded on the figure, because a CAM from the wrong layer is not obviously
wrong to look at.
"""

import argparse
import contextlib
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from physdecal import config as N


@contextlib.contextmanager
def _params_need_grad(model):
    """
    Temporarily mark the victim's parameters as requiring grad.

    The victims are frozen for evaluation, and making the input the graph
    source is normally enough. It is not enough for CLIPSeg: transformers runs
    its frozen CLIP backbone under no_grad, so the decoder receives a detached
    input and, with its own parameters frozen too, nothing downstream of the
    image requires grad and there is no graph to attach a CAM to. Flipping the
    parameters back on for the duration of one backward restores the graph.
    Nothing is stepped and the flags are restored, so the model is unchanged.
    """
    saved = [(q, q.requires_grad) for q in model.net.parameters()]
    try:
        for q, _ in saved:
            q.requires_grad_(True)
        yield
    finally:
        for q, flag in saved:
            q.requires_grad_(flag)


def _as_map(t):
    """
    A hooked activation as B x C x H x W, reshaping transformer tokens.

    A ViT carries its spatial state as a B x N x C token sequence, so a CAM
    over it needs the tokens laid back on the grid they came from. N is the
    grid plus however many prefix tokens the architecture prepends (CLIPSeg's
    CLS), so the prefix is dropped by taking the largest square that fits and
    reading the tokens from the END, which is where the patch tokens sit.
    Returns None when the tensor is neither a map nor a square-able sequence.
    """
    if t.dim() == 4:
        return t
    if t.dim() != 3:
        return None
    b, n, c = t.shape
    side = int(math.isqrt(n))
    while side > 1 and side * side > n:
        side -= 1
    if side < 2:
        return None
    keep = side * side
    return t[:, n - keep:, :].transpose(1, 2).reshape(b, c, side, side)


MIN_CAM_HW = 6          # below this a CAM is a few blocks and reads as noise
MIN_CAM_CH = 32         # below this the layer is a prediction head, not features

# Grad-CAM weights each channel by how much the score depends on it, so it
# needs a feature map with channels to weigh. The deepest 4D activation in a
# segmentation model is the class logit stack -- two channels here -- and a CAM
# over it is the prediction redrawn in orange, which looks convincing and shows
# nothing the prediction panel did not. The channel floor is what keeps the
# hook on the last real feature map instead.


def pick_target_layer(model, x01, min_hw=MIN_CAM_HW, min_ch=MIN_CAM_CH):
    """
    The deepest leaf module whose output is a feature map, found by running it.

    Returns (name, module). Records are kept of every 4D output so the choice
    can be reported rather than assumed.
    """
    seen = []
    hooks = []

    def make(name, mod):
        def hook(_m, _i, o):
            t = o[0] if isinstance(o, (tuple, list)) and len(o) else o
            # requires_grad is the test that matters. A model can hold
            # 4D activations that never reach the vehicle posterior --
            # upernet's auxiliary head, CLIPSeg's text branch -- and picking
            # one of those yields either a None gradient or a tensor that
            # cannot even retain one. Probing with grad enabled makes the
            # graph itself answer which layers are eligible.
            if not (torch.is_tensor(t) and t.requires_grad):
                return
            m = _as_map(t)
            if (m is not None and min(m.shape[-2:]) >= min_hw
                    and m.shape[1] >= min_ch):
                seen.append((name, mod, tuple(m.shape)))
        return hook

    for name, mod in model.net.named_modules():
        if len(list(mod.children())):          # leaves only
            continue
        hooks.append(mod.register_forward_hook(make(name, mod)))
    try:
        was = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        with _params_need_grad(model):
            model(x01.detach().clone().requires_grad_(True))
    finally:
        torch.set_grad_enabled(was)
        for h in hooks:
            h.remove()
    if not seen:
        raise SystemExit(
            "No 4D activation with spatial extent >= %d and >= %d channels "
            "found in this model. Grad-CAM has nothing to attach to."
            % (min_hw, min_ch))
    return seen


def gradcam(model, x01, layer=None, roi=None):
    """
    Grad-CAM for the vehicle channel, as a 0..1 map at the input's resolution.

    The scalar that is differentiated is the summed vehicle posterior over the
    region of interest, which is the quantity the attack is trying to move. A
    mean would give the same map up to the normalisation that follows.
    """
    if layer is None:
        layer = pick_target_layer(model, x01)[-1][1]

    acts, grads = {}, {}

    def fwd(_m, _i, o):
        t = o[0] if isinstance(o, (tuple, list)) and len(o) else o
        acts["a"] = t
        t.retain_grad()

    h = layer.register_forward_hook(fwd)
    try:
        was = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        # The victims are loaded frozen for evaluation, so no parameter
        # requires grad and nothing downstream of them does either. The input
        # is therefore made the source of the graph: that is enough for the
        # activations to be differentiable, and it leaves the weights alone.
        x = x01.detach().clone().requires_grad_(True)
        with _params_need_grad(model):
            p = model(x)["p_veh"]
            score = (p * roi).sum() if roi is not None else p.sum()
            model.net.zero_grad(set_to_none=True)
            score.backward()
        a = acts["a"]
        grads["g"] = a.grad
    finally:
        h.remove()
        torch.set_grad_enabled(was)

    a, g = acts["a"].detach(), grads["g"]
    if g is None:
        raise SystemExit(
            "The chosen layer received no gradient. It sits outside the path "
            "from input to the vehicle posterior; pass --layer explicitly.")
    a, g = _as_map(a), _as_map(g)
    w = g.mean(dim=(2, 3), keepdim=True)             # channel importance
    cam = F.relu((w * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x01.shape[-2:], mode="bilinear",
                        align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max()).detach().cpu().numpy() if float(cam.max()) > 0 \
        else cam.detach().cpu().numpy()


def gradcam_det(model, x01, layer=None, roi=None):
    """
    Grad-CAM for a dense detector's vehicle response.

    The differentiated scalar is the summed per-location vehicle probability
    over the region of interest, taken across FPN levels. Only the dense
    families expose that map: fasterrcnn scores sampled proposals and raises
    by design, and yolo is wrapped behind a predictor, so neither has a
    gradient path to attach to and neither gets a CAM here.
    """
    if layer is None:
        layer = pick_target_layer_det(model, x01)[-1][1]
    acts = {}

    def fwd(_m, _i, o):
        t = o[0] if isinstance(o, (tuple, list)) and len(o) else o
        acts["a"] = t
        t.retain_grad()

    h = layer.register_forward_hook(fwd)
    try:
        was = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        x = x01.detach().clone().requires_grad_(True)
        with _params_need_grad(model):
            maps = model.dense_vehicle_map(x)
            tot = sum((m * roi).sum() if roi is not None else m.sum()
                      for m in maps)
            model.net.zero_grad(set_to_none=True)
            tot.backward()
        g = acts["a"].grad
    finally:
        h.remove()
        torch.set_grad_enabled(was)

    a = acts["a"].detach()
    if g is None:
        raise SystemExit("The chosen layer received no gradient.")
    a, g = _as_map(a), _as_map(g)
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x01.shape[-2:], mode="bilinear",
                        align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max()).detach().cpu().numpy() if float(cam.max()) > 0         else cam.detach().cpu().numpy()


def pick_target_layer_det(model, x01, min_hw=MIN_CAM_HW, min_ch=MIN_CAM_CH):
    """As pick_target_layer, but driven through the dense vehicle map so only
    modules on the detector's classification path are considered."""
    seen, hooks = [], []

    def make(name, mod):
        def hook(_m, _i, o):
            t = o[0] if isinstance(o, (tuple, list)) and len(o) else o
            # requires_grad is the test that matters. A model can hold
            # 4D activations that never reach the vehicle posterior --
            # upernet's auxiliary head, CLIPSeg's text branch -- and picking
            # one of those yields either a None gradient or a tensor that
            # cannot even retain one. Probing with grad enabled makes the
            # graph itself answer which layers are eligible.
            if not (torch.is_tensor(t) and t.requires_grad):
                return
            m = _as_map(t)
            if (m is not None and min(m.shape[-2:]) >= min_hw
                    and m.shape[1] >= min_ch):
                seen.append((name, mod, tuple(m.shape)))
        return hook

    for name, mod in model.net.named_modules():
        if len(list(mod.children())):
            continue
        hooks.append(mod.register_forward_hook(make(name, mod)))
    try:
        was = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        with _params_need_grad(model):
            model.dense_vehicle_map(x01.detach().clone().requires_grad_(True))
    finally:
        torch.set_grad_enabled(was)
        for h in hooks:
            h.remove()
    if not seen:
        raise SystemExit("No usable feature map found for Grad-CAM.")
    # Two things make the plain deepest-last rule wrong for a detector.
    # Its head is SHARED across pyramid levels, so one module fires once per
    # level and the last firing is the COARSEST -- for RetinaNet a 6x11 map,
    # which upsamples to a CAM with no visible structure. Ranking purely by
    # area instead reaches the stem, which has resolution but no semantics.
    # So: keep the FPN and head modules, which are the deep ones, then take
    # the finest level among them. Sorting is stable, so within one level the
    # deepest module still comes last.
    deep = [t for t in seen if "fpn" in t[0] or "head" in t[0]]
    if deep:
        seen = deep
    seen.sort(key=lambda t: t[2][-1] * t[2][-2])
    return seen


def resolve_layer(model, x01, roi=None, det=False):
    """
    The deepest candidate layer that actually receives gradient, found by
    trying them.

    requires_grad is necessary and not sufficient: upernet's auxiliary head and
    CLIPSeg's conditional branch both descend from the input, so their
    activations require grad, but the vehicle posterior does not depend on
    them and a backward pass leaves their .grad as None. Only running the
    backward answers it, so candidates are tried deepest first and the first
    that returns a gradient is kept for the rest of the figure.
    """
    cands = (pick_target_layer_det if det else pick_target_layer)(model, x01)
    fn = gradcam_det if det else gradcam
    errs = []
    for name, mod, shape in reversed(cands):
        try:
            fn(model, x01, mod, roi)
            return name, mod, shape
        except (SystemExit, RuntimeError) as e:
            errs.append("%s: %s" % (name, str(e).split("\n")[0][:70]))
    raise SystemExit(
        "No candidate layer in this model receives gradient from the vehicle "
        "posterior. Tried %d: %s" % (len(cands), "; ".join(errs[:6])))


def _norm_crop(a):
    """
    Renormalise a CAM crop to its own range.

    gradcam normalises over the whole frame, so a border artefact far from the
    vehicle sets the scale and the region actually shown comes back flat. The
    fraction of CAM mass under the decal is a ratio and is computed on the
    full-frame map, so it is unaffected by this rescaling.
    """
    a = a - a.min()
    m = float(a.max())
    return a / m if m > 0 else a


def overlay_cam(rgb, cam, alpha=0.5):
    """CAM on the image, through a perceptually ordered map.

    jet is avoided deliberately: its bands invent structure a reader takes for
    a boundary in the evidence, which is the one thing a CAM figure must not
    do. inferno is monotonic in lightness, so a region that looks hotter is
    hotter.
    """
    import matplotlib
    heat = matplotlib.colormaps["inferno"](np.clip(cam, 0, 1))[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)


def _cam_panel_det(tags, victim, n=None, select="success", frames=None,
                   device=None, out=None):
    """The detector counterpart, over the dense vehicle map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from physdecal.core import data as D
    from physdecal.victims import detection as DET
    from physdecal.evaluation import evaluate as E
    from physdecal.figures import panels as FIG
    from physdecal.core import patch as P

    device = device or N.DEVICE
    n = n or N.PANEL_FRAMES
    rows, n_pool, note = FIG.pick_instances(tags[0], victim, "det", n, select,
                                            frames)
    index = D.build_index("holdout", nadir_only=False)
    model = DET.load_detector(victim, device=device)
    if N.DET_MODELS[victim][2] != "dense":
        raise SystemExit(
            "%s is not a dense detector, so it exposes no differentiable "
            "per-location vehicle map and Grad-CAM has no gradient path. "
            "Only %s do."
            % (victim, ", ".join(k for k, v in N.DET_MODELS.items()
                                 if v[2] == "dense")))
    loaded = [(t, E.load_patch(t, device), E.patch_size_of(t),
               E.size_mode_of(t)) for t in tags]
    cols = 1 + len(loaded)
    fig, axes = plt.subplots(2 * len(rows), cols,
                             figsize=(3.05 * cols, 2.15 * 2 * len(rows)),
                             squeeze=False)
    chosen = None

    for r, rec in enumerate(rows):
        item = FIG.instance_for(index, rec["frame_id"], rec["target"])
        b, meta = FIG.load_one(item, device)
        img, inst = b["image"], b["inst"]
        roi = b["gt"]
        if chosen is None:
            cname, layer, shape = resolve_layer(model, img, roi, det=True)
            chosen = "%s %s" % (cname, list(shape))
        x0, y0, x1, y1 = FIG.crop_of(meta, img.shape[-2:], expand=3.0)
        cut3 = lambda t: t[0].permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]
        top, bot = axes[2 * r], axes[2 * r + 1]

        cam_c = gradcam_det(model, img, layer, roi)
        FIG.tile(top[0], cut3(img), "Clean" if r == 0 else "",
                 "%s, %.0f m" % (rec["target"], float(rec["altitude_m"])))
        FIG.tile(bot[0], overlay_cam(cut3(img), _norm_crop(cam_c[y0:y1, x0:x1])), "",
                 "vehicle CAM")

        for c, (tag, patch, size_m, size_mode) in enumerate(loaded, start=1):
            adv, alpha, _ = P.composite(img, b["quad"], b["meta"], patch, inst,
                                        train=False, device=device,
                                        size_m=size_m, chain=None,
                                        size_mode=size_mode)
            cam_a = gradcam_det(model, adv, layer, roi)
            al = (alpha[0, 0] > 0.5).cpu().numpy()
            frac = float(cam_a[al].sum() / max(cam_a.sum(), 1e-9))
            FIG.tile(top[c], FIG.outline(cut3(adv), al[y0:y1, x0:x1], FIG.AMBER),
                     tag if r == 0 else "", "%.1f m decal" % size_m)
            FIG.tile(bot[c], overlay_cam(cut3(adv), _norm_crop(cam_a[y0:y1, x0:x1])), "",
                     "%.0f%% of CAM mass under the decal" % (100 * frac))

    fig.suptitle("Where the vehicle evidence sits, %s\nGrad-CAM at %s"
                 % (victim, chosen), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.subplots_adjust(hspace=0.30, wspace=0.06)
    out = out or os.path.join(N.FIGURE_DIR, "gradcam_%s__%s.png"
                              % ("_vs_".join(tags), victim))
    FIG._save(fig, out)
    return out


# ------------------------------------------------------------------- panel

def cam_panel(tags, victim, n=None, select="success", frames=None,
              device=None, out=None, layer_name=None, arm="seg", labels=None):
    """
    One row per instance: the clean image and every patch applied to it, each
    under the CAM the victim produces for it.
    """
    if arm == "det":
        return _cam_panel_det(tags, victim, n, select, frames, device, out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from physdecal.core import data as D
    from physdecal.evaluation import evaluate as E
    from physdecal.figures import panels as FIG
    from physdecal.victims import segmentation as M
    from physdecal.core import patch as P

    device = device or N.DEVICE
    n = n or N.PANEL_FRAMES
    rows, n_pool, note = FIG.pick_instances(tags[0], victim, "seg", n, select,
                                            frames)
    model_key, prompt = FIG.resolve_victim(victim, rows)
    index = D.build_index("holdout", nadir_only=False)
    model = M.load_model(model_key, device=device)
    if prompt:
        model.set_prompt(prompt)
    loaded = [(t, E.load_patch(t, device), E.patch_size_of(t),
               E.size_mode_of(t)) for t in tags]
    cols = 1 + len(loaded)

    fig, axes = plt.subplots(2 * len(rows), cols,
                             figsize=(3.05 * cols, 2.15 * 2 * len(rows)),
                             squeeze=False)
    chosen = None

    for r, rec in enumerate(rows):
        item = FIG.instance_for(index, rec["frame_id"], rec["target"])
        if item is None:
            raise SystemExit("instance %s/%s is in the CSV but not the index"
                             % (rec["frame_id"], rec["target"]))
        b, meta = FIG.load_one(item, device)
        img, inst = b["image"], b["inst"]
        roi = b["gt"]
        if chosen is None:
            cname, layer, shape = resolve_layer(model, img, roi, det=False)
            chosen = "%s %s" % (cname, list(shape))
        x0, y0, x1, y1 = FIG.crop_of(meta, img.shape[-2:])
        cut = lambda t: (t[0].permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]
                         if torch.is_tensor(t) and t.dim() == 4 and t.shape[1] == 3
                         else t[y0:y1, x0:x1])

        top, bot = axes[2 * r], axes[2 * r + 1]
        cam_c = gradcam(model, img, layer, roi)
        FIG.tile(top[0], cut(img), "Clean" if r == 0 else "",
                 "%s, %.0f m" % (rec["target"], float(rec["altitude_m"])))
        FIG.tile(bot[0], overlay_cam(cut(img), _norm_crop(cut(cam_c))), "",
                 "vehicle CAM")

        for c, (tag, patch, size_m, size_mode) in enumerate(loaded, start=1):
            adv, alpha, _ = P.composite(img, b["quad"], b["meta"], patch, inst,
                                        train=False, device=device,
                                        size_m=size_m, chain=None,
                                        size_mode=size_mode)
            cam_a = gradcam(model, adv, layer, roi)
            # mass of the CAM that the decal itself carries: the number that
            # separates "suppressed under the decal" from "attention pulled
            # elsewhere", which is the question the figure exists to answer
            al = (alpha[0, 0] > 0.5).cpu().numpy()
            frac = float(cam_a[al].sum() / max(cam_a.sum(), 1e-9))
            FIG.tile(top[c], FIG.outline(cut(adv), cut(al), FIG.AMBER),
                     (labels[c - 1] if labels else tag) if r == 0 else "",
                     "%.1f m decal" % size_m)
            FIG.tile(bot[c], overlay_cam(cut(adv), _norm_crop(cut(cam_a))), "",
                     "%.0f%% of CAM mass under the decal" % (100 * frac))

    fig.suptitle("Where the vehicle evidence sits, %s\nGrad-CAM at %s"
                 % (victim, chosen), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.subplots_adjust(hspace=0.30, wspace=0.06)
    out = out or os.path.join(N.FIGURE_DIR, "gradcam_%s__%s.png"
                              % ("_vs_".join(tags), victim))
    FIG._save(fig, out)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", nargs="*", required=True)
    ap.add_argument("--victim", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--select", default="success",
                    choices=["success", "attackable"])
    ap.add_argument("--frames", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", default="seg", choices=["seg", "det"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cam_panel(a.patches, a.victim or N.ATTACK_MODELS[0], a.n, a.select,
              a.frames, a.device, a.out, arm=a.arm)


if __name__ == "__main__":
    main()
