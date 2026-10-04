"""
losses.py
The objective.

    L = L_seg                        hinged suppression of the vehicle posterior
      + w_det   * L_det              hinged suppression of the dense vehicle score
      + w_ink   * L_ink              soft distance to the nearest printable ink
      + w_flat  * L_flat             anchored anisotropic total variation
      + w_anchor* L_anchor           hinged CIEDE2000 to the reference design
      + w_gamut * L_gamut            stay inside the measured gamut
      + w_nps   * L_nps              Sharif et al., for comparability only

WHAT THE REALISM TERMS ARE FOR NOW, WHICH IS NOT WHAT THEY WERE FOR IN paper2
----------------------------------------------------------------------------
In paper2 these terms were load bearing: they were the only thing standing
between the optimiser and television static, and they lost. Here the decal
manifold projection in physdecal decal already guarantees the patch is band
limited, flat, made of real inks, and laid out on a real design. The iterate
is inside the feasible set before any of these terms are evaluated.

So their job has changed. They now choose AMONG feasible patches, which is a
much easier job and one a soft penalty is actually suited to. L_anchor decides
whether the recoloured cargo marking keeps the reference palette or drifts to
whatever ink attacks best; L_flat decides whether region boundaries stay
where the design put them. Neither is being asked to invent realism from
nothing.

L_nps is kept for one reason only: every physical patch paper since Sharif
et al. reports it, and a paper that does not is hard to compare against. It is
close to zero by construction here because the patch is literally made of the
ink set, and physdecal report refuses to print it as a headline number while
INK_MEASURED is False.

THE ANCHORED TV TERM IS THE ONE NEW SOFT TERM
---------------------------------------------
Plain total variation penalises every edge, including the edges the design is
made of, so pushing it up turns a cargo marking into a blur. The version here
weights the penalty DOWN wherever the anchor has an edge:

    L_flat = mean( |grad p| * exp(-|grad anchor| / k) )

which reads as "be flat, except where the reference is not". Under it the
chevrons keep their boundaries while everything between them is driven to a
constant, which is exactly what a printed decal looks like and what plain TV
cannot express.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from physdecal import config as N
from physdecal.core import decal as D


# The colour terms below are measured in CIEDE2000 units, which run to tens,
# while the attack hinges run to about one. Summed raw, the realism block
# outbids the attack by an order of magnitude and the weights in config.py
# would all have to be tiny numbers whose size meant nothing. Every colour
# term is therefore divided by this scale, so all weights in the config are
# comparable and a weight of 1.0 means "this term matters as much as the
# attack does". DE_SCALE is one just-noticeable step times ten, which is about
# where a colour difference stops being a nuance and becomes a different
# colour.
DE_SCALE = 10.0


# =================================================================== attack

# The seg hinge below is in logit units, which run to about ten on a confident
# vehicle pixel, while the realism terms were weighted against an attack term
# that ran to about one. Divided by this so the weights in config.py keep
# their meaning, as DE_SCALE does for the colour terms.
LOGIT_SCALE = 10.0


def seg_attack_loss(m_veh, gt_mask, alpha=None, tau=None, objective=None,
                    p_veh=None):
    """
    m_veh    B x 1 x H x W   the model's vehicle log-odds, log p/(1-p)
    gt_mask  B x 1 x H x W   ground truth vehicle pixels
    alpha    B x 1 x H x W   decal footprint, needed by "vanish_away"
    objective                see config.SEG_OBJECTIVE
    p_veh    B x 1 x H x W   the vehicle posterior, needed by "vanish_prob"

    "vanish" hinges every vehicle pixel, including the ones under the decal,
    and those are won by covering them. The optimiser can meet most of the
    objective by occlusion alone, and measured over every patch in this study
    it did: pixels lost away from the decal stayed under 4% of the vehicle
    even with free pixels. "vanish_away" drops the footprint from the hinge
    so only the adversarial effect is rewarded. "untargeted" is the
    whole-frame formulation of the patch literature (LaVAN, IAP): every pixel
    is pushed off its true label, so background is also pushed to vehicle.

    Hinged so the optimiser stops spending gradient on pixels already beaten
    down to tau and moves to the ones still holding out. Without the hinge a
    handful of easy pixels dominate and the patch converges on one vehicle.

    WHY LOG-ODDS AND NOT THE POSTERIOR. "vanish_prob" hinges p_veh itself.
    d p / d logit = p(1-p), so a pixel the victim is confident about gets
    almost no gradient: measured over the finetuned victims, the median
    vehicle pixel away from the decal has log-odds 4.6 to 8.2, which is a
    gradient of 1e-2 to 3e-4, and p_veh is clamped at 1-1e-6 so up to 8% of
    them got exactly zero. Only the uncertain pixels, under and beside the
    decal, could move. In log-odds every vehicle pixel still above tau pulls
    with the same unit gradient until it flips. Measured, the correction
    changes nothing held out (ASR 0.147 vs 0.136 on segformer_b0 at 1000
    steps): the limit is that one patch must serve every vehicle, not the
    gradient. vanish_prob stays the default because it is what the reported
    patches were optimised with.
    """
    tau = N.ATTACK_TAU if tau is None else tau
    objective = objective or N.SEG_OBJECTIVE
    m_tau = float(np.log(tau / (1.0 - tau)))
    if objective == "vanish_prob":
        denom = gt_mask.flatten(1).sum(1).clamp(min=1.0)
        hinge = (F.relu(p_veh - tau) * gt_mask).flatten(1).sum(1) / denom
        return hinge.mean()
    # Plain per-pixel cross entropy over the whole frame, no hinge, the
    # supervisor's formulation (adversarial-ml-tutorial.org). m_veh is the
    # two-class logit difference, so BCE-with-logits on it IS the softmax CE.
    # "ce_targeted": the target label map is background everywhere, i.e. the
    # vehicle label is replaced by background; minimise CE(pred, target).
    # "ce_untargeted": maximise CE(pred, gt), i.e. minimise -CE(pred, gt).
    if objective == "ce_targeted":
        return F.binary_cross_entropy_with_logits(m_veh, torch.zeros_like(m_veh))
    if objective == "ce_untargeted":
        return -F.binary_cross_entropy_with_logits(m_veh, gt_mask)
    if objective == "vanish":
        mask, margin = gt_mask, m_veh
    elif objective == "vanish_away":
        mask, margin = gt_mask * (alpha < 0.5).to(gt_mask.dtype), m_veh
    elif objective == "untargeted":
        # log-odds of the TRUE label: m on vehicle pixels, -m elsewhere
        mask = torch.ones_like(gt_mask)
        margin = torch.where(gt_mask > 0.5, m_veh, -m_veh)
    else:
        raise SystemExit("SEG_OBJECTIVE must be vanish_prob, vanish, "
                         "vanish_away, untargeted, ce_targeted or "
                         "ce_untargeted, not '%s'" % objective)
    denom = mask.flatten(1).sum(1).clamp(min=1.0)
    hinge = (F.relu(margin - m_tau) * mask).flatten(1).sum(1) / denom
    return hinge.mean() / LOGIT_SCALE


def det_attack_loss(dense_maps, gt_mask, boxes=None, tau=None,
                    objective="hinge"):
    """
    The detection analogue of the above, and deliberately the same shape so
    the two terms are commensurate and can be summed without a scale factor
    that would need justifying.

    dense_maps  list over FPN levels of B x 1 x H x W vehicle probability
    gt_mask     B x 1 x H x W ground truth vehicle pixels
    boxes       optional B x 4 xyxy. When given, the hinge covers the whole
                ground truth BOX rather than the mask.

    Why the box is an option and the mask is the default. A detector is
    predicting a box, so the locations that matter are the ones inside it,
    including the background pixels between the vehicle and the box edge. But
    hinging those teaches the patch to suppress whatever happens to sit beside
    the car in the training frames, which does not transfer. The mask is the
    conservative choice and is the default; the box is available for the
    ablation that shows the difference.

    Levels are averaged, not summed, so adding a level to the FPN would not
    silently reweight this term against the segmentation one.
    """
    tau = N.DET_ATTACK_TAU if tau is None else tau
    # The cross-entropy forms of seg_attack_loss, on the detector's per-location
    # vehicle score p (a per-class sigmoid, max over anchors and vehicle
    # classes), averaged over the frame and then over FPN levels.
    # "ce_targeted": the target is background everywhere, CE = -log(1 - p).
    # "ce_untargeted": minus the CE to the ground-truth vehicle mask.
    if objective in ("ce_targeted", "ce_untargeted"):
        target = (torch.zeros_like(gt_mask) if objective == "ce_targeted"
                  else gt_mask)
        sign = 1.0 if objective == "ce_targeted" else -1.0
        terms = [sign * F.binary_cross_entropy(dm.clamp(1e-6, 1 - 1e-6), target)
                 for dm in dense_maps]
        return torch.stack(terms).mean()
    if objective != "hinge":
        raise SystemExit("det objective must be hinge, ce_targeted or "
                         "ce_untargeted, not '%s'" % objective)
    if boxes is not None:
        m = torch.zeros_like(gt_mask)
        for i, b in enumerate(boxes):
            x0, y0, x1, y1 = [int(v) for v in b]
            m[i, :, max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 1.0
    else:
        m = gt_mask

    denom = m.flatten(1).sum(1).clamp(min=1.0)
    terms = []
    for dm in dense_maps:
        hinge = (F.relu(dm - tau) * m).flatten(1).sum(1) / denom
        terms.append(hinge.mean())
    return torch.stack(terms).mean() if terms else torch.zeros((), device=gt_mask.device)


# ================================================================== realism

def anchored_tv(patch, anchor, k=0.10):
    """
    Total variation weighted down where the anchor has an edge.

    patch, anchor  B x 3 x H x W in [0,1]

    k sets how strong an anchor edge has to be before the penalty is released.
    At 0.10 a region boundary in any of the shipped designs is well past it
    and a print-chain shading gradient is well inside it, which is the
    separation the term needs.
    """
    def grad(t):
        dh = (t[:, :, 1:, :] - t[:, :, :-1, :]).abs().mean(1, keepdim=True)
        dw = (t[:, :, :, 1:] - t[:, :, :, :-1]).abs().mean(1, keepdim=True)
        return F.pad(dh, (0, 0, 0, 1)), F.pad(dw, (0, 1, 0, 0))

    gph, gpw = grad(patch)
    gah, gaw = grad(anchor)
    wh = torch.exp(-gah / k)
    ww = torch.exp(-gaw / k)
    return (gph * wh).mean() + (gpw * ww).mean()


def total_variation(patch):
    dh = (patch[:, :, 1:, :] - patch[:, :, :-1, :]).abs().mean()
    dw = (patch[:, :, :, 1:] - patch[:, :, :, :-1]).abs().mean()
    return dh + dw


def ink_loss(patch, inks, beta=0.25):
    """
    Soft-min CIEDE2000 to the nearest ink.

    Soft-min rather than min, because a hard min has zero gradient to every
    ink except the winner, and early in the run the winner is arbitrary. beta
    controls the softness; at 0.25 the two nearest inks both receive gradient
    and the rest effectively do not.
    """
    lab_p = D.rgb_to_lab(patch)
    d = torch.stack([
        D.ciede2000(lab_p, D.rgb_to_lab(inks[k].view(1, 3, 1, 1)).expand_as(lab_p))
        for k in range(inks.shape[0])], dim=0)                 # K x B x 1 x H x W
    return (-beta * torch.logsumexp(-d / beta, dim=0)).mean() / DE_SCALE


def anchor_loss(patch, anchor, budget=None):
    """
    Hinged CIEDE2000 to the reference design: free inside the budget,
    penalised beyond it.

    Hinged rather than plain, because the point is not to reproduce the
    reference -- that patch has no attack in it -- but to stop the recolouring
    running away to a palette nobody would print. Inside the budget the
    optimiser should be spending its effort on the attack, and a plain
    distance would keep charging it for colour it is entitled to.
    """
    budget = N.ANCHOR_DELTA_E if budget is None else budget
    de = D.ciede2000_rgb(patch, anchor.expand_as(patch))
    return F.relu(de - budget).mean() / budget


def gamut_loss(patch, inks):
    """
    Distance outside the convex hull of the ink set, approximated by distance
    to the nearest ink beyond a tolerance.

    An honest convex-hull distance in Lab would be better and is not worth the
    code: at INK_COUNT colours the hull is a coarse polytope and the nearest
    ink is within a few dE of it everywhere that matters.
    """
    lab_p = D.rgb_to_lab(patch)
    d = torch.stack([
        D.ciede2000(lab_p, D.rgb_to_lab(inks[k].view(1, 3, 1, 1)).expand_as(lab_p))
        for k in range(inks.shape[0])], dim=0).min(0).values
    return F.relu(d - DE_SCALE).mean() / DE_SCALE


def nps_loss(patch, inks):
    """
    Sharif et al. non-printability, in RGB as the original defines it.

    In RGB and not in Lab on purpose. The point of carrying this term is
    comparability with the prior literature, and reimplementing it in a better
    colour space would produce a number that is not the number other papers
    report.
    """
    p = patch.permute(0, 2, 3, 1).reshape(-1, 3)
    d = torch.cdist(p, inks.view(-1, 3))
    return d.min(dim=1).values.mean()


# ================================================================ the whole

class Objective:
    """
    Holds the fixed tensors -- inks, anchor -- so the optimiser loop does not
    rebuild them every step, and reports its own breakdown so a run log says
    where the loss actually went.
    """

    def __init__(self, manifold, task=None, device=None, seg_objective=None,
                 det_objective="hinge"):
        self.man = manifold
        self.task = task or N.ATTACK_TASK
        self.seg_objective = seg_objective or N.SEG_OBJECTIVE
        self.det_objective = det_objective
        self.device = device or N.DEVICE
        self.inks = manifold.inks
        self.anchor = manifold.anchor
        if self.task not in ("seg", "det", "joint"):
            raise SystemExit("ATTACK_TASK must be seg, det or joint, not '%s'"
                             % self.task)

    def realism(self, patch):
        """Returns (total, breakdown dict). Terms with a zero weight, or with
        no tensor to compare against at this realism level, are skipped rather
        than added as zero, so the log shows what was actually evaluated."""
        parts = {}
        total = torch.zeros((), device=patch.device)

        if self.anchor is not None:
            if N.W_FLAT:
                parts["flat"] = anchored_tv(patch, self.anchor.expand_as(patch))
            if N.W_ANCHOR:
                parts["anchor"] = anchor_loss(patch, self.anchor)
        elif N.W_FLAT:
            # No anchor to protect, so the ordinary isotropic penalty. Below
            # level 4 there are no design edges worth keeping.
            parts["flat"] = total_variation(patch)

        if self.inks is not None:
            if N.W_INK:
                parts["ink"] = ink_loss(patch, self.inks)
            if N.W_GAMUT:
                parts["gamut"] = gamut_loss(patch, self.inks)
            if N.W_NPS:
                parts["nps"] = nps_loss(patch, self.inks)

        weights = {"flat": N.W_FLAT, "anchor": N.W_ANCHOR, "ink": N.W_INK,
                   "gamut": N.W_GAMUT, "nps": N.W_NPS}
        for k, v in parts.items():
            total = total + weights[k] * v
        return total, {k: float(v) for k, v in parts.items()}

    def __call__(self, patch, m_veh=None, gt_mask=None, dense_maps=None,
                 boxes=None, alpha=None, p_veh=None):
        parts = {}
        total = torch.zeros((), device=patch.device)

        if self.task in ("seg", "joint"):
            if m_veh is None:
                raise SystemExit("task '%s' needs the segmentation log-odds"
                                 % self.task)
            l = seg_attack_loss(m_veh, gt_mask, alpha=alpha,
                                objective=self.seg_objective, p_veh=p_veh)
            parts["seg"] = float(l)
            total = total + l

        if self.task in ("det", "joint"):
            if not dense_maps:
                raise SystemExit("task '%s' needs detector dense maps"
                                 % self.task)
            l = det_attack_loss(dense_maps, gt_mask, boxes,
                                objective=self.det_objective)
            parts["det"] = float(l)
            w = N.JOINT_DET_WEIGHT if self.task == "joint" else 1.0
            total = total + w * l

        r, rparts = self.realism(patch)
        parts.update(rparts)
        parts["realism_total"] = float(r)
        total = total + r

        parts["total"] = float(total)
        return total, parts
