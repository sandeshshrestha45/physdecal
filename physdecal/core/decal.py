"""
physdecal decal
The decal manifold, its projection, and the reference designs it is anchored
to.

WHY THIS FILE EXISTS
--------------------
paper2 optimised a free 3 x H x W pixel tensor and added soft penalties asking
it to look benign. The measured result was that the realism multiplier could
be raised sixteen-fold and the patch stayed television static. That is not a
tuning failure, it is structural: a soft penalty and the attack term are
summed, and at every pixel the attack term is willing to pay.

So realism here is a CONSTRAINT. The optimiser still takes a gradient step on
free pixels, and then the iterate is PROJECTED back onto a set whose every
member is a printable decal. Projected gradient descent, where the constraint
set is "things a printer can make and a person reads as a sticker" rather than
an epsilon ball.

The set is defined by four operators, composed in this order:

    band     Frequencies above what a PRINT_DPI printer can lay down, and
             above what the camera resolves at the operating GSD, cannot
             survive the physical channel. Optimising into them is optimising
             into a null space. Removing them costs nothing real and removes
             the static.

    flat     Real vinyl and real printed decals are flat colour areas with
             crisp boundaries. A guided filter with self-guidance is the cheap
             projection onto that: it smooths within regions and leaves
             boundaries alone.

    palette  Snap to INK_COUNT colours in CIELAB. This is what makes the thing
             a spot-colour decal rather than a gradient field, and it is also
             the only honest way to make a printability claim: the patch is
             made of the inks, so non-printability is zero by construction
             rather than small by penalty.

    anchor   Force the patch to be piecewise constant on the REGIONS of a
             named reference design. Without this the palette-quantised patch
             is a plausible-looking but meaningless mosaic. With it, it is a
             cargo marking, or a sunroof, or a helipad H, recoloured by the
             attack. The constraint is on layout, not colour -- see
             DecalManifold for the measurement that forced that choice.

Gradients pass through the projection straight through (Bengio et al. 2013):
the forward value is the projected one, the backward pass sees the identity.
The alternative, differentiating the projection itself, gives the optimiser a
route to exploit the projection's own smoothing and is how you end up with a
patch that only works before it is projected.

COLOUR SPACE
------------
Every colour comparison in this file is CIEDE2000 in CIELAB under D65, not
Euclidean RGB. The distinction matters here more than usual: the ink snap and
the anchor colour budget are both claims about what a person or a printer
sees, and RGB distance is not that. The implementation is differentiable
torch, so the same function backs the hard ink projection here and the soft
anchor term in losses.py.
"""

import math
import os
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from physdecal import config as N


# ===================================================== colour, sRGB <-> CIELAB

_M_RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
_WHITE_D65 = torch.tensor([0.95047, 1.00000, 1.08883])


def _srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92,
                       ((c.clamp(min=1e-8) + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb):
    """
    rgb  ... x 3 x H x W  in [0,1], sRGB
    out  ... x 3 x H x W  L in [0,100], a and b roughly [-128,127]
    """
    shp = rgb.shape
    lin = _srgb_to_linear(rgb.clamp(0, 1))
    m = _M_RGB2XYZ.to(rgb.device, rgb.dtype)
    flat = lin.reshape(-1, 3, shp[-2] * shp[-1])
    xyz = torch.einsum("ij,bjn->bin", m, flat)
    xyz = xyz / _WHITE_D65.to(rgb.device, rgb.dtype).view(1, 3, 1)

    eps = 216.0 / 24389.0
    kap = 24389.0 / 27.0
    f = torch.where(xyz > eps, xyz.clamp(min=1e-8) ** (1.0 / 3.0),
                    (kap * xyz + 16.0) / 116.0)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    lab = torch.stack([116.0 * fy - 16.0,
                       500.0 * (fx - fy),
                       200.0 * (fy - fz)], dim=1)
    return lab.reshape(shp)


def lab_to_rgb(lab):
    """Inverse of rgb_to_lab. Output is clamped to [0,1]: leaving the sRGB cube
    is exactly the out-of-gamut case, and it is penalised by the gamut loss
    rather than being allowed to persist as an unrepresentable colour."""
    shp = lab.shape
    flat = lab.reshape(-1, 3, shp[-2] * shp[-1])
    L, a, b = flat[:, 0], flat[:, 1], flat[:, 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    eps = 216.0 / 24389.0
    kap = 24389.0 / 27.0

    def finv(t):
        t3 = t ** 3
        return torch.where(t3 > eps, t3, (116.0 * t - 16.0) / kap)

    xyz = torch.stack([finv(fx), finv(fy), finv(fz)], dim=1)
    xyz = xyz * _WHITE_D65.to(lab.device, lab.dtype).view(1, 3, 1)
    minv = torch.linalg.inv(_M_RGB2XYZ.to(lab.device, torch.float32)).to(lab.dtype)
    lin = torch.einsum("ij,bjn->bin", minv, xyz).clamp(min=0.0)
    srgb = torch.where(lin <= 0.0031308, lin * 12.92,
                       1.055 * lin.clamp(min=1e-8) ** (1 / 2.4) - 0.055)
    return srgb.clamp(0, 1).reshape(shp)


def ciede2000(lab1, lab2):
    """
    Differentiable CIEDE2000 between two Lab tensors of matching shape
    ... x 3 x H x W. Returns ... x 1 x H x W.

    This is the full formula, not delta-E 76 wearing its name. The rotation
    term matters in the blues, which is where a printed decal on a dark roof
    lives, so the cheap version would quietly mis-state exactly the region the
    experiment cares about.

    ACHROMATIC COLOURS ARE A GRADIENT TRAP AND THEY ARE GUARDED BELOW.
    CIEDE2000 is stated in polar coordinates, and at zero chroma -- grey,
    white, black -- the hue angle is undefined. The forward value is fine, so
    nothing looks wrong; the BACKWARD pass returns NaN, from two places:

      atan2(0, 0)                     undefined derivative
      sqrt(C^7 / (C^7 + 25^7))        d sqrt(u)/du is infinite at u = 0, and
                                      du/dC is zero there, so the chain rule
                                      evaluates 0 * inf

    Every reference design here is mostly white, grey and black, so this is
    not a corner case for this experiment, it is the common case. Unguarded,
    the patch goes to NaN within about five optimiser steps and the run
    produces a blank file with no error. Measured, not hypothesised.
    """
    L1, a1, b1 = lab1[..., 0, :, :], lab1[..., 1, :, :], lab1[..., 2, :, :]
    L2, a2, b2 = lab2[..., 0, :, :], lab2[..., 1, :, :], lab2[..., 2, :, :]
    eps = 1e-8

    def _safe_atan2(y, x):
        """atan2 with a defined derivative at the origin. Where the chroma is
        zero the hue is arbitrary, and every term that consumes it is scaled
        by a chroma that is also zero, so returning 0 there changes no value
        and removes the NaN."""
        degen = (x.abs() + y.abs()) < 1e-6
        xs = torch.where(degen, torch.ones_like(x), x)
        ys = torch.where(degen, torch.zeros_like(y), y)
        return torch.atan2(ys, xs) % (2 * math.pi)

    C1 = torch.sqrt(a1 ** 2 + b1 ** 2 + eps)
    C2 = torch.sqrt(a2 ** 2 + b2 ** 2 + eps)
    Cbar = 0.5 * (C1 + C2)
    # eps UNDER the root, not in the denominator: it is the root that is
    # singular at zero, not the ratio.
    G = 0.5 * (1 - torch.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7) + eps))

    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p = torch.sqrt(a1p ** 2 + b1 ** 2 + eps)
    C2p = torch.sqrt(a2p ** 2 + b2 ** 2 + eps)
    h1p = _safe_atan2(b1, a1p)
    h2p = _safe_atan2(b2, a2p)

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = dhp - 2 * math.pi * torch.round(dhp / (2 * math.pi))
    dHp = 2 * torch.sqrt(C1p * C2p + eps) * torch.sin(dhp / 2)

    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hdiff = h1p - h2p
    hbp = torch.where(torch.abs(hdiff) <= math.pi, 0.5 * hsum,
                      torch.where(hsum < 2 * math.pi,
                                  0.5 * (hsum + 2 * math.pi),
                                  0.5 * (hsum - 2 * math.pi)))

    T = (1
         - 0.17 * torch.cos(hbp - math.radians(30))
         + 0.24 * torch.cos(2 * hbp)
         + 0.32 * torch.cos(3 * hbp + math.radians(6))
         - 0.20 * torch.cos(4 * hbp - math.radians(63)))
    dtheta = math.radians(30) * torch.exp(
        -(((hbp - math.radians(275)) / math.radians(25)) ** 2))
    Rc = 2 * torch.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7) + eps)
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / torch.sqrt(20 + (Lbp - 50) ** 2 + eps)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -torch.sin(2 * dtheta) * Rc

    de = torch.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                    + Rt * (dCp / Sc) * (dHp / Sh) + eps)
    return de.unsqueeze(-3)


def ciede2000_rgb(rgb1, rgb2):
    return ciede2000(rgb_to_lab(rgb1), rgb_to_lab(rgb2))


# ===================================================== reference decal designs

def _draw(design, S):
    """
    Procedural reference designs, BGR uint8, S x S.

    Procedural rather than a downloaded asset, for two reasons. One, the
    experiment then has no external file to lose, and anyone can reproduce the
    anchor exactly from this source. Two, each design is a real thing that
    appears on a real car roof, so the resulting patch is plausible in the
    scene and not merely smooth. A reviewer asking "would anyone look twice at
    that on a roof" has an answer.
    """
    img = np.full((S, S, 3), 235, np.uint8)
    u = lambda f: int(round(f * S))

    if design == "sunroof":
        cv2.rectangle(img, (0, 0), (S, S), (215, 215, 215), -1)
        cv2.rectangle(img, (u(.10), u(.14)), (u(.90), u(.86)), (60, 55, 50), -1)
        cv2.rectangle(img, (u(.10), u(.14)), (u(.90), u(.86)), (150, 150, 150),
                      max(2, u(.02)))
        cv2.line(img, (u(.10), u(.50)), (u(.90), u(.50)), (120, 118, 115),
                 max(2, u(.015)))

    elif design == "cargo":
        cv2.rectangle(img, (0, 0), (S, S), (245, 245, 245), -1)
        cv2.rectangle(img, (u(.06), u(.06)), (u(.94), u(.94)), (40, 90, 200),
                      max(3, u(.035)))
        cv2.rectangle(img, (u(.14), u(.16)), (u(.86), u(.44)), (40, 90, 200), -1)
        for i in range(4):                      # chevron band
            x0 = u(.14) + i * u(.19)
            pts = np.array([[x0, u(.56)], [x0 + u(.11), u(.56)],
                            [x0 + u(.05), u(.86)], [x0 - u(.06), u(.86)]], np.int32)
            cv2.fillPoly(img, [pts], (30, 160, 235))

    elif design == "helipad":
        cv2.rectangle(img, (0, 0), (S, S), (45, 45, 45), -1)
        cv2.circle(img, (S // 2, S // 2), u(.42), (240, 240, 240), max(3, u(.03)))
        cv2.rectangle(img, (u(.30), u(.22)), (u(.40), u(.78)), (240, 240, 240), -1)
        cv2.rectangle(img, (u(.60), u(.22)), (u(.70), u(.78)), (240, 240, 240), -1)
        cv2.rectangle(img, (u(.30), u(.45)), (u(.70), u(.55)), (240, 240, 240), -1)

    elif design == "solar":
        cv2.rectangle(img, (0, 0), (S, S), (120, 118, 115), -1)
        for r in range(3):
            for c in range(3):
                x0, y0 = u(.06 + c * .30), u(.06 + r * .30)
                cv2.rectangle(img, (x0, y0), (x0 + u(.26), y0 + u(.26)),
                              (90, 55, 30), -1)
                cv2.rectangle(img, (x0, y0), (x0 + u(.26), y0 + u(.26)),
                              (170, 168, 165), max(1, u(.008)))

    elif design == "rack":
        cv2.rectangle(img, (0, 0), (S, S), (200, 200, 205), -1)
        for y in (.22, .50, .78):
            cv2.rectangle(img, (u(.04), u(y - .045)), (u(.96), u(y + .045)),
                          (70, 70, 75), -1)
        cv2.rectangle(img, (u(.06), u(.06)), (u(.13), u(.94)), (45, 45, 50), -1)
        cv2.rectangle(img, (u(.87), u(.06)), (u(.94), u(.94)), (45, 45, 50), -1)

    else:
        raise SystemExit("Unknown ANCHOR_DESIGN '%s'. Known: %s"
                         % (design, ", ".join(N.ANCHOR_DESIGN_SWEEP)))
    return img


def anchor_image(design=None, size=None, device=None, save=True):
    """The reference design as 1 x 3 x S x S float RGB in [0,1]."""
    design = design or N.ANCHOR_DESIGN
    S = size or N.PATCH_RES
    bgr = _draw(design, S)
    if save:
        os.makedirs(N.REFERENCE_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(N.REFERENCE_DIR, "anchor_%s.png" % design), bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return t.to(device or N.DEVICE)


# ============================================================ the ink set

def load_inks(path=None, device=None, k=None):
    """
    The printable ink set, as k colours.

    If PRINTABLE_COLOURS_CSV holds a measured scan it is reduced to k
    representative colours by k-means in CIELAB, which is the right space to
    cluster in because the clustering is a claim about what looks the same.
    If it does not, a coarse cube stands in and INK_MEASURED stays False, which
    physdecal report checks before it will print a printability number.
    """
    path = path or N.PRINTABLE_COLOURS_CSV
    k = k or N.INK_COUNT
    if path and os.path.isfile(path):
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32).reshape(-1, 3)
        if arr.max() > 1.5:
            arr = arr / 255.0
        source = os.path.basename(path)
    else:
        g = np.linspace(0.05, 0.95, 6)
        arr = np.array([[r, gg, b] for r in g for gg in g for b in g], np.float32)
        source = "built-in cube (NOT a printer gamut)"

    arr = np.clip(arr, 0, 1)
    if len(arr) > k:
        lab = rgb_to_lab(torch.from_numpy(arr).T.reshape(1, 3, -1, 1)
                         ).reshape(3, -1).T.numpy()
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
        _, labels, centres = cv2.kmeans(lab.astype(np.float32), k, None,
                                        crit, 8, cv2.KMEANS_PP_CENTERS)
        # The ink is the real colour nearest each centre, never the centroid:
        # a centroid is an average of inks and no printer has that cartridge.
        inks = []
        for c in range(k):
            d = np.linalg.norm(lab - centres[c], axis=1)
            inks.append(arr[int(np.argmin(d))])
        arr = np.stack(inks)
    print("    inks: %d colours from %s" % (len(arr), source))
    return torch.from_numpy(arr.astype(np.float32)).to(device or N.DEVICE)


# ============================================================ the projections

def _gauss_kernel(sigma, device, dtype):
    r = max(1, int(round(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).view(1, 1, -1)


def gaussian_blur(x, sigma):
    if sigma <= 1e-3:
        return x
    k = _gauss_kernel(sigma, x.device, x.dtype)
    c = x.shape[1]
    pad = k.shape[-1] // 2
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"),
                 k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"),
                 k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x


def texels_per_image_px(patch_res=None, patch_size_m=None, gsd=None):
    """
    How many patch texels fall inside one image pixel at the operating ground
    sample distance. This is the number that decides what the camera can and
    cannot see of the texture.
    """
    res = patch_res or N.PATCH_RES
    size_m = patch_size_m or N.PATCH_SIZE_M
    gsd = gsd or N.BAND_GSD_M_PER_PX
    return res * gsd / size_m


def band_cutoff(patch_res=None, patch_size_m=None, gsd=None, dpi=None):
    """
    Cutoff frequency of the physical channel, in cycles per patch texel.

    Two limits, and the binding one wins:

      print   a dpi printer cannot lay a feature finer than 25.4/dpi mm, so
              anything above that frequency is a fiction of the tensor.
      camera  at the operating GSD the patch spans size_m/gsd image pixels,
              so texture finer than two texels per image pixel is averaged
              away by the sensor before the model ever sees it.

    At the settings this study runs -- a 1.2 m patch at 256 texels printed at
    300 dpi and photographed at roughly 0.028 m per pixel -- the print limit
    is around 27 cycles per texel and the camera limit around 0.08. The camera
    binds by two and a half orders of magnitude, which is the whole point:
    paper2's optimiser spent its budget in a band the sensor deletes, and that
    is why the static looked fearsome on screen and did nothing on paper.
    """
    res = patch_res or N.PATCH_RES
    size_m = patch_size_m or N.PATCH_SIZE_M
    dpi = dpi or N.PRINT_DPI

    mm_per_texel = size_m * 1000.0 / res
    print_cut = (dpi / 25.4 / 2.0) * mm_per_texel            # cycles / texel
    cam_cut = 1.0 / (2.0 * max(texels_per_image_px(res, size_m, gsd), 1e-6))
    return min(0.5, print_cut, cam_cut)


def band_sigma(patch_res=None, patch_size_m=None, gsd=None, dpi=None):
    """Gaussian sigma, in patch texels, whose transfer function is down to
    about 1/e at band_cutoff. sigma = 1/(2*pi*f_c) up to the usual factor."""
    fc = band_cutoff(patch_res, patch_size_m, gsd, dpi)
    return float(min(8.0, 1.0 / (2.0 * math.pi * max(fc, 1e-6))))


def guided_filter(x, radius=4, eps=1e-3):
    """
    Self-guided filter (He et al. 2013). Smooths inside regions, preserves
    edges. Used as the projection onto "flat areas with crisp boundaries",
    which is what a printed decal is and what plain total variation is not:
    TV shrinks every edge including the ones the design is made of.
    """
    def boxf(t):
        k = 2 * radius + 1
        t = F.avg_pool2d(F.pad(t, (radius,) * 4, mode="reflect"), k, stride=1)
        return t

    mean_x = boxf(x)
    mean_xx = boxf(x * x)
    var = mean_xx - mean_x * mean_x
    a = var / (var + eps)
    b = mean_x - a * mean_x
    return boxf(a) * x + boxf(b)


def _ink_distance_stack(x, inks):
    """B x K x H x W of CIEDE2000 from every pixel of x to every ink."""
    B, _, H, W = x.shape
    lab_x = rgb_to_lab(x)
    lab_i = rgb_to_lab(inks.T.reshape(1, 3, -1, 1))          # 1 x 3 x K x 1
    return torch.cat([ciede2000(lab_x, lab_i[:, :, k:k + 1, :].expand(B, 3, H, W))
                      for k in range(inks.shape[0])], dim=1)


def _gather_inks(idx, inks, shape):
    B, _, H, W = shape
    K = inks.shape[0]
    out = inks.T.view(1, 3, K, 1, 1).expand(B, 3, K, H, W)
    return out.gather(2, idx.unsqueeze(1).expand(B, 3, 1, H, W)).squeeze(2)


def snap_to_inks(x, inks):
    """
    Nearest ink per pixel, measured in CIELAB with CIEDE2000.

    x     B x 3 x H x W in [0,1]
    inks  K x 3         in [0,1]
    """
    d = _ink_distance_stack(x, inks)
    return _gather_inks(d.argmin(dim=1, keepdim=True), inks, x.shape)


def anchor_regions(anchor, supercell=None):
    """
    The region layout of the reference design: an integer label map in which
    every label is one flat area of the printed decal.

    A label is (which colour of the design this pixel belongs to) crossed with
    (which cell of a supercell x supercell grid it falls in). The first factor
    is what makes the output recognisable -- the chevrons stay chevrons, the H
    stays an H, because region boundaries are the design's own boundaries. The
    second factor is what gives the attack somewhere to go: without it the
    whole chevron block is one colour and the patch has four degrees of
    freedom in total.

    Returned as 1 x 1 x H x W of int64, labels compacted to 0..R-1.
    """
    supercell = supercell or N.ANCHOR_SUPERCELL
    a = anchor[0].permute(1, 2, 0).detach().cpu().numpy()
    H, W = a.shape[:2]

    q = np.round(a * 255).astype(np.int32)
    flat = q.reshape(-1, 3)
    _, colour_id = np.unique(flat, axis=0, return_inverse=True)
    colour_id = colour_id.reshape(H, W)

    ys = (np.arange(H) * supercell // H).reshape(-1, 1)
    xs = (np.arange(W) * supercell // W).reshape(1, -1)
    cell_id = ys * supercell + xs

    combined = colour_id.astype(np.int64) * (supercell * supercell) + cell_id
    _, labels = np.unique(combined, return_inverse=True)
    labels = labels.reshape(1, 1, H, W)
    return torch.from_numpy(labels).to(anchor.device)


def project_to_regions(x, labels, n_regions):
    """
    Make x constant inside every region of the label map, at the region mean.

    This is the projection onto "a flat-colour design with the anchor's
    layout". It is a true projection -- the region mean is the nearest
    piecewise-constant image in L2 -- and it is idempotent, so applying it
    every optimiser step costs one pass and does not drift.
    """
    B, C, H, W = x.shape
    lab = labels.expand(B, C, H, W).reshape(B * C, -1)
    flat = x.reshape(B * C, -1)

    sums = torch.zeros(B * C, n_regions, device=x.device, dtype=x.dtype)
    sums.scatter_add_(1, lab, flat)
    cnts = torch.zeros(B * C, n_regions, device=x.device, dtype=x.dtype)
    cnts.scatter_add_(1, lab, torch.ones_like(flat))
    means = sums / cnts.clamp(min=1.0)
    return means.gather(1, lab).reshape(B, C, H, W)


def edge_agreement(patch, anchor, thresh=0.08):
    """
    Fraction of the anchor's edge pixels that are still edges in the patch.

    This is the structural realism number, and it is the one that replaced a
    per-pixel colour distance. Colour distance to the anchor turned out to be
    the wrong measure of "does this still look like a cargo marking": a decal
    printed in different colours is still that decal, and a decal with the
    right colours in the wrong places is not.
    """
    def edges(t):
        g = t.mean(1, keepdim=True)
        gx = g[..., :, 1:] - g[..., :, :-1]
        gy = g[..., 1:, :] - g[..., :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return (gx.abs() + gy.abs()) > thresh

    ea, ep = edges(anchor), edges(patch)
    inter = (ea & ep).sum().float()
    return float(inter / ea.sum().clamp(min=1).float())


# ============================================================ the manifold

class DecalManifold:
    """
    The feasible set, and the projection onto it.

    level  0  none                                  free pixels, paper2's setting
           1  band                                  physically resolvable only
           2  band + flat                           flat areas, crisp edges
           3  band + flat + palette                 a printable spot-colour sheet
           4  band + flat + palette + anchor        a named decal, recoloured

    The anchor at level 4 is STRUCTURAL, not colorimetric, and that is a
    correction to the first version of this file. Constraining every pixel to
    a CIEDE2000 ball around the reference was measured to leave a median of
    ONE admissible ink per pixel on the cargo design and exactly one on
    helipad -- that is, the projection returned the reference itself and the
    attack had no degrees of freedom at all. The constraint was satisfied and
    the experiment was empty.

    Constraining the LAYOUT instead is both weaker and more faithful to what
    realism means here. The patch must be piecewise constant on the design's
    own regions, so the chevrons stay chevrons and the H stays an H, while the
    ink filling each region is the attacker's free variable. A decal printed
    in unexpected colours is still recognisably that decal; a decal with the
    right colours in scrambled places is not. The measured number that goes
    with this is edge agreement against the anchor, not colour distance.

    project() is called after every optimiser step. Every operator in it is
    idempotent, so repeated application costs one pass and does not drift.
    """

    def __init__(self, level=None, design=None, device=None, patch_size_m=None,
                 patch_res=None, inks=None, gsd=None, supercell=None,
                 palette_only=False):
        self.level = N.REALISM_LEVEL if level is None else level
        # Level 3 without the band and flat steps: the only constraint is
        # that every pixel is a measured printable ink.
        self.palette_only = palette_only
        if palette_only and self.level != 3:
            raise SystemExit("palette_only needs realism level 3")
        self.device = device or N.DEVICE
        self.design = design or N.ANCHOR_DESIGN
        self.res = patch_res or N.PATCH_RES
        self.size_m = patch_size_m or N.PATCH_SIZE_M
        self.gsd = gsd or N.BAND_GSD_M_PER_PX
        self.supercell = supercell or N.ANCHOR_SUPERCELL
        self.cutoff = band_cutoff(self.res, self.size_m, self.gsd)
        self.sigma = band_sigma(self.res, self.size_m, self.gsd)

        if self.level >= 4:
            self.anchor = anchor_image(self.design, self.res, self.device)
            self.labels = anchor_regions(self.anchor, self.supercell)
            self.n_regions = int(self.labels.max()) + 1
        else:
            self.anchor = None
            self.labels = None
            self.n_regions = 0

        if self.level >= 3:
            base = inks if inks is not None else load_inks(device=self.device)
            # The design's own spot colours join the ink set, which is what a
            # decal printer does. Without them the projection cannot reproduce
            # its own anchor -- measured at dE 29 and two surviving inks on the
            # cargo design -- so the anchor would be a target the feasible set
            # could not reach and the realism curve would start from a patch
            # that already looks nothing like the reference.
            self.inks = (self._merge_anchor_colours(base, self.anchor)
                         if self.level >= 4 else base)
        else:
            self.inks = None

    @staticmethod
    def _merge_anchor_colours(base, anchor):
        q = (anchor.clamp(0, 1) * 255).round().to(torch.uint8)
        cols = torch.unique(q.permute(0, 2, 3, 1).reshape(-1, 3), dim=0)
        cols = cols.to(base.dtype) / 255.0
        merged = torch.cat([cols, base], 0)
        keep, seen = [], []
        for i in range(merged.shape[0]):
            c = merged[i]
            if all(float(torch.norm(c - s)) > 1e-4 for s in seen):
                seen.append(c)
                keep.append(i)
        return merged[keep]

    def describe(self):
        names = {0: "free pixels (no constraint)", 1: "band",
                 2: "band+flat", 3: "band+flat+palette",
                 4: "band+flat+palette+anchor"}
        s = "realism level %d: %s" % (self.level, names[self.level])
        if self.palette_only:
            s = "realism level 3: palette only (no band, no flat)"
        elif self.level >= 1:
            s += ", band sigma %.2f px (cutoff %.3f cyc/texel)" % (self.sigma,
                                                                   self.cutoff)
        if self.level >= 3:
            s += ", %d inks" % self.inks.shape[0]
        if self.level >= 4:
            s += ", anchor '%s', %d regions" % (self.design, self.n_regions)
        return s

    @torch.no_grad()
    def _project_raw(self, x):
        x = x.clamp(0, 1)
        if self.level >= 1 and not self.palette_only:
            x = gaussian_blur(x, self.sigma)
        if self.level >= 2 and not self.palette_only:
            for _ in range(N.FLAT_ITERS):
                x = guided_filter(x, radius=max(2, self.res // 64),
                                  eps=N.FLAT_SIGMA_COLOR ** 2)
        if self.level >= 4:
            # Layout first, inks second. Snapping last is what guarantees the
            # shipped file is made of colours the printer actually has, and
            # because every region is already constant the snap cannot
            # reintroduce structure the anchor does not have.
            x = project_to_regions(x, self.labels, self.n_regions)
        if self.level >= 3:
            x = snap_to_inks(x.clamp(0, 1), self.inks)
        return x.clamp(0, 1)

    def project(self, x):
        """Hard projection. Use between optimiser steps."""
        return self._project_raw(x)

    def forward(self, x):
        """
        Straight-through projection. Use inside the graph: the model sees the
        projected patch, the gradient reaches the free variable unchanged.
        """
        return x + (self._project_raw(x) - x).detach()

    def degrees_of_freedom(self):
        """
        How many numbers the attacker actually controls. Reported in the paper
        beside every realism level, because an ASR is only interesting next to
        the size of the search space that produced it.
        """
        if self.level >= 4:
            return self.n_regions * 3
        return self.res * self.res * 3


class InkRegionTexture(torch.nn.Module):
    """
    The level-4 free variable, parametrised as the choice it actually is: one
    logit per (anchor region, ink). The forward value is the argmax ink in
    every region, so every iterate is exactly on the manifold; the backward
    value is the softmax mixture (straight-through softmax), so the gradient
    says what switching a region to each ink would do.

    WHY NOT A FREE RGB TENSOR BEHIND manifold.forward. That was the first
    version. Its backward is the identity, which says nothing about the snap
    to the nearest ink, so the raw pixels drift into tanh saturation without
    crossing a snap boundary and the printed patch stops changing. Measured
    overfitting 4 frames with no EOT: the loss went flat by step 100 at 0.68
    and 19.5% of vehicle pixels away from the decal were lost. Under this
    parametrisation, same frames and steps: loss 0.43, 28.2% lost.

    Initialised to the reference design itself, i.e. the unoptimised decal.
    """

    def __init__(self, manifold, init_logit=2.0):
        super().__init__()
        if manifold.level < 4:
            raise SystemExit("InkRegionTexture needs realism level 4")
        self.labels = manifold.labels[0, 0]                  # H x W
        self.register_buffer("inks", manifold.inks)
        R, K = manifold.n_regions, manifold.inks.shape[0]
        with torch.no_grad():
            ref = project_to_regions(manifold.anchor, manifold.labels, R)
            d = _ink_distance_stack(ref, manifold.inks)[0]   # K x H x W
            idx = torch.zeros(R, dtype=torch.long, device=d.device)
            idx[self.labels.flatten()] = d.argmin(0).flatten()
        raw = torch.zeros(R, K, device=d.device)
        raw[torch.arange(R, device=d.device), idx] = init_logit
        self.raw = torch.nn.Parameter(raw)

    def forward(self):
        soft = F.softmax(self.raw, dim=1)
        hard = F.one_hot(soft.argmax(1), soft.shape[1]).to(soft.dtype)
        w = hard + soft - soft.detach()
        return (w @ self.inks)[self.labels].permute(2, 0, 1)    # 3 x H x W


class InkPixelTexture(torch.nn.Module):
    """
    InkRegionTexture with every pixel its own region: one logit per (pixel,
    ink), for the level-3 palette-only patch. Same straight-through softmax,
    so every iterate is made of measured inks and the gradient still says
    what switching a pixel to each ink would do. Initialised to the ink
    nearest mid-grey, the level-0 texture's starting point.
    """

    def __init__(self, manifold, init_logit=2.0):
        super().__init__()
        if manifold.level != 3 or not manifold.palette_only:
            raise SystemExit("InkPixelTexture needs realism level 3 with "
                             "--palette-only; it bypasses band and flat")
        self.register_buffer("inks", manifold.inks)
        self.res = manifold.res
        K = manifold.inks.shape[0]
        with torch.no_grad():
            grey = torch.full((1, 3, 1, 1), 0.5, device=manifold.inks.device)
            idx = int(_ink_distance_stack(grey, manifold.inks)[0, :, 0, 0].argmin())
        raw = torch.zeros(self.res * self.res, K, device=manifold.inks.device)
        raw[:, idx] = init_logit
        self.raw = torch.nn.Parameter(raw)

    def forward(self):
        soft = F.softmax(self.raw, dim=1)
        hard = F.one_hot(soft.argmax(1), soft.shape[1]).to(soft.dtype)
        w = hard + soft - soft.detach()
        return (w @ self.inks).T.reshape(3, self.res, self.res)


# ============================================================ realism metrics

@torch.no_grad()
def palette_size(meta):
    """
    How many colours the manifold OFFERED a run, parsed from its own manifold
    string, or None if that run recorded none.

    This exists because "9 inks" under a figure is not a palette size and reads
    exactly like one. ink_count is how many colours the finished patch USED. At
    level 4 the anchor's spot colours join the ink set, so a run with
    INK_COUNT = 8 legitimately shows 10 used of 11 available, and a reader who
    takes the label at face value concludes the palette constraint is not being
    enforced.
    """
    m = re.search(r", (\d+) inks", (meta or {}).get("manifold") or "")
    return int(m.group(1)) if m else None


def realism_metrics(patch, anchor=None, inks=None, patch_size_m=None):
    """
    The numbers that go in the realism column of the results table. Every one
    is objective; none of them is a human study, and the paper should say so.

      ink_count     distinct colours after 8-bit quantisation. A printed decal
                    has a handful. Static has thousands.
      delta_e_med   median CIEDE2000 to the anchor design
      delta_e_p95   the tail, which is what a person notices
      hf_ratio      fraction of texture energy above the physical channel
                    cutoff. This is the static index: energy up there cannot
                    survive printing and photographing, so a high value means
                    the digital number was bought with texture that will not
                    exist on paper.
                    CAVEAT, and it must be stated wherever this number is:
                    crisp region boundaries are legitimately broadband, so a
                    genuine flat-colour decal does not score zero. The metric
                    separates static from structure only in combination with
                    ink_count -- static is high hf AND high ink count, a decal
                    is moderate hf and a handful of inks.
      edge_iou      fraction of the anchor's edges still present in the patch.
                    The structural realism number. Only defined at level 4,
                    where there is an anchor to be structurally faithful to.
      nps           Sharif et al. non-printability, for comparability only
    """
    patch = patch.clamp(0, 1)
    q = (patch * 255).round().to(torch.uint8)
    cols = q.permute(0, 2, 3, 1).reshape(-1, 3)
    ink_count = int(torch.unique(cols, dim=0).shape[0])

    out = {"ink_count": ink_count}

    if anchor is not None:
        de = ciede2000_rgb(patch, anchor.expand_as(patch)).flatten()
        out["delta_e_med"] = float(de.median())
        out["delta_e_p95"] = float(torch.quantile(de.float(), 0.95))
        out["edge_iou"] = edge_agreement(patch, anchor.expand_as(patch))

    # Spectral energy above the physical channel cutoff. The same function
    # the projection uses, so the metric measures the thing the constraint
    # constrains rather than a second, differently defined band.
    size_m = patch_size_m or N.PATCH_SIZE_M
    res = patch.shape[-1]
    cutoff = band_cutoff(res, size_m)
    g = patch.mean(1, keepdim=True)
    g = g - g.mean()                    # DC carries most of the energy of any
                                        # image and says nothing about texture;
                                        # leaving it in makes every patch look
                                        # band limited
    Fm = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs() ** 2
    fy = torch.fft.fftshift(torch.fft.fftfreq(res, device=patch.device))
    fx = fy
    rr = torch.sqrt(fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2)
    hi = (rr > cutoff).float()
    out["hf_ratio"] = float((Fm * hi).sum() / Fm.sum().clamp(min=1e-9))

    if inks is not None:
        lab_x = rgb_to_lab(patch)
        d = torch.stack([ciede2000(lab_x,
                                   rgb_to_lab(inks[k].view(1, 3, 1, 1)).expand_as(lab_x))
                         for k in range(inks.shape[0])], 0)
        out["nps"] = float(d.min(0).values.mean())
    return out


def main():
    # Self-check. Prints the numbers that justify the design choice, so the
    # claim in the proposal is reproducible in one command.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(0)

    print("anchor designs:", ", ".join(N.ANCHOR_DESIGN_SWEEP))
    noise = torch.rand(1, 3, N.PATCH_RES, N.PATCH_RES, device=dev)
    inks = load_inks(device=dev)
    for lvl in N.REALISM_SWEEP:
        man = DecalManifold(level=lvl, device=dev, inks=inks)
        p = man.project(noise)
        m = realism_metrics(p, man.anchor, inks)
        extra = (("edge_iou=%.2f dE50=%.1f" % (m["edge_iou"], m["delta_e_med"]))
                 if "edge_iou" in m else "")
        print("  level %d  ink=%-6d hf=%.3f dof=%-7d %s"
              % (lvl, m["ink_count"], m["hf_ratio"],
                 man.degrees_of_freedom(), extra))
        print("           %s" % man.describe())


if __name__ == "__main__":
    main()
