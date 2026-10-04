"""
patch.py
Puts a square metric patch on the roof of a vehicle and composites it into the
frame, differentiably, so gradients reach the patch texture.

The placement is a homography, not a paste. The roof is a planar rectangle of
known metric size, and the projected roof quad is already in the geometry JSON
written at capture time, so the map from patch pixels to image pixels is exact
up to the accuracy of the 3D box. Nothing here estimates a pose or a plane.

Corner order everywhere is the one gp_camera.oriented_box produces:

    quad[0] front-left   quad[1] front-right
    quad[3] rear-left    quad[2] rear-right

so the patch's +x axis always runs across the vehicle and its +y axis always
runs nose to tail. Every vehicle carries the patch in the same orientation,
which matters because otherwise a per-vehicle rotation would leak into the
result as if it were a viewpoint effect.

THE PATCH SURFACE IS THE ROOF, ALWAYS. It is fixed before optimisation and is
never selected as a function of the look-down angle. Do not add that branch.

Three things that will bite you if they are wrong:

  1. Composite at CAPTURE resolution, then resize once for the model. Resizing
     first and pasting after gives the patch a sharpness the camera could not
     have produced, and every number comes out optimistic.
  2. Multiply the patch alpha by the vehicle's own instance mask. That handles
     occlusion for free and stops the projected quad, which is a coarse
     cuboid face, from spilling past the silhouette at oblique angles.
  3. Keep EOT photometric jitter on the PATCH only, not the frame. Jittering
     the frame changes the ground truth's appearance and confounds the
     lighting axis of the evaluation.

WHAT CHANGED FROM paper2's np_patch
-----------------------------------
The geometry is unchanged and deliberately so: the roof homography was
verified numerically against the true 3D square and there is no reason to
touch it. The photometric half is replaced. Where paper2 jittered brightness,
contrast, noise and blur -- a digital-compositing EOT -- this composites
through printchain.PrintChain, which models the printer, the ink, the
sheet, the light and the sensor. Pass chain=None to get the old behaviour
back; that is the ablation, not the default.
"""

import math

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from physdecal import config as N

try:
    import kornia
    HAVE_KORNIA = True
except Exception:                                   # kornia is optional
    HAVE_KORNIA = False


# --------------------------------------------------------- patch parameters

class PatchTexture(torch.nn.Module):
    """
    The only trainable object in this experiment.

    Stored as an unconstrained tensor and squashed through tanh, so the patch
    is always inside [0,1] without a projection step. Clipping after each
    optimiser step also works, but tanh keeps the gradient meaningful at the
    boundary instead of zeroing it.
    """

    def __init__(self, res=None, init="grey", seed=None):
        super().__init__()
        res = res or N.PATCH_RES
        g = torch.Generator().manual_seed(N.SEED if seed is None else seed)
        if init == "random":
            raw = torch.rand(3, res, res, generator=g) * 2 - 1
        elif init == "grey":
            raw = torch.zeros(3, res, res)
        else:                                        # path to an image file
            img = cv2.cvtColor(cv2.imread(init), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (res, res)).astype(np.float32) / 255.0
            raw = torch.atanh(torch.from_numpy(img).permute(2, 0, 1)
                              .clamp(0.01, 0.99) * 2 - 1)
        self.raw = torch.nn.Parameter(raw)

    def forward(self):
        return (torch.tanh(self.raw) + 1.0) * 0.5     # 3 x R x R in [0,1]

    def save_png(self, path, upscale=8):
        img = (self().detach().cpu().permute(1, 2, 0).numpy() * 255)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.resize(img, None, fx=upscale, fy=upscale,
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------- geometry

def roof_metric_dims(meta):
    """
    (length, width) of this vehicle's roof rectangle in metres.

    box_len_m is logged per instance and already includes the actor's scale.
    The width is recovered from the authored aspect ratio, which is the only
    part of VEHICLE_DIMS that scaling cannot change.
    """
    auth = N.dims_for(meta["target"])
    L_auth, W_auth = float(auth[0]), float(auth[1])
    L = float(meta.get("box_len_m") or L_auth)
    if L <= 0.1:
        L = L_auth
    return L, L * (W_auth / max(L_auth, 1e-6))


def roof_panel_dims(meta):
    """
    (length, width) of the flat sheet-metal ROOF PANEL in metres, which is a
    different surface from the bounding-box top face and is the one a decal can
    actually be stuck to.

    The box top face of a 3.94 x 1.80 m hatchback is 7.1 square metres and
    includes the windscreen and the rear glass. Its roof panel is about
    1.21 x 1.35 m. Placing a decal by the first number and printing it by the
    second is how a simulated attack becomes physically impossible.
    """
    L, W = roof_metric_dims(meta)
    return L * N.ROOF_LEN_FRAC, W * N.ROOF_WID_FRAC


def max_patch_size_m(meta, fill=None):
    """The largest square decal that fits this vehicle's roof panel."""
    fill = N.ROOF_FILL if fill is None else fill
    rl, rw = roof_panel_dims(meta)
    return float(min(rl, rw) * fill)


def size_for(meta, size_m=None, mode=None):
    """
    The decal's metric side for one instance.

    mode "fixed"  every vehicle gets size_m, which is the conventional
                  protocol and is kept so results stay comparable with the
                  literature and with paper2. Across a mixed fleet it puts
                  decals on roofs too small to hold them.

    mode "roof"   every vehicle gets the largest square its own roof panel
                  admits, so no instance carries a decal that could not be
                  applied. Across a mixed fleet this is the mode whose number
                  means something: see the note in config.py section 6.

    IN MODE "roof", size_m IS AN UPPER CAP, NOT THE SIZE. This is the "keep it
    large where it fits, shrink it only where it does not" protocol:

        --size-mode roof --size 2.0

    gives a 2.0 m decal on any vehicle whose roof holds one and the roof
    maximum on every vehicle that does not. Pass no size, or ROOF_MAX_M None,
    and the cap is off and every vehicle gets its own roof maximum.

    A cap is worth having for a reason beyond tidiness: a decal is something
    an attacker has to print, carry and apply without being noticed, so an
    arbitrarily large one is not a free win even when a roof would hold it.
    The cap makes that a stated parameter rather than an accident of whichever
    vehicles happen to be in the fleet.
    """
    mode = mode or N.PATCH_SIZE_MODE
    if mode == "roof":
        fits = max_patch_size_m(meta)
        cap = size_m if size_m else N.ROOF_MAX_M
        return float(min(fits, cap)) if cap else float(fits)
    if mode != "fixed":
        raise SystemExit("PATCH_SIZE_MODE must be 'fixed' or 'roof', not '%s'"
                         % mode)
    return float(size_m if size_m else N.PATCH_SIZE_M)


def coverage_fraction(meta, size_m):
    """
    Decal area as a fraction of the vehicle's footprint.

    This is the quantity that predicts attack strength across vehicle classes,
    where absolute metres does not, so it is written into every results row.
    """
    L, W = roof_metric_dims(meta)
    return float((size_m ** 2) / max(L * W, 1e-6))


def patch_quad_in_image(roof_quad, L, W, centre_uv=None, size_m=None,
                        rot_deg=0.0, scale=1.0):
    """
    Image-space corners of the metric patch square lying on the roof.

    roof_quad : 4 x 2 float, image pixels, in gp_camera corner order
    L, W      : roof length and width, metres
    Returns   : 4 x 2 float32
    """
    centre_uv = centre_uv or N.PATCH_CENTRE_UV
    s = (size_m or N.PATCH_SIZE_M) * scale * 0.5

    # roof plane coordinates in metres: u across (0..W), v nose to tail (0..L)
    src = np.array([[0.0, 0.0], [W, 0.0], [W, L], [0.0, L]], dtype=np.float32)
    Hm = cv2.getPerspectiveTransform(src, np.asarray(roof_quad, np.float32))

    cu, cv_ = centre_uv[0] * W, centre_uv[1] * L
    th = math.radians(rot_deg)
    ct, st = math.cos(th), math.sin(th)
    local = np.array([[-s, -s], [s, -s], [s, s], [-s, s]], dtype=np.float32)
    rot = np.stack([local[:, 0] * ct - local[:, 1] * st,
                    local[:, 0] * st + local[:, 1] * ct], axis=1)
    sq = rot + np.array([cu, cv_], dtype=np.float32)

    out = cv2.perspectiveTransform(sq.reshape(1, 4, 2).astype(np.float32), Hm)
    return out.reshape(4, 2).astype(np.float32)


def homography_patch_to_image(dst_quad, res):
    """3x3 mapping patch pixel coordinates onto the image."""
    src = np.array([[0, 0], [res, 0], [res, res], [0, res]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, np.asarray(dst_quad, np.float32))


# ------------------------------------------------------------------ warping

def _warp_native(src, Hmat, out_hw):
    """
    Perspective warp with grid_sample, so kornia is optional.

    src   B x C x h x w
    Hmat  B x 3 x 3, mapping SOURCE pixels to DESTINATION pixels
    """
    B, Cc, h, w = src.shape
    H, W = out_hw
    dev, dt = src.device, src.dtype
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=dt),
                            torch.arange(W, device=dev, dtype=dt),
                            indexing="ij")
    ones = torch.ones_like(xs)
    dst = torch.stack([xs, ys, ones], 0).reshape(3, -1)          # 3 x HW
    Hinv = torch.inverse(Hmat.to(dt))                            # B x 3 x 3
    srcp = Hinv @ dst.unsqueeze(0)                               # B x 3 x HW
    z = srcp[:, 2]
    # Keep the sign. Clamping the magnitude away from zero avoids a divide by
    # zero on the horizon line without folding points behind the plane onto
    # the wrong side, which a plain clamp(min=eps) would do.
    z = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
    u = srcp[:, 0] / z
    v = srcp[:, 1] / z
    gx = (u / (w - 1)) * 2 - 1
    gy = (v / (h - 1)) * 2 - 1
    grid = torch.stack([gx, gy], -1).reshape(B, H, W, 2)
    return F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=True)


def warp(src, Hmat, out_hw):
    if HAVE_KORNIA:
        return kornia.geometry.transform.warp_perspective(
            src, Hmat, dsize=out_hw, mode="bilinear",
            padding_mode="zeros", align_corners=True)
    return _warp_native(src, Hmat, out_hw)


# ---------------------------------------------------------------------- EOT

def photometric_eot(patch, train=True, chain=None):
    """
    The photometric half of EOT.

    With a PrintChain, this is the physical channel: dot gain, ink tone
    response, sheen, shading, illuminant, exposure and sensor. Without one it
    falls back to additive noise only, which is the ablation arm.

    The old brightness/contrast pair is gone rather than kept alongside. It
    was a coarse stand-in for exposure and illuminant, and the chain models
    both properly; keeping both would double-count the same physical effect
    and widen the distribution past anything a camera does.
    """
    if not train:
        return patch
    if chain is not None:
        return chain(patch)
    return (patch + torch.randn_like(patch) * N.EOT_NOISE_STD).clamp(0, 1)


# ------------------------------------------------------------- compositing

def composite(images, quads, metas, patch, inst_masks, train=True,
              device=None, size_m=None, chain=None, size_mode=None):
    """
    images     B x 3 x H x W in [0,1]
    quads      B x 4 x 2 roof corners in image pixels
    metas      list of B meta dicts
    patch      3 x R x R in [0,1], the shared texture
    inst_masks B x 1 x H x W in {0,1}, this vehicle's visible pixels
    size_m     patch edge length in metres. Defaults to PATCH_SIZE_M, but an
               EVALUATION must pass the size the patch was OPTIMISED at, which
               evaluate.py reads back from that run's run.json. Scoring a
               1.0 m patch as though it were 1.2 m silently inflates it.
    chain      an printchain.PrintChain, or None for the ablation
    size_mode  "fixed" or "roof". See patch.size_for. With "roof", size_m
               is ignored and each vehicle gets the largest decal its own roof
               panel admits.

    Returns (patched_images, patch_alpha). patch_alpha is what the patch
    actually covers after occlusion, and it is needed later to separate damage
    under the patch from damage propagated away from it.
    """
    device = device or images.device
    B, _, H, W = images.shape
    res = patch.shape[-1]

    Hs, sizes = [], []
    cu, cv_ = N.PATCH_CENTRE_UV
    for i in range(B):
        rot = float(np.random.uniform(-N.EOT_ROTATION_DEG,
                                      N.EOT_ROTATION_DEG)) if train else 0.0
        sc = float(np.random.uniform(*N.EOT_SCALE)) if train else 1.0
        # Placement jitter. An operator sticking a decal on a roof by hand does
        # not hit the planned centre to better than a centimetre or two, and a
        # patch that only works when placed perfectly is not a physical attack.
        # Applied through the roof-plane centre rather than as an image-space
        # shift, so the offset is in metres on the vehicle and means the same
        # thing at every viewpoint.
        if train and N.EOT_TRANSLATE_FRAC:
            t = N.EOT_TRANSLATE_FRAC
            centre = (cu + float(np.random.uniform(-t, t)),
                      cv_ + float(np.random.uniform(-t, t)))
        else:
            centre = (cu, cv_)
        L, Wm = roof_metric_dims(metas[i])
        this_size = size_for(metas[i], size_m, size_mode)
        sizes.append(this_size)
        dst = patch_quad_in_image(quads[i].detach().cpu().numpy(), L, Wm,
                                  centre_uv=centre, size_m=this_size,
                                  rot_deg=rot, scale=sc)
        Hs.append(homography_patch_to_image(dst, res))
    Hmat = torch.from_numpy(np.stack(Hs)).to(device=device, dtype=images.dtype)

    tex = patch.unsqueeze(0).expand(B, -1, -1, -1)
    tex = photometric_eot(tex, train=train, chain=chain)
    body = torch.cat([tex, torch.ones(B, 1, res, res, device=device,
                                      dtype=images.dtype)], dim=1)

    warped = warp(body, Hmat, (H, W))
    rgb, alpha = warped[:, :3], warped[:, 3:4].clamp(0, 1)
    alpha = alpha * inst_masks.to(alpha.dtype)          # occlusion handling

    out = images * (1 - alpha) + rgb * alpha
    # sizes ride back with the result because an evaluation has to record the
    # size each instance actually received, which under mode="roof" differs
    # per vehicle and cannot be recovered from the config afterwards.
    return out.clamp(0, 1), alpha, sizes


def patch_pixel_count(alpha):
    return alpha.flatten(1).sum(1)
