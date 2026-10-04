"""
physdecal printchain
A differentiable forward model of what happens to a patch between "a tensor on
a GPU" and "pixels a segmentation model sees", when the patch is physically
printed, stuck on a roof, and photographed.

WHY THIS FILE EXISTS
--------------------
Every patch paper applies expectation over transformation. Almost all of them
apply the SAME EOT: brightness, contrast, additive noise, blur, small rotation
and scale. paper2 did too. Not one of those terms models a printer, an ink, a
sheet of paper, or the light falling on it, and the gap between a digital
composite number and a physical number is made of exactly those things.

So the EOT distribution here is the physical channel, sampled per draw:

    ink        gamma and saturation of the ink-on-substrate tone response
    dot gain   the printer's MTF, a Gaussian in patch texels
    sheen      a low-frequency specular highlight off the sheet, which is the
               single most under-modelled term in printed-patch work: a glossy
               print at the wrong angle is a white blob and the attack is gone
    shadow     low-frequency shading across the decal from the roof curve
    balance    per-channel illuminant cast, because indoor light is not D65
    exposure   the camera's own response to the whole scene
    sensor     read noise and a DCT-domain low pass standing in for the JPEG
               every phone writes

A patch optimised under this distribution is optimised for the thing that will
actually be photographed. A patch optimised without it is optimised for a
monitor.

EVERY TERM IS APPLIED TO THE PATCH ONLY, NEVER THE FRAME. Jittering the frame
changes the appearance of the ground truth and confounds the lighting axis of
the evaluation, which is the one thing this dataset was captured to measure.

All of it is differentiable and cheap: the whole chain is a handful of
pointwise ops and two separable blurs, so it costs a few percent of a forward
pass through the victim.
"""

import torch
import torch.nn.functional as F

from physdecal import config as N
from physdecal.core.decal import gaussian_blur


def _u(lo, hi, shape, device):
    return torch.rand(shape, device=device) * (hi - lo) + lo


def _low_freq_field(B, H, W, device, scale=8):
    """
    A smooth random field in [0,1], used for sheen and shading.

    Generated at low resolution and upsampled bilinearly rather than blurred
    from noise: it is cheaper, and it guarantees the field really is smooth
    instead of being noise with the high frequencies attenuated, which still
    has structure at the patch texel scale and would let the optimiser treat
    the "shading" as texture it can exploit.
    """
    small = torch.rand(B, 1, scale, scale, device=device)
    return F.interpolate(small, size=(H, W), mode="bilinear",
                         align_corners=False)


def rgb_to_gray(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def dct_lowpass(x, strength):
    """
    Stand-in for the JPEG a phone writes and a drone logs.

    Implemented as a smooth spectral roll-off rather than a real 8x8 DCT
    quantisation, because true JPEG quantisation has zero gradient almost
    everywhere and would silently stop the optimiser from seeing this term at
    all. The purpose here is to deny the attack the highest frequencies, and a
    roll-off does that while staying differentiable.
    """
    if strength <= 0:
        return x
    H, W = x.shape[-2:]
    fx = torch.fft.fftfreq(W, device=x.device).view(1, 1, 1, -1)
    fy = torch.fft.fftfreq(H, device=x.device).view(1, 1, -1, 1)
    r = torch.sqrt(fx ** 2 + fy ** 2)
    # cutoff at a quarter of Nyquist when strength is 1
    fc = 0.5 * (1.0 - 0.75 * strength)
    H_filt = 1.0 / (1.0 + (r / max(fc, 1e-3)) ** 4)
    X = torch.fft.fft2(x)
    return torch.real(torch.fft.ifft2(X * H_filt))


class PrintChain:
    """
    Samples a physical channel and applies it to a patch.

    Call with the patch in [0,1]; get back the patch as it would look coming
    off a printer, stuck to a roof, lit, and photographed. Set
    N.PRINT_CHAIN False to fall back to the ordinary digital EOT, which is the
    ablation the paper needs to show the chain is doing something.
    """

    def __init__(self, enabled=None, device=None):
        self.enabled = N.PRINT_CHAIN if enabled is None else enabled
        self.device = device or N.DEVICE

    def sample(self, B):
        """Draw one set of channel parameters per batch element."""
        d = self.device
        return {
            "gamma":     _u(*N.PC_GAMMA, (B, 1, 1, 1), d),
            "sat":       _u(*N.PC_SATURATION, (B, 1, 1, 1), d),
            "exposure":  _u(*N.PC_EXPOSURE, (B, 1, 1, 1), d),
            "dot_gain":  float(_u(*N.PC_DOT_GAIN, (1,), d)),
            "wb":        1.0 + (torch.rand(B, 3, 1, 1, device=d) * 2 - 1)
                         * N.PC_WHITE_BALANCE,
            "sheen_amt": torch.rand(B, 1, 1, 1, device=d) * N.PC_SHEEN,
            "shadow_amt": torch.rand(B, 1, 1, 1, device=d) * N.PC_SHADOW,
        }

    def __call__(self, patch, params=None):
        """
        patch  B x 3 x H x W in [0,1]
        returns the same shape, in [0,1]
        """
        B, _, H, W = patch.shape
        x = patch.clamp(0, 1)

        if not self.enabled:
            # Ablation path: the ordinary digital EOT, so the paper can report
            # what the print chain is worth by removing it and nothing else.
            x = x + torch.randn_like(x) * N.EOT_NOISE_STD
            return x.clamp(0, 1)

        p = params or self.sample(B)

        # --- printer: dot gain spreads ink beyond the commanded dot
        x = gaussian_blur(x, p["dot_gain"])

        # --- ink and substrate: tone response, then loss of saturation.
        # Order matters. Gamma is a property of how much ink lands; the
        # saturation loss is a property of the ink itself being less pure than
        # a monitor primary. Doing saturation first would make the tone curve
        # act on a colour the printer never produced.
        x = x.clamp(min=1e-6) ** p["gamma"]
        g = rgb_to_gray(x)
        x = g + (x - g) * p["sat"]

        # --- the sheet in the world: shading across it, specular off it
        shadow = _low_freq_field(B, H, W, x.device, scale=4)
        x = x * (1.0 - p["shadow_amt"] * shadow)
        sheen = _low_freq_field(B, H, W, x.device, scale=3)
        x = x + p["sheen_amt"] * sheen * (1.0 - x)

        # --- illuminant and exposure
        x = x * p["wb"] * p["exposure"]

        # --- sensor
        x = dct_lowpass(x, N.PC_JPEG_LIKE)
        x = x + torch.randn_like(x) * N.EOT_NOISE_STD
        return x.clamp(0, 1)

    def describe(self):
        if not self.enabled:
            return "print chain OFF (digital EOT only, ablation)"
        return ("print chain ON: dot gain %.1f-%.1f px, gamma %.2f-%.2f, "
                "sat %.2f-%.2f, sheen<=%.2f, shadow<=%.2f, wb+-%.0f%%, "
                "exposure %.2f-%.2f, jpeg %.2f"
                % (N.PC_DOT_GAIN + N.PC_GAMMA + N.PC_SATURATION
                   + (N.PC_SHEEN, N.PC_SHADOW, N.PC_WHITE_BALANCE * 100)
                   + N.PC_EXPOSURE + (N.PC_JPEG_LIKE,)))


class PrinterLUT:
    """
    The printer's colour response, measured: requested sRGB -> the colour the
    scanner read back off the printed ink chart (physdecal inkchart). The chart is
    a 6 x 6 x 6 grid, so this is a trilinear 3D lookup, differentiable and
    deterministic. Requests outside the grid's span take the nearest edge.

    Unlike snapping to a handful of inks, this keeps continuous tone, which is
    what an inkjet or laser printer actually delivers by halftoning. The patch
    stays the REQUESTED colours (what is sent to the printer); the victim sees
    lut(patch), what comes out of it.
    """

    def __init__(self, requested_csv=None, measured_csv=None, device=None):
        import os
        import numpy as np
        ref = N.REFERENCE_DIR
        req = np.loadtxt(requested_csv or os.path.join(ref, "printable_requested.csv"),
                         delimiter=",", dtype=np.float32)
        mea = np.loadtxt(measured_csv or os.path.join(ref, "printable.csv"),
                         delimiter=",", dtype=np.float32)
        levels = np.unique(req[:, 0])
        n = len(levels)
        # Rows must be the full grid, blue fastest, so they reshape to r,g,b.
        grid = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), -1)
        if len(req) != n ** 3 or not np.allclose(req, grid.reshape(-1, 3), atol=1e-3):
            raise SystemExit("printer LUT: requested chart is not a full "
                             "r,g,b grid in blue-fastest order")
        self.lo, self.hi = float(levels[0]), float(levels[-1])
        vol = torch.from_numpy(mea.reshape(n, n, n, 3)).permute(3, 0, 1, 2)
        self.vol = vol.unsqueeze(0).to(device or N.DEVICE)    # 1 x 3 x R x G x B

    def __call__(self, patch):
        """patch B x 3 x H x W in [0,1] -> the printed colours, same shape."""
        B, _, H, W = patch.shape
        t = ((patch.clamp(self.lo, self.hi) - self.lo) / (self.hi - self.lo)) * 2 - 1
        # grid_sample's last axis is (x, y, z) = (B, G, R) of the volume
        g = t.permute(0, 2, 3, 1).flip(-1).reshape(B, 1, H, W, 3)
        out = F.grid_sample(self.vol.expand(B, -1, -1, -1, -1), g,
                            mode="bilinear", padding_mode="border",
                            align_corners=True)
        return out.reshape(B, 3, H, W)


def main():
    # Renders the chain applied to the anchor design, so you can see what a
    # printed decal is expected to look like before you print one.
    import argparse
    import os

    import cv2
    import numpy as np

    from physdecal.core import decal as D

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--design", default=None)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    torch.manual_seed(N.SEED)
    chain = PrintChain(device=a.device)
    print(chain.describe())

    anchor = D.anchor_image(a.design, device=a.device)
    draws = [anchor] + [chain(anchor) for _ in range(a.n)]
    row = np.concatenate([(t[0].permute(1, 2, 0).cpu().numpy() * 255)
                          .astype(np.uint8)[:, :, ::-1] for t in draws], axis=1)
    os.makedirs(N.FIGURE_DIR, exist_ok=True)
    out = os.path.join(N.FIGURE_DIR, "printchain_draws.png")
    cv2.imwrite(out, row)
    print("wrote %s  (leftmost is the file, the rest are draws)" % out)


if __name__ == "__main__":
    main()
