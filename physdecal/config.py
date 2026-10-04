"""
config.py
The only file in the PhysDecal experiment you edit by hand.

Two rules keep this experiment from contaminating anything else, and both are
enforced here rather than by discipline:

  1. Every directory this experiment writes to sits inside this repository
     (out/, weights/, data/reference/), never under the dataset root. The
     import fails if any of them ever resolves inside the dataset.
  2. The dataset is opened READ ONLY. Its location comes from the environment
     variable PHYSDECAL_DATA_ROOT, falling back to the sibling checkout of the
     GeoPatchCity capture project.

Archive this file with your results. It is the provenance record for a run.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Outputs, weights and reference data are resolved against the checkout, so
# the package must run from it (pip install -e .), never from site-packages.
if not os.path.isfile(os.path.join(REPO_ROOT, "pyproject.toml")):
    raise SystemExit(
        "physdecal is not running from its repository checkout (%s). Install "
        "it with  pip install -e .  from the repository root." % REPO_ROOT)

# ============================================================ 0. DATASET
# The GeoPatchCity capture (AirSim, Downtown West, Vehicle Variety Pack v1).
# Only the values this experiment reads are mirrored here from the capture
# pipeline's gp_config.py, so the repository runs without that pipeline on the
# path. If the capture is ever re-made with a different camera or fleet, these
# must be updated to match its gp_config.py.
DATA_ROOT = os.path.normpath(os.environ.get(
    "PHYSDECAL_DATA_ROOT",
    os.path.join(REPO_ROOT, os.pardir, "SegDet-GeoPatchCity", "airsim_city_v1")))

IMAGES_DIR = os.path.join(DATA_ROOT, "images")
SEG_DIR = os.path.join(DATA_ROOT, "segmentation")
GEOM_DIR = os.path.join(DATA_ROOT, "geometry")
META_CSV = os.path.join(DATA_ROOT, "frames.csv")
PALETTE_JSON = os.path.join(DATA_ROOT, "palette.json")

IMAGE_WIDTH = 1280             # capture resolution, must match settings.json
IMAGE_HEIGHT = 720

# AUTHORED 3D box dimensions at actor scale 1.0, metres, keyed by a substring
# of the actor name: (length, width, height, offset_forward, offset_right,
# offset_up). patch.roof_metric_dims takes the width-to-length ratio from
# here; the length itself is logged per instance with the actor scale applied.
VEHICLE_DIMS = {
    "Hatchback":   (3.94, 1.80, 1.23, 0, 0, 0),
    "SportsCar":   (4.36, 2.24, 1.13, 0, 0, 0),
    "SUV":         (4.21, 1.96, 1.66, 0, 0, 0),
    "Pickup":      (4.29, 1.91, 1.42, 0, 0, 0),
    "BoxTruck":    (7.32, 3.06, 3.18, 0, 0, 0),
    "TruckCab":    (7.32, 3.06, 3.18, 0, 0, 0),
}
DEFAULT_DIMS = (4.60, 1.90, 1.50, 0.0, 0.0, 0.0)


def dims_for(name):
    for key, dims in VEHICLE_DIMS.items():
        if key.lower() in name.lower():
            return dims
    return DEFAULT_DIMS


# ============================================================ 1. PATHS
# PHYSDECAL_OUT_ROOT moves run outputs (patches, results, figures, ...) off
# the repository, e.g. to a larger drive. Finetuned weights and the measured
# reference data stay in the repository because they are tracked.
EXP_ROOT = os.path.normpath(os.environ.get("PHYSDECAL_OUT_ROOT",
                                           os.path.join(REPO_ROOT, "out")))

PATCH_DIR     = os.path.join(EXP_ROOT, "patches")     # optimised patches
RESULT_DIR    = os.path.join(EXP_ROOT, "results")     # per-instance CSVs
FIGURE_DIR    = os.path.join(EXP_ROOT, "figures")     # panels and plots
CACHE_DIR     = os.path.join(EXP_ROOT, "cache")       # index files
LOG_DIR       = os.path.join(EXP_ROOT, "logs")
PRINT_DIR     = os.path.join(EXP_ROOT, "print")       # print-ready sheets
PHYSICAL_DIR  = os.path.join(EXP_ROOT, "physical")    # photographs and scores
FINETUNE_DIR  = os.path.join(REPO_ROOT, "weights")    # decoder + detector heads
REFERENCE_DIR = os.path.join(REPO_ROOT, "data", "reference")  # inks, designs

# A guard, not a comment. If any output directory ever ends up inside the
# dataset the import fails instead of overwriting somebody's results.
for _d in (EXP_ROOT, FINETUNE_DIR, REFERENCE_DIR):
    if os.path.normcase(os.path.abspath(_d)).startswith(
            os.path.normcase(os.path.abspath(DATA_ROOT)) + os.sep):
        raise SystemExit(
            "%s is inside the dataset root (%s). This experiment must never "
            "write into the dataset." % (_d, DATA_ROOT))
del _d

# ============================================================ 2. DEVICE
DEVICE = "cuda"
AMP = True
SEED = 20260911

# ============================================================ 3. DATASET VIEW
# Optimisation happens at nadir only; every oblique frame is a held-out
# generalisation test. Inherited from paper2 and still the right call.
NADIR_MAX_THETA_DEG = 8.0
PRIMARY_ONLY = True
MIN_INSTANCE_PX = 900
MIN_ROOF_QUAD_PX = 250

SPLIT_TRAIN = os.path.join(DATA_ROOT, "splits", "frames_train.csv")
SPLIT_HOLDOUT = os.path.join(DATA_ROOT, "splits", "frames_holdout.csv")

# ============================================================ 4. VICTIMS
# --- segmentation -----------------------------------------------------
# key -> (family, checkpoint, attention_scope, label_space)
MODELS = {
    "fcn_r50":            ("torchvision", "fcn_resnet50", "none", "voc"),
    "deeplabv3_r101":     ("torchvision", "deeplabv3_resnet101", "none", "voc"),
    "upernet_convnext_t": ("hf_upernet", "openmmlab/upernet-convnext-tiny",
                           "none", "ade20k"),
    "upernet_swin_t":     ("hf_upernet", "openmmlab/upernet-swin-tiny",
                           "local", "ade20k"),
    "segformer_b0":       ("hf_segformer",
                           "nvidia/segformer-b0-finetuned-ade-512-512",
                           "global", "ade20k"),
    "segformer_b2":       ("hf_segformer",
                           "nvidia/segformer-b2-finetuned-ade-512-512",
                           "global", "ade20k"),
    "clipseg":            ("hf_clipseg", "CIDAS/clipseg-rd64-refined",
                           "global", "prompt"),
}
ATTACK_MODELS = ["segformer_b0", "upernet_convnext_t", "upernet_swin_t"]
EVAL_MODELS = list(MODELS.keys())
CLIPSEG_PROMPTS = ["a car", "a vehicle seen from above", "a parked car"]

# --- detection --------------------------------------------------------
# The detection arm is a first-class victim in this study, not a stretch item.
# retinanet is the white-box arm because its dense classification head exposes
# a per-anchor vehicle logit the attack can descend directly. Faster R-CNN is
# scored as a transfer target only: a two-stage detector scores sampled
# proposals, which makes the gradient sparse and the comparison unfair to call
# white box.
DET_MODELS = {
    "retinanet":  ("torchvision", "retinanet_resnet50_fpn_v2", "dense"),
    "fasterrcnn": ("torchvision", "fasterrcnn_resnet50_fpn_v2", "two_stage"),
    "fcos":       ("torchvision", "fcos_resnet50_fpn", "dense"),
    # YOLO is the detector an operational aerial pipeline is most likely to be
    # running, so its absence from a roster invites the obvious question. It is
    # a TRANSFER TARGET ONLY: Ultralytics wraps its forward pass in a predictor
    # with letterboxing and NMS inside, and reaching a differentiable dense map
    # through that is a different piece of engineering with its own failure
    # modes. Calling it white box without that work would be a claim the code
    # does not support.
    "yolo":       ("ultralytics", "yolov8n", "black_box"),
}
DET_ATTACK_MODELS = ["retinanet"]
DET_EVAL_MODELS = list(DET_MODELS.keys())
DET_SCORE = 0.50               # below this is not a detection
DET_IOU_MATCH = 0.50           # box IoU that counts as finding the vehicle

# ============================================================ 5. INFERENCE
INPUT_LONG_SIDE = 1024         # -> 1024 x 576 after rounding to a multiple of 32
DET_INPUT_LONG_SIDE = 1024
CLEAN_IOU_GATE = 0.35
# paper2 carried an ROI-crop protocol as a fallback for the nadir domain gap.
# It is deliberately not reimplemented here: it was measured to do nothing at
# nadir, which is the only viewpoint the patch is optimised at, and head
# finetuning solved the gap outright. Scoring is full frame, one protocol, so
# no table needs a footnote saying which one produced it.

# ============================================================ 6. PLACEMENT
# ---------------------------------------------------------------------------
# HOW BIG THE DECAL IS, AND WHY THERE ARE TWO MODES.
#
# "fixed" pastes the same metric square on every vehicle. It is the mode every
# patch paper uses and it is what paper2 used. It is also misleading across a
# mixed fleet, and this was measured: a 2.0 m square scores nadir ASR 0.49 to
# 0.62 on a Pickup and 0.035 on a BoxTruck. The apparent win on the Pickup is
# an artefact -- 2.0 m OVERHANGS a pickup's 1.32 m roof panel onto the bed, so
# the simulation was pasting a decal no operator could stick down, while the
# BoxTruck, the one vehicle a 2.0 m decal genuinely fits, resisted it.
#
# The variable that actually predicts ASR is the fraction of the vehicle the
# decal covers, not its absolute size:
#
#     Pickup    4.0 m2 decal /  8.2 m2 footprint = 49 percent -> ASR 0.49
#     BoxTruck  4.0 m2 decal / 22.4 m2 footprint = 18 percent -> ASR 0.035
#
# "roof" therefore sizes the decal per vehicle, to the largest square that
# fits that vehicle's own roof panel. Two consequences, both wanted: every
# instance carries a decal an operator could physically apply, and the
# reported number becomes a function of a coverage fraction that is comparable
# across classes. Because roof panels scale with vehicles, the deployable
# coverage lands near 20 percent for every class, which is the physical
# ceiling on this attack and is a result in its own right.
PATCH_SIZE_MODE = "fixed"      # "fixed" | "roof"
PATCH_SIZE_M = 1.2             # metric side of the square, mode="fixed" only
PATCH_SIZE_SWEEP = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

# Roof panel as a fraction of the 3D bounding box, used by mode="roof".
# Calibrated against the dataset's own dimensions: a 3.94 x 1.80 m hatchback
# has roughly a 1.21 x 1.35 m roof panel. The same ratios give 2.25 x 2.29 m
# for the 7.32 x 3.06 m BoxTruck, which is the right order for a box body.
ROOF_LEN_FRAC = 0.307
ROOF_WID_FRAC = 0.75
ROOF_FILL = 0.95               # how much of that panel the decal occupies.
                               # 0.95 leaves a 2-3 cm border, which is what
                               # sticking a decal by hand actually achieves.

# Optional upper cap on the decal in mode "roof", in metres. None means every
# vehicle gets its own roof maximum, which on this dataset is 1.25 m for a
# Pickup and 2.13 m for a BoxTruck.
#
# Set it to keep the decal large where the roof allows and shrink it only
# where the roof does not: ROOF_MAX_M = 2.0 gives 2.0 m on the BoxTruck and
# 1.25 m on the Pickup. The reason to cap at all is that a decal has to be
# printed, carried and applied unnoticed, so an arbitrarily large one is not
# free even when the roof would hold it. --size overrides this on the command
# line, where in mode "roof" it is read as the cap.
ROOF_MAX_M = None

# --------------------------------------------------------------- multi-panel
# ---------------------------------------------------------------------------
# COVERAGE IS CAPPED PER PANEL, NOT PER VEHICLE, and that is the route past
# the ~20 percent single-panel ceiling without a millimetre of overhang.
#
# A vehicle offers several flat upper surfaces an operator can reach and a
# nadir camera can see. Each takes a decal that fits it; together they reach
# 50 to 70 percent coverage. The measured curve says 49 percent coverage
# scores ASR 0.49, so that is the regime where this attack works.
#
# THIS IS A CHANGE TO THE THREAT MODEL and must be reported as one: several
# decals of one design, not a single larger decal. Report coverage fraction
# with a per-panel fits flag, never a decal count -- the 2.0 m result failed
# precisely because one piece exceeded one surface, and three fitting pieces
# do not share that defect. Collapsing the two in the write-up invites exactly
# the criticism that sank the 2.0 m number.
#
# The SAME optimised texture goes on every panel. That is the honest attacker:
# print one design three times. Per-panel textures would triple the degrees of
# freedom and are left as an ablation, because "three different printed
# designs" is a larger capability than the paper needs to claim.
PATCH_PANELS = ["roof"]        # default is the single-panel threat model, so
                               # every earlier result reproduces unchanged.
                               # Multi-surface: ["roof", "bonnet", "boot"]

# Panel layout, in roof-plane coordinates of the 3D box top face:
#   name: (v_centre, length_frac, width_frac, height_frac_of_box)
# v runs 0 at the nose to 1 at the tail. height_frac 1.0 is the roof plane;
# lower values place the decal on its own lower plane, interpolated toward the
# box floor, which matters at oblique angles: a 0.33 m height error displaces
# a decal by 0.57 m on the ground at 60 degrees off nadir.
PANEL_LAYOUT_CAR = {
    "roof":   (0.500, 0.307, 0.75, 1.00),
    "bonnet": (0.170, 0.220, 0.70, 0.72),
    "boot":   (0.840, 0.200, 0.70, 0.78),
}
# A box-bodied vehicle has one flat top the whole way along, so its panels are
# tiles of that top at roof height rather than a bonnet and a boot.
PANEL_LAYOUT_BOX = {
    "roof":   (0.500, 0.280, 0.85, 1.00),
    "bonnet": (0.180, 0.280, 0.85, 1.00),
    "boot":   (0.820, 0.280, 0.85, 1.00),
}
# Which layout a vehicle gets is decided by its own proportions, not by its
# name: a tall short box is box-bodied, a long low one is a car. Data-driven,
# so it needs no per-vehicle table and extends to a fleet this study has not
# seen.
BOX_BODY_HGT_LEN = 0.35
PATCH_RES = 256                # texture resolution. 256 across 1.2 m is
                               # 4.7 mm/px, finer than a 300 dpi print holds,
                               # so the print is never the limiting resolution.
PATCH_CENTRE_UV = (0.5, 0.5)
# THE PATCH SURFACE IS THE ROOF, ALWAYS, and it is fixed before optimisation
# rather than selected per viewpoint. There is deliberately no setting for it:
# a surface chosen as a function of the look-down angle would make the
# transfer result a property of the choice rather than of the patch.

# ============================================================ 7. ATTACK
# Optimisation length. One epoch is one pass over the nadir train instances
# (len(index) // ATTACK_BATCH steps). ATTACK_STEPS, or --steps, overrides the
# epoch count with a fixed step budget; every patch reported in the paper was
# optimised for 3000 steps, and the experiments/*.sh scripts still pass that.
ATTACK_EPOCHS = 50
ATTACK_STEPS = None
ATTACK_BATCH = 2
ATTACK_LR = 0.02
# How the level-4 patch is parametrised.
#   "rgb_snap"    a free RGB texture behind the straight-through projection.
#                 What every reported patch used, so it stays the default.
#   "ink_logits"  one logit per (region, ink): decal.InkRegionTexture
LEVEL4_PARAM = "rgb_snap"
INK_LOGIT_LR = 0.10            # learning rate when LEVEL4_PARAM = "ink_logits"
# The objective is vanishing suppression, and it is the only one implemented.
# paper2 carried a "targeted" mode that pushed toward a road class; finetuning
# the head to two classes deletes the road class, so that mode has nowhere to
# push and is gone rather than left as a setting that raises.
ATTACK_TAU = 0.10              # hinge floor on the vehicle posterior
# The segmentation attack term. See losses.seg_attack_loss.
#   "vanish_prob"  hinge on the vehicle POSTERIOR over every ground truth
#                  vehicle pixel. What every reported patch was optimised
#                  with (paper Eq. segloss), so it stays the default.
#   "vanish"       the same pixels, hinged on the vehicle LOG-ODDS, which
#                  does not starve confident pixels of gradient
#   "vanish_away"  log-odds, only vehicle pixels OUTSIDE the decal footprint,
#                  so occlusion earns nothing
#   "untargeted"   log-odds, every pixel pushed off its true label, both ways
#   "ce_targeted"  unhinged cross entropy to an all-background target over
#                  the whole frame (supervisor's formulation)
#   "ce_untargeted" minus the cross entropy to the ground truth
# Measured on segformer_b0 at 1000 steps, held-out ASR 0.147 / 0.136 / 0.019
# / 0.000 respectively (paper, objective verification section).
SEG_OBJECTIVE = "vanish_prob"
DET_ATTACK_TAU = 0.10          # hinge floor on the per-anchor vehicle score
ATTACK_TASK = "joint"          # "seg" | "det" | "joint"
JOINT_DET_WEIGHT = 1.0         # weight on the detection term when task=joint

# ============================================================ 8. REALISM
# ---------------------------------------------------------------------------
# This block is the methodological change from paper2 and the reason this
# project exists.
#
# paper2 optimised free pixels and then ASKED them to look benign with soft
# LPIPS and Gram penalties. Measured outcome: raising the realism multiplier
# from 0 to 16 turned television static into slightly blurrier television
# static. A soft penalty on a 3 x 256 x 256 free variable is a suggestion, and
# the attack term outbids it at every pixel.
#
# Here realism is a CONSTRAINT, not a penalty. The patch is projected onto a
# decal manifold after every optimiser step, so every iterate -- not just the
# converged one -- is something a printer can reproduce and a person would
# read as a sticker. The soft terms remain, but they now shape a variable
# already inside the feasible set instead of fighting to drag it there.
#
# The projection is four operators, applied in this order (decal.py):
#   band     low-pass to the frequency a printer can print and a camera at the
#            operating GSD can resolve. Kills the pixel-level static outright.
#   flat     edge-aware piecewise-constant projection. Vinyl is flat areas
#            with crisp boundaries, not a gradient field.
#   palette  snap to K measured inks in CIELAB, straight-through gradient.
#   anchor   clamp into a CIEDE2000 ball around a named reference design, so
#            the result still reads as the thing it is pretending to be.
# ---------------------------------------------------------------------------

# One knob for the whole realism axis. This is the x of the RQ4 trade-off
# curve, and unlike a loss multiplier it changes the feasible set, so moving
# it is guaranteed to change the patch.
#   0  unconstrained free pixels          (the paper2 baseline, for comparison)
#   1  band limited only
#   2  band + flat
#   3  band + flat + palette              (printable decal)
#   4  band + flat + palette + anchor     (anchored printable decal, default)
REALISM_LEVEL = 4
REALISM_SWEEP = [0, 1, 2, 3, 4]

# --- projection parameters
PRINT_DPI = 300                # the printer you will actually use

# Ground sample distance the band limit is computed against, metres per image
# pixel. This is the parameter that decides how coarse the decal has to be,
# and it is a real design choice, not a constant:
#
#   the nadir optimisation set of this dataset spans 0.028 (20 m) to 0.087
#   (62 m) m/px, median 0.056.
#
# Setting it to the FINEST GSD lets the optimiser keep texture that survives
# at 20 m and is erased by 60 m. Setting it to the COARSEST forces a patch
# coarse enough to survive the whole altitude range, at some cost in attack
# strength close in. The median is the default because the evaluation is
# reported over the whole range and a patch tuned to one end of it would be
# reported at the other.
BAND_GSD_M_PER_PX = 0.056
BAND_GSD_SWEEP = [0.028, 0.056, 0.087]
INK_COUNT = 8                  # colours in the final decal. Real vinyl prints
                               # a spot-colour design in a handful of inks.
FLAT_ITERS = 3                 # guided-filter passes in the flatness projection
FLAT_SIGMA_COLOR = 0.12        # edge-preservation threshold of that filter
# How finely the anchor's regions are subdivided. The patch must be constant
# on each (design colour x supercell) region, so this is the knob that trades
# recognisability against attack capacity, and it is the honest x-axis for the
# realism curve inside level 4:
#   4  -> tens of flat areas, unmistakably the design, weakest attack
#   8  -> the default
#   16 -> hundreds of areas, still crisp and printable, starts to read as a
#         patterned decal rather than a specific marking
ANCHOR_SUPERCELL = 8
ANCHOR_SUPERCELL_SWEEP = [4, 8, 16, 32]

# Soft CIEDE2000 budget toward the anchor's own colours. This is a PENALTY,
# not a constraint: the layout is already pinned, so this only decides whether
# the recoloured design keeps the reference palette or drifts to whatever ink
# attacks best. Free inside the budget, penalised beyond it.
ANCHOR_DELTA_E = 22.0

# --- the named reference design the patch is anchored to.
# Procedurally generated by decal.py so the experiment has no external
# asset dependency and anyone can reproduce it exactly.
#   "sunroof"   a dark glass panel with a light surround
#   "cargo"     a courier-style cargo marking with a chevron block
#   "helipad"   an H landing marking
#   "solar"     a roof solar panel array
#   "rack"      a roof rack with cross bars
ANCHOR_DESIGN = "cargo"
ANCHOR_DESIGN_SWEEP = ["sunroof", "cargo", "helipad", "solar", "rack"]

# --- soft terms. These now shape a feasible iterate, they do not create one.
W_INK = 0.10       # soft-min CIEDE2000 to the nearest printable ink
W_FLAT = 0.05      # anchored anisotropic TV: flat EXCEPT where the anchor has
                   # an edge, so the design's own boundaries survive
W_ANCHOR = 0.20    # hinged CIEDE2000 to the anchor, free inside the ball
W_GAMUT = 0.05     # penalty for leaving the measured gamut
W_NPS = 0.02       # Sharif et al. NPS, kept only so the number is comparable
                   # with the prior literature. It is not load bearing here.

# The ink set: data/reference/printable.csv is a scan of the printed ink chart
# (`physdecal inkchart`), and printable_requested.csv is the 6x6x6 cube that
# was sent to the printer. Re-measure for a different printer or paper before
# reporting a printability number.
PRINTABLE_COLOURS_CSV = os.path.join(REFERENCE_DIR, "printable.csv")
INK_MEASURED = True            # False while PRINTABLE_COLOURS_CSV is not a
                               # scan of your printer. report.py refuses to
                               # print an NPS number while this is False.

# ============================================================ 9. PRINT CHAIN
# ---------------------------------------------------------------------------
# What happens between "a tensor" and "pixels a model sees" when the patch is
# physically printed. paper2's EOT jittered brightness, contrast, noise, blur,
# rotation and scale, which is the digital-compositing EOT every patch paper
# uses. None of those terms model a printer or a sheet of paper, and that is
# the gap physical results fall into.
#
# Every parameter below is sampled per EOT draw during optimisation, so the
# patch is optimised over the distribution of things printing and
# photographing actually do to it.
# ---------------------------------------------------------------------------
PRINT_CHAIN = True
PC_DOT_GAIN = (0.6, 1.4)       # printer MTF, Gaussian sigma in patch pixels
PC_GAMMA = (0.85, 1.20)        # ink + substrate tone response
PC_SATURATION = (0.80, 1.00)   # inks are less saturated than a monitor
PC_WHITE_BALANCE = 0.06        # per-channel illuminant cast, +/- fraction
PC_SHEEN = 0.10                # low-frequency specular highlight off the sheet
PC_SHADOW = 0.08               # low-frequency shading across the decal
PC_EXPOSURE = (0.85, 1.15)     # multiplicative, the whole decal
PC_JPEG_LIKE = 0.5             # strength of the DCT-domain low pass, 0 to 1

# --- geometric EOT, as before, on the patch only and never on the frame
EOT_NOISE_STD = 0.01
EOT_ROTATION_DEG = 1.5
EOT_SCALE = (0.97, 1.03)
EOT_TRANSLATE_FRAC = 0.02      # a decal is not pasted exactly where planned

# ============================================================ 10. PHYSICAL
# ---------------------------------------------------------------------------
# The print, paste and photograph loop. Three targets are supported and the
# point of the scale table is that they are metrically equivalent: the patch
# occupies the same fraction of the roof and subtends the same angle at the
# camera, so a number measured on a die-cast model is a number about the real
# geometry and not a number about a toy.
# ---------------------------------------------------------------------------
# ROOF PANEL, NOT THE BOUNDING BOX TOP FACE. These two are different
# surfaces and the difference is a real fidelity gap between the simulation
# and the physical experiment, so it is written down here rather than left to
# be discovered.
#
# patch.roof_metric_dims returns the vehicle's full length and width, so in
# simulation the patch is placed on the TOP FACE OF THE 3D BOUNDING BOX --
# about 4.0 x 1.8 m for a hatchback. A 1.2 m square centred on that face
# extends over the windscreen and the rear glass, which is a surface you can
# composite onto and cannot stick paper to.
#
# The numbers below are the flat sheet-metal ROOF PANEL, which is what a
# physical decal actually goes on: roughly a third of the vehicle length and
# three quarters of its width. The guard in export_print.target_geometry
# refuses to export a decal larger than this, because a decal that overhangs
# the roof is not a scaled version of the attack the simulation scored.
#
# The consequence to state in the paper: at PATCH_SIZE_M = 1.2 m the decal
# occupies most of the available roof panel. Report the PATCH_SIZE_SWEEP
# alongside it so a reader can see the attack at sizes an operator would find
# less conspicuous.
PHYS_TARGETS = {
    # key:      (scale, vehicle_len_m, roof_len_m, roof_wid_m, note)
    "diecast":  (1 / 18.0, 4.50, 1.38, 1.35, "1:18 die-cast model, tabletop"),
    "rc":       (1 / 10.0, 4.50, 1.38, 1.35, "1:10 RC car, floor or driveway"),
    "real":     (1.0,      4.50, 1.38, 1.35, "full size car, roof decal"),

    # BOX-BODIED TARGETS. A 2.0 m decal does not fit any car roof panel, but it
    # fits a box truck's 2.25 x 2.29 m roof with room to spare, and that is the
    # one configuration measured where the attack BOTH works and fits: L_dof
    # scores nadir ASR 0.213 and every piece is inside the surface it sits on.
    # Dimensions are the dataset's own GP_Car_09_BoxTruck, 7.32 x 3.06 m, with
    # the same roof-panel fractions used for the cars.
    "truck_diecast": (1 / 18.0, 7.32, 2.25, 2.29,
                      "1:18 die-cast box truck or delivery van, tabletop"),
    "truck_rc":      (1 / 10.0, 7.32, 2.25, 2.29,
                      "1:10 model box truck"),
    "truck_real":    (1.0,      7.32, 2.25, 2.29,
                      "full size box truck or panel van, roof decal"),

    # MEASURED MODELS. The scale is not the box's claim, it is derived from the
    # roof that was actually measured: s = 0.95 * min(roof) / PATCH_SIZE_M, so
    # the decal fills the short side to a 5 percent border and nothing
    # overhangs. Roof metres below are the prototype those measurements imply,
    # which is what makes the fit guard meaningful. Coverage lands at 0.72 and
    # 0.74 against the 0.776 the simulation scored: the short side binds before
    # coverage does, and the gap must be reported with the number.
    # Measured bodies: A is 250 x 80 mm with a 100 x 80 mm roof, B is
    # 170 x 75 mm whose sides taper to a 55 x 45 mm flat top. A's box roof
    # runs the full body width (1.00) and B's tapers well inside it (0.60),
    # so the two BRACKET the dataset's 0.75 rather than either matching it.
    # The consequence to report: the scale the ROOF implies and the scale the
    # BODY implies disagree by -10 percent on A and +9 percent on B, so each
    # vehicle is photographed that much smaller or larger than a faithful
    # replica would be. They err in OPPOSITE directions, which is the useful
    # part -- a result that holds on both is not an artefact of the scale
    # bookkeeping.
    "measured_a": (1 / 26.316, 6.579, 2.632, 2.105,
                   "measured model A, 100 x 80 mm roof, 76.0 mm decal"),
    "measured_b": (1 / 46.784, 7.953, 2.573, 2.105,
                   "measured model B, 55 x 45 mm flat roof (sides taper), "
                   "42.8 mm decal"),
}
PHYS_TARGET = "diecast"

# Reference aerial geometry the physical capture reproduces. A photograph at
# scale s from altitude ALT * s sees the same thing the simulated camera saw
# at ALT, to within the camera's own intrinsics.
PHYS_REF_ALTITUDE_M = 20.0
PHYS_REF_THETA_DEG = [0, 15, 30, 45, 60]
PHYS_LIGHTING = ["indoor_diffuse", "indoor_directional", "daylight"]
PHYS_REPEATS = 3               # photographs per (target, theta, lighting) cell

# --- print sheet
SHEET = "A4"                   # "A4" | "A3" | "Letter". Set before printing.
SHEET_MM = {"A4": (210.0, 297.0), "A3": (297.0, 420.0),
            "Letter": (215.9, 279.4)}
SHEET_MARGIN_MM = 10.0
TILE_OVERLAP_MM = 5.0          # glued seam allowance on multi-sheet patches
FIDUCIAL = True                # ArUco border, recovers scale and pose per photo
FIDUCIAL_DICT = "DICT_4X4_50"
FIDUCIAL_MM = 20.0             # nominal marker side for the "real" target

# MARKER SIZE IS SET BY WHAT THE CAMERA CAN RESOLVE, NOT BY THE DECAL SCALE.
# This was measured rather than assumed. Scaling the marker with the target,
# which is the obvious thing to do, gives an 8 mm marker on the 1:18 sheet;
# photographed from the scaled-equivalent distance of 1.11 m it spans about 9
# pixels and ArUco decodes exactly none of them. A 4x4 marker needs 6 modules
# plus a quiet zone, so it wants roughly 40 pixels before detection is
# reliable.
#
# So the marker is sized backwards from the capture: given the distance, the
# camera's focal length in pixels and a floor on detected size, solve for the
# millimetres. At 1:18 that lands near 15 mm, which is larger than the 8 mm a
# naive scaling gives and larger than the decal's own border.
#
# The physical consequence, and it must be in the capture instructions: for
# the scaled targets the MARKER FRAME STAYS ON THE TABLE around the model car
# while only the decal is cut out and stuck to the roof. The markers were
# never meant to be on the vehicle; they establish the plane the vehicle sits
# on, and that plane is the table.
FIDUCIAL_MIN_PX = 40           # detected marker side, in pixels, below which
                               # ArUco stops being reliable
PHYS_PHOTO_LONG_SIDE = 4032    # long side of the photographs you will take.
                               # Set it to your actual camera before printing:
                               # it decides how big the markers have to be.
PRINT_CAPTION = True           # patch id, size, dpi, sheet index on every sheet

# --- physical scoring
# Horizontal field of view assumed when a photograph carries no EXIF focal
# length. About right for a modern phone's main camera. It affects the
# RECOVERED ANGLE only; it cannot affect whether the victim found the vehicle,
# so a wrong value mis-bins a row rather than inventing a success. Every row
# records which source was used.
PHYS_DEFAULT_FOV_DEG = 66.0

PHYS_MIN_MARKERS = 3           # fewer than this and the photo has no pose, so
                               # it is reported unscored rather than guessed
# The annotator always offers GrabCut from a dragged box; there is no setting
# to turn it off, because the manual brush is available in the same window.

# ============================================================ 11. REPORTING
REPORT_ANGLE_BINS = [(0, 10), (10, 20), (20, 30), (30, 40),
                     (40, 50), (50, 60), (60, 70)]
REPORT_ALT_BINS = [(19, 30), (30, 40), (40, 50), (50, 63)]
ASR_IOU_THRESHOLD = 0.50
ATTACKABLE_IOU = 0.50
# R50 and R90 are computed over all away-flips with no radius cap. A cap would
# be a second free parameter inside a number the paper reports in metres.

# Figure panels. These are the deliverable the study is judged on, so they are
# configured here rather than hardcoded in the plotting script.
PANEL_FRAMES = 4               # rows in a seg or det failure panel
PANEL_DPI = 160

# ============================================================ 12. FINETUNING
USE_FINETUNED = True
FINETUNE_EPOCHS = 6
FINETUNE_BATCH = 4
FINETUNE_LR = 3e-4
FINETUNE_VAL_FRACTION = 0.10
FINETUNE_MAX_FRAMES = 0
FINETUNE_WORKERS = 4
IGNORE_INDEX = 255

DET_FINETUNE_EPOCHS = 4
DET_FINETUNE_BATCH = 2
DET_FINETUNE_LR = 5e-4


# ============================================================ 13. BOOTSTRAP

def ensure_dirs():
    """Create every output folder. Called by the entry points, so a fresh
    checkout does not fail three hours into a run on a missing directory."""
    for d in (EXP_ROOT, PATCH_DIR, RESULT_DIR, FIGURE_DIR, CACHE_DIR,
              LOG_DIR, FINETUNE_DIR, REFERENCE_DIR, PRINT_DIR, PHYSICAL_DIR):
        os.makedirs(d, exist_ok=True)
