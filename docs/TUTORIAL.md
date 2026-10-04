# PhysDecal Tutorial

Everything from a clean checkout to a printed decal on a car roof and a number
that says whether the detector saw it.

Run every command from the repository root after `pip install -e .`.

> **This repository never writes into the dataset.** Run outputs go to
> `out/`, weights to `weights/`, measured reference data to `data/reference/`.
> The dataset is opened read only. `physdecal check` asserts this on every
> run — it is a check, not a promise.

---

## 0. The map

| Module | Command | What it is |
|---|---|---|
| `physdecal/config.py` | | **the only file you edit by hand** |
| `evaluation/check.py` | `physdecal check` | preflight and the eligibility gate |
| `core/data.py` | | dataset index, roof quads, loaders |
| `victims/segmentation.py` | | the seven segmentation victims |
| `victims/detection.py`, `victims/yolo.py` | `physdecal yolo` | the detection victims |
| `victims/finetune.py` | `physdecal finetune` | decoder-only and head-only finetuning, both arms |
| `core/decal.py` | `physdecal decal` | **the decal manifold and its projection** |
| `core/printchain.py` | `physdecal printchain` | **the differentiable print-and-photograph channel**, and the measured printer LUT |
| `core/losses.py` | | the objective |
| `core/patch.py` | | roof homography and compositing |
| `attack/optimize.py` | `physdecal optimize` | the attack |
| `evaluation/evaluate.py` | `physdecal evaluate` | digital scoring, both arms |
| `figures/panels.py` | `physdecal figures` | **the inference panels** |
| `physical/export_print.py` | `physdecal export-print` | **print-ready sheets with fiducials** |
| `physical/capture.py` | `physdecal physical` | **plan, ingest, annotate, score photographs** |
| `evaluation/report.py` | `physdecal report` | tables and the summary |

The bolded parts are the ones that did not exist in the predecessor study
(`paper2`, the free-pixel NadirPatch attack).

---

## 1. Preflight

```bash
physdecal check
```

Checks the environment, the dataset, the split, the decal manifold and which
finetuned heads exist. Every line is `ok`, `warn` or `FAIL`. Fix the `FAIL`s
before anything else; the `warn`s tell you what is not done yet.

A warning that is expected and is not a problem:

- **no head for some detectors.** Correct. See step 3. (The repository ships
  every finetuned head in `weights/` through Git LFS; run `git lfs pull` if
  they show up as tiny pointer files.)

---

## 2. Measure the gate before you do anything else

```bash
physdecal check --gate --published
```

This measures clean performance at nadir on published COCO and ADE20K weights.
Expect it to fail, and expect that to be the point:

```
[ FAIL ] seg segformer_b0        clean IoU 0.000, 0% attackable
[ FAIL ] seg upernet_convnext_t  clean IoU 0.000, 0% attackable
[ FAIL ] det retinanet           clean recall 0.017 at score 0.50
```

A victim that cannot see the vehicle cannot be attacked. The attackable set is
empty, so ASR is **undefined, not zero**. Record these numbers — they are the
measurement that justifies everything in step 3, and they belong in the paper.

---

## 3. Finetune the heads, backbones frozen

Segmentation decoders for the three white-box arms (about 30-50 min each on a
6 GB card):

```bash
physdecal finetune --model segformer_b0 upernet_convnext_t upernet_swin_t
physdecal finetune --model all          # the whole roster
```

Detection heads:

```bash
physdecal finetune --det retinanet
physdecal finetune --det all
```

> Decoders copied from an earlier run already sit in `weights/`.
> If `physdecal check` lists them with a validation IoU, you can skip the
> segmentation half of this step.

Confirm the gate now passes:

```bash
physdecal check --gate
```

```
[  ok  ] seg segformer_b0        clean IoU 0.778, 100% attackable
[  ok  ] seg upernet_convnext_t  clean IoU 0.900, 100% attackable
```

If a detector still fails the gate, train it longer before attacking it. An
attack on a victim with a 20 percent attackable set is a measurement of the
victim, not of the attack.

---

## 4. Look at the decal manifold before you optimise anything

```bash
physdecal decal --device cpu
physdecal printchain --device cpu
```

The first prints what each realism level does to a noise image — ink count,
static index, degrees of freedom. The second renders the anchor design as it
would look printed and photographed, to
`out/figures/printchain_draws.png`. Open it. If the draws do not look
like photographs of a printed sheet, tune `PC_*` in `config.py` before
spending GPU hours optimising through a channel that is wrong.

To change the design the decal is anchored to:

```python
ANCHOR_DESIGN = "cargo"     # sunroof | cargo | helipad | solar | rack
ANCHOR_SUPERCELL = 8        # 4 = unmistakably the design, 32 = a patterned decal
```

---

## 5. Optimise

**How long a run is.** By default a run is `ATTACK_EPOCHS` = 50 passes over
the nadir train instances (855 instances at batch 2 is 427 steps per epoch,
about 21,000 steps). `--epochs N` changes the count; `--steps N` sets a fixed
step budget instead and overrides it. Every patch reported in the paper used
`--steps 3000`, and the scripts in `experiments/` still pass that so they
reproduce the paper. `run.json` records `steps`, `epochs` and
`steps_per_epoch`.

The simple printable patch (the current direction: targeted cross entropy,
one universal patch, no reference design, optimised through the measured
printer response):

```bash
physdecal optimize --task seg --model segformer_b0 --realism 0 \
    --size 2.0 --size-mode fixed --seg-objective ce_targeted \
    --printer-lut --tag ce_lut_segformer_b0
physdecal evaluate --patch ce_lut_segformer_b0 --models segformer_b0 --printed
physdecal evaluate --patch ce_lut_segformer_b0 --models segformer_b0 --printed --baseline grey
```

The anchored decal — both arms at once, full realism:

```bash
physdecal optimize --task joint --model segformer_b0 --det retinanet \
    --tag main
```

The realism curve for RQ1, five runs:

```bash
physdecal optimize --task joint --sweep-realism --tag curve
```

The print-chain ablation for RQ2:

```bash
physdecal optimize --task joint --no-print-chain --tag nochain
```

Each run writes to `out/patches/<tag>/`:

```
patch.pt          the projected patch, which is what gets printed
patch.png         the same, at texture resolution
patch_large.png   8x nearest neighbour, for figures
run.json          every setting, plus the realism metrics
history.csv       the loss, per step
```

**Read the header the run prints.** It states the constraint set, the degrees
of freedom and the channel, and those three lines are what a reader will ask
you for:

```
patch      : 1.20 m, 256 px, realism level 4: band+flat+palette+anchor,
             band sigma 3.80 px (cutoff 0.042 cyc/texel), 11 inks,
             anchor 'cargo', 126 regions
free vars  : 378 of 196608 raw pixels
channel    : print chain ON: dot gain 0.6-1.4 px, gamma 0.85-1.20, ...
```

**Memory.** On a 6 GB card `--task joint` holds a segmenter and a detector at
once and is the first thing to fall over. Drop to `--batch 1`, or attack the
arms separately and score each on both.

---

## 6. Evaluate

```bash
physdecal evaluate --patch main --models all --detectors all
```

And the controls, which every ASR must be reported beside:

```bash
physdecal evaluate --patch main --models all --detectors all --baseline grey
physdecal evaluate --patch main --models all --detectors all --baseline random
physdecal evaluate --patch main --models all --detectors all --baseline anchor
```

The `anchor` control is the one the realism claim needs: it is the reference
decal, unoptimised. If it already suppresses the vehicle, the attack is not
what is doing the work.

The full holdout grid is 2,534 instances per victim. `--stride 4` gives the
same conclusions in a quarter of the time; if you use it, say so in the paper.

Each run prints the line that matters:

```
seg    segformer_b0    attackable 1180/2534 (47%)  ASR 0.612  mean IoU drop 0.39
```

If you ever see `attackable set EMPTY`, do not report a zero. Report undefined
and go back to step 3.

---

## 7. The inference figures

```bash
physdecal figures --patch main --panel both
physdecal figures --panel gallery --patches curve_r0 curve_r2 curve_r4
```

The segmentation panel is five columns — clean frame with ground truth, clean
prediction, patched frame, prediction under attack, and the vehicle pixels lost
split into under-patch (occlusion) and away (the adversarial effect). The
detection panel shows boxes and scores clean against attacked, with the lost
vehicles marked.

Each panel writes a `_caption.txt` beside it stating the **selection rule** —
by default the panel shows successes, and the caption says so in those words
along with how many instances they were drawn from. Paste that sentence into
the paper. For an unselected draw:

```bash
physdecal figures --patch main --panel both --select attackable
```

Or name exact frames and remove the choice entirely with `--frames <id> ...`.

---

## 8. Before you print: the ink set

The shipped `data/reference/printable.csv` is MEASURED: a scan of a 6x6x6
chart printed on the printer and paper used for the physical experiment.
`printable_requested.csv` holds the colours that were sent to the printer, and
the pair is what `--printer-lut` / `--printed` (`printchain.PrinterLUT`)
interpolate between. `inkchart/inkchart.json` is the chart layout needed to
read a scan back.

For a different printer or paper, measure again before making a
printability claim:

```bash
physdecal inkchart make --colours data/reference/printable_requested.csv
# print data/reference/inkchart/inkchart.pdf at 100 %, colour management OFF, scan it
physdecal inkchart read --scan scan.png
```

Pass `--colours` explicitly: without it `make` charts `printable.csv`, which
is now the measured set, not the requested grid. `read` writes the measured
colours to `printable.csv`. While `INK_MEASURED` is False, `physdecal report`
refuses to headline an NPS number. The attack works either way; the
*printability claim* is what is gated.

---

## 9. Export the print

Look at the scale table first:

```bash
physdecal export-print --scale-table --patch main
```

```
target      scale    vehicle      decal   roof frac   camera at    marker
diecast      1:18        250       66.7       0.773        1.11      14.3
rc           1:10        450      120.0       0.773        2.00      25.8
real          1:1       4500     1200.0       0.773       20.00     257.7
```

All three rows are metrically equivalent: the decal covers the same fraction of
the roof and subtends the same angle. "Camera at" is where you stand to
reproduce 20 m of altitude.

Then export:

```bash
physdecal export-print --patch main --target diecast
physdecal export-print --patch main --target real --sheet A3
```

You get `artwork.png`, numbered `sheet_NN.png`, a multi-page PDF at true page
size, and `print.json` carrying the marker pitch and decal size that
`physdecal physical` needs later.

**Printing rules, and they are not optional:**

1. **Print at 100 percent.** Turn off "fit to page". This is the most common
   way a physical experiment silently produces the wrong decal size.
2. **Measure the ruler on the sheet** before cutting. If it does not read what
   it says, the scale is wrong and every number after it is wrong too.
3. **Cut on the outer crop marks** so the ArUco markers stay attached.
4. **Matte media.** Gloss makes a specular highlight that saturates the sensor
   and destroys the decal in exactly the frames you most want.
5. The arrow marked `NOSE` points toward the vehicle's front.

If the export warns that markers will not be detected, read the four remedies
it prints. The usual one at scaled targets: print the marker frame as a
separate sheet at full size and lay it on the table around the model. The
markers establish the ground plane, not the roof.

The full-size target at A4 is 88 sheets. Use A3, a print shop, or a smaller
`PATCH_SIZE_M` — and note that the size sweep is a result worth reporting
anyway.

---

## 10. Capture

```bash
physdecal physical plan --patch main --target diecast
```

Writes `capture_plan.csv` (one row per photograph) and `HOW_TO_CAPTURE.txt`
into `out/physical/main__diecast/`. Read the second one; it is the
session checklist.

The protocol in one line: **paired shots**. For each pose, photograph the
vehicle without the decal, lay the decal on, photograph again from the same
position, remove it. Everything that could confound the comparison is shared
within the pair and cancels.

Names carry the pairing:

```
t20_indoor_diffuse_00__clean__00.jpg
t20_indoor_diffuse_00__patched__00.jpg
```

Lock exposure and focus if your phone allows it. Auto-exposure changing between
the two shots is the fastest way to stop a pair being a controlled comparison.

Set `PHYS_PHOTO_LONG_SIDE` in `config.py` to your camera's actual resolution
**before** exporting the print — it decides how big the markers have to be.

---

## 11. Score the photographs

Copy the photos into the session folder, then:

```bash
physdecal physical ingest   --dir out/physical/main__diecast
physdecal physical annotate --dir out/physical/main__diecast
physdecal physical score    --dir out/physical/main__diecast
physdecal physical panel    --dir out/physical/main__diecast
```

**ingest** detects the markers and recovers, per photograph, the nadir angle
and the camera distance. A clean shot has no markers on it by construction and
inherits its pair's pose, which is exactly right because the camera did not
move. Check the reported angle range covers what you meant to shoot.

**annotate** draws the vehicle mask, once per scene, shared by both members of
the pair. Drag a box, press `g` for GrabCut, paint with `f` and `b` to correct,
`s` to save. About ten seconds per scene.

**score** runs the victims and prints the physical table:

```
                                  n  attack.      ASR
segmentation                     30       27    0.481   mean IoU drop 0.402
detection                        30       24    0.375   mean score drop 0.284
```

Put those numbers next to the digital ones from step 6. **The gap between them
is the sim-to-real gap, and it is the most informative number in the paper.**

A photograph with fewer than `PHYS_MIN_MARKERS` readable markers is reported
`unscored` rather than guessed at.

---

## 12. Report

```bash
physdecal report
```

Collects every CSV under `out/results/` and every physical session into
LaTeX tables and summary plots under `out/figures/`.

---

## Troubleshooting

**The patch goes to NaN.** It should not — `decal.ciede2000` guards the
achromatic gradient trap that caused it, and every reference design is mostly
white and grey, so the trap is the common case here rather than a corner. If it
comes back, run `physdecal decal --device cpu` and check the gradient
finiteness self-test before looking anywhere else.

**Level 4 produces a patch identical to the anchor.** The feasible set has
collapsed. Raise `ANCHOR_SUPERCELL`; `physdecal check` warns when the region count
falls below 16.

**ASR is high but the patch looks like noise.** Check `REALISM_LEVEL`. Level 0
is the paper2 baseline and is *supposed* to look like noise — it is the control,
not the result.

**ASR is zero at level 4 and high at level 0.** That is RQ1's answer, not a
bug. Report the curve. Also check `degrees_of_freedom` in `run.json`: if it is
in the tens, raise the supercell before concluding the constraint is fatal.

**Markers are never detected.** Compare `fiducial_px_expected` in `print.json`
against `FIDUCIAL_MIN_PX`. Usually `PHYS_PHOTO_LONG_SIDE` is set below what
your camera actually shoots, or the print came out scaled.

**CUDA out of memory in `--task joint`.** Two victims at once. `--batch 1`, or
attack the arms separately.
