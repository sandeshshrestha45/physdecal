# PhysDecal

Printable adversarial roof decals against aerial **segmentation** and
**detection**, carried through to a physical measurement: optimise the patch,
print it at true scale with fiducials, put it on a vehicle, photograph it, and
score the photographs with the same metrics as the simulation.

Dataset: GeoPatchCity (AirSim, Downtown West, Vehicle Variety Pack v1).

## What is in here

- **Victims.** Seven segmentation models (FCN, DeepLabv3, UPerNet-ConvNeXt-T/Swin-T,
  SegFormer-B0/B2, CLIPSeg) and four detectors (RetinaNet, Faster R-CNN, FCOS,
  YOLOv8n). Heads are finetuned with frozen backbones, because no published
  checkpoint can see a nadir vehicle.
- **Attack.** One universal patch, optimised on nadir train frames, placed on
  the roof through the metric roof homography. Objectives: hinge (`vanish*`) or
  targeted / untargeted cross entropy, for `seg`, `det` or `joint`.
- **Realism as a constraint.** Patches are projected onto a decal manifold
  (band → flat → measured inks → anchor design, realism levels 0–4). Optimisation
  runs through a differentiable print-and-photograph chain and, optionally, the
  measured printer colour response (`--printer-lut`).
- **Evaluation.** Holdout vehicles over the full viewpoint grid; ASR over the
  attackable set only; under/away pixel split; grey, random and anchor controls;
  vehicle-level bootstrap intervals.
- **Physical loop.** Print sheets with ArUco markers, a capture plan, pose
  recovery from markers, mask annotation and tracking, and scoring of
  photograph pairs.

## The bright-colour study (`paper/`)

`paper/main.tex` is the paper built on this code. Its patches hold every
pixel at a bright, saturated colour (`optimize --bright`), because photographs
of the earlier printed patch showed it lost about half its lightness contrast
between print and camera (`physdecal photofit`). Every patch is optimised for
the 50-epoch default. Reproduce with

```bash
sh experiments/run_bright.sh > out/logs/run_bright.log 2>&1
physdecal fig-bright --out paper/figures
```

New pieces: `patch.BrightTexture` (`--bright`, `BRIGHT_S_MIN`/`BRIGHT_V_MIN`),
`evaluate --stress glare|dim|flat|desat|blur|photo` and `--baseline bright`,
`physdecal photofit` (writes `data/reference/photo_channel.json`),
`physdecal figures --full` (whole frames instead of vehicle crops), and
`physdecal fig-bright`.

## Install

```bash
git lfs install            # weights/ is tracked with Git LFS
git clone <repo-url> physdecal && cd physdecal
# install torch/torchvision for your CUDA version from pytorch.org first
pip install -e .           # editable install is REQUIRED (see below)
pip install -e ".[yolo,kornia]"   # optional: YOLO arm, faster warping
```

The package resolves `out/`, `weights/` and `data/` against the checkout, so it
refuses to run from a non-editable install. `requirements.txt` pins the
versions it was last run with (Python 3.11, torch 2.6, CUDA 12.4).

## Dataset
Dataset  link: https://drive.google.com/file/d/1u8yCl0wTjy8mGnqs9wLZOO5wDPeKv4Mm/view?usp=sharing
The GeoPatchCity capture is opened **read only**. Point the repository at it:

```bash
export PHYSDECAL_DATA_ROOT=/path/to/airsim_city_v1
```

Without the variable it defaults to `../SegDet-GeoPatchCity/airsim_city_v1`.
The experiment needs `images/`, `segmentation/`, `geometry/`, `frames.csv`,
`palette.json` and `splits/frames_{train,holdout}.csv`. The splits come from
`make_splits.py` in the capture pipeline. The YOLO mirror also reads
`labels/` and `splits/{train,holdout}.txt`. The importer refuses to start if
any output directory resolves inside the dataset.

## Workflow

Everything goes through one command: `physdecal <command>` (or
`python -m physdecal <command>`). `physdecal --help` lists them all.

```bash
physdecal check                      # 1. preflight
physdecal check --gate               #    clean performance: is anything attackable?
physdecal finetune --model all       # 2. only if weights/ is missing a head
physdecal finetune --det retinanet
physdecal yolo setup && physdecal yolo train

physdecal optimize --task seg --model segformer_b0 --realism 0 \
    --size 2.0 --size-mode fixed --seg-objective ce_targeted \
    --printer-lut --tag ce_lut_segformer_b0        # 3. 50 epochs by default

physdecal evaluate --patch ce_lut_segformer_b0 --models all --printed       # 4.
physdecal evaluate --patch ce_lut_segformer_b0 --models all --printed --baseline grey
physdecal report                     #    tables + realism / envelope plots
physdecal stats --patch ce_lut_segformer_b0__printed
physdecal viewpoint --patch ce_lut_segformer_b0__printed
physdecal figures --patch ce_lut_segformer_b0__printed --panel seg   # 5.
physdecal allfigs --patch ce_lut_segformer_b0

physdecal export-print --scale-table --patch ce_lut_segformer_b0     # 6.
physdecal export-print --patch ce_lut_segformer_b0 --target measured_a
physdecal physical plan --patch ce_lut_segformer_b0 --target measured_a
physdecal physical ingest   --dir out/physical/<session>
physdecal automask          --dir out/physical/<session>   # or: physical annotate
physdecal physical track    --dir out/physical/<session>
physdecal physical score    --dir out/physical/<session> --arm seg
physdecal physical panel    --dir out/physical/<session>
```

`docs/TUTORIAL.md` is the step-by-step runbook, with the reasoning behind each
step. `experiments/` holds the resumable scripts that produced the paper's
runs.

### Optimisation length

`physdecal optimize` runs `ATTACK_EPOCHS` = **50 epochs** by default. One epoch
is one pass over the 855 nadir training instances, so 427 steps at batch 2,
about 21,000 steps in all. Use `--epochs N` to change the count, or
`--steps N` for a fixed step budget, which overrides it. The paper's patches
used `--steps 3000`, and the `experiments/` scripts keep that.

## Commands

| Stage | Command | Purpose |
|---|---|---|
| setup | `check` | preflight, isolation, and the `--gate` eligibility measurement |
| | `finetune` | decoder-only (`--model`) / head-only (`--det`) finetuning |
| | `yolo` | YOLO dataset mirror (`setup`), `train`, `check` |
| | `decal`, `printchain` | manifold self-check; print-chain draws of the anchor |
| | `inkchart` | `make` a printable ink chart, `read` its scan into `printable.csv` |
| | `palettes` | the H1 ink sets (chroma / cardinality / random hue) |
| attack | `optimize` | universal patch; `--sweep-realism`, `--printer-lut`, `--palette-only`, `--level4-param`, seeds |
| | `objcheck` | per-frame vs universal patches: is the objective or universality the limit? |
| evaluate | `evaluate` | per-instance CSVs, both arms; `--baseline grey/random/anchor`, `--printed`, `--prompts` |
| | `report`, `stats`, `asr`, `viewpoint` | tables, bootstrap CIs, nadir/all ASR, transfer envelope |
| | `lighting`, `manifest` | lighting-label verification; `out/EXPERIMENTS.md` |
| figures | `figures`, `allfigs`, `gradcam` | inference panels (seg, det, compare, gallery), roster tables, Grad-CAM |
| | `fig-ce`, `fig-h1`, `fig-loss`, `fig-bright` | the papers' summary figures |
| physical | `export-print` | true-scale sheets with ArUco fiducials, multi-page PDF |
| | `physical` | `plan`, `ingest`, `annotate`, `track`, `score`, `panel`, `segpanel` |
| | `automask`, `altitude` | draft vehicle masks; fiducial-free altitude recovery |
| | `photofit` | register a printed patch in its photographs, fit the print+photo colour channel |

## Layout

```
physdecal/
  config.py        the only file you edit by hand: every setting, with why
  cli.py           the `physdecal` command
  core/            data, patch placement, decal manifold, print chain, losses
  victims/         segmentation and detection wrappers, YOLO, finetuning
  attack/          optimize, objcheck
  evaluation/      evaluate, check, report, stats, asr, viewpoint, lighting, manifest
  figures/         panels, allfigs, gradcam, paper figures
  physical/        export_print, capture, automask, altitude, inkchart, palettes
data/reference/    measured ink chart (printable.csv), the requested grid, H1 palettes
weights/           finetuned decoders and heads (Git LFS)
experiments/       resumable run scripts behind the paper
docs/TUTORIAL.md   the runbook
out/               every run output (gitignored); PHYSDECAL_OUT_ROOT moves it
```

## Outputs

| Path | Contents |
|---|---|
| `out/patches/<tag>/` | `patch.pt` (projected, what gets printed), `patch.png`, `patch_large.png`, `run.json`, `history.csv` |
| `out/results/` | `seg_<tag>[__<control>][__printed]__<victim>.csv`, one row per instance |
| `out/figures/` | panels with `_caption.txt` selection statements, reports, tables (`.txt`/`.tex`) |
| `out/print/<tag>__<target>/` | artwork, sheets, true-size PDF, `print.json` |
| `out/physical/<session>/` | photographs, `manifest.csv`, `annotations/`, `physical_results*.csv` |
| `out/logs/`, `out/cache/` | run transcripts; dataset index caches |

## Reporting rules built into the code

- ASR is computed over the **attackable set** only. An empty set is reported
  as *undefined*, never as zero.
- Every control is pasted at the **size of the patch it controls for**. That
  size is read from the patch's `run.json`, not from the config.
- Figures state their **selection rule** in a caption file written beside them.
- With three holdout vehicles, use the **vehicle bootstrap** (`physdecal stats`)
  for intervals.
