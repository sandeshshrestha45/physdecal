# Experiment scripts

The run scripts behind the paper's results, ported from the original working
folder. Each script is **resumable**: a step is skipped when its output
(`out/patches/<tag>/run.json` or the results CSV) already exists. They run
sequentially on purpose, because one GPU is assumed.

Run them from anywhere; each one `cd`s to the repository root first.

```bash
PYTHON=/path/to/venv/python sh experiments/run_ce.sh > out/logs/run_ce.log 2>&1
```

- `PYTHON` selects the interpreter (default: `python` on the PATH).
- `PHYSDECAL_OUT_ROOT` moves outputs, the same as for the package.

They pass `--steps 3000` explicitly, so they reproduce the paper's runs
unchanged by the 50-epoch default of `physdecal optimize`. To rerun a recipe
for 50 epochs, replace `--steps 3000` with `--epochs 50` and give the run a
new tag.

| Script | What it produces |
|---|---|
| `run_ce.sh` | targeted-CE simple patch on SegFormer-B0, with and without the print chain |
| `run_ce_ink.sh` | the same at realism level 3 with 32 measured inks. Waits for `CE DONE` in `out/logs/run_ce.log`, so run `run_ce.sh` with its output there |
| `run_ce_palette.sh` | printable-colour constraint only: snap vs per-pixel ink logits |
| `run_ce_lut.sh` | the printed patch, optimised through the measured printer LUT, scored `--printed` |
| `run_ce_seeds.sh` | seeds 101 and 202 for the LUT / free-pixel and CE / hinge claims |
| `run_ce_untargeted.sh` | the untargeted (-CE) form, three seeds |
| `run_ce_roster.sh` | transfer of the printed patches to every segmentation victim, plus ink-row seeds |
| `run_det_ce.sh` | the detection mirror on RetinaNet, with transfer to Faster R-CNN, FCOS and YOLO |
| `run_random_ctrl.sh` | random-noise control, printed, on every victim |
| `run_h1.sh` | H1 palette ablation (needs `physdecal palettes`; the CSVs ship in `data/reference/palettes/`) |
| `run_seed_roster.sh` | roster evaluation of the `seedtest_*` patches |
| `run_wb_fixed.sh` | the ConvNeXt-T vs Swin-T controlled pair with the objective fixes |

The figures built from these results are `physdecal fig-ce`, `physdecal fig-h1`
and `physdecal fig-loss` (see `physdecal <command> --help`).
