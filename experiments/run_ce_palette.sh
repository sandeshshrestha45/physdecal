#!/bin/sh
# Follow-up to run_ce_ink.sh: ce_ink_segformer_b0 (level 3, band + flat +
# 32 measured inks) fell from ASR 0.293 to 0.138, with a flat loss. These two
# runs keep ONLY the printable-colour constraint (no band, no flat) and
# differ only in how the optimiser searches:
#
#   ce_pal_snap_segformer_b0    free RGB snapped to the nearest ink
#                               (straight-through identity gradient)
#   ce_pal_logit_segformer_b0   one logit per pixel and ink (InkPixelTexture)
#
# Everything else matches ce_ink_segformer_b0. Scored on segformer_b0 with the
# grey-square control already on disk (seg_ce_ink_segformer_b0__grey__...).
PD="${PYTHON:-python} -m physdecal"
cd "$(dirname "$0")/.." || exit 1
OUT="${PHYSDECAL_OUT_ROOT:-out}"

run() {  # $1 tag, rest extra optimise args
  tag="$1"; shift
  if [ ! -f "$OUT/patches/${tag}/run.json" ]; then
    echo "=== optimise ${tag}  $(date +%H:%M:%S)"
    $PD optimize --task seg --model segformer_b0 --realism 3 \
      --size 2.0 --size-mode fixed --steps 3000 --batch 2 \
      --seg-objective ce_targeted --inks data/reference/printable.csv \
      --ink-count 32 --palette-only --tag "$tag" "$@" 2>&1 \
      | grep -E "inks:|^step +[0-9]*(00|99)/|wrote|Error|Traceback"
  fi
  f="$OUT/results/seg_${tag}__segformer_b0.csv"
  if [ ! -f "$f" ]; then
    echo "=== score ${tag}  $(date +%H:%M:%S)"
    $PD evaluate --patch "$tag" --models segformer_b0 2>&1 \
      | grep -E "ASR|Error|Traceback"
  fi
}

run ce_pal_snap_segformer_b0
run ce_pal_logit_segformer_b0 --level4-param ink_logits
echo "CE PALETTE DONE $(date +%H:%M:%S)"
