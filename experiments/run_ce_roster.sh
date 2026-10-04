#!/bin/sh
# Close the "single run, single victim" gaps of paper Sec. res-ce.
#   1. Transfer: the printed simple patches and a printed grey control scored
#      on every other segmentation victim (black box; optimised on segformer_b0).
#   2. Seeds 101 and 202 for the three 32-ink rows, on segformer_b0. The ink
#      palette (k-means to 32) is identical across processes, so the seed
#      varies only the optimisation.
#   3. Transfer for the remaining seeds of the two printed patches.
# Resumable: every step is skipped when its output exists.
PD="${PYTHON:-python} -m physdecal"
cd "$(dirname "$0")/.." || exit 1
OUT="${PHYSDECAL_OUT_ROOT:-out}"
ROSTER="upernet_convnext_t upernet_swin_t fcn_r50 deeplabv3_r101 segformer_b2 clipseg"

score() {  # $1 tag, $2 victim, $3 "printed" or "", $4 optional baseline
  pre="$OUT/results/seg_$1__${4:+$4__}${3:+printed__}"
  f="${pre}$2.csv"; [ "$2" = clipseg ] && f="${pre}clipseg__a_car.csv"
  [ -f "$f" ] && return
  echo "=== score $1 $2 ${3} ${4}  $(date +%H:%M:%S)"
  $PD evaluate --patch "$1" --models "$2" ${3:+--printed} \
    ${4:+--baseline $4} 2>&1 | grep -E "ASR|Error|Traceback"
}

# 1. transfer, original seed
for v in $ROSTER; do
  score ce_lut_segformer_b0 "$v" printed
  score ce_lut_segformer_b0 "$v" printed grey
  score ce_simple_segformer_b0 "$v" printed
done
echo "ROSTER SEED0 DONE $(date +%H:%M:%S)"

# 2. seeds for the 32-ink rows
INK="--task seg --model segformer_b0 --realism 3 --size 2.0 --size-mode fixed --steps 3000 --batch 2 --seg-objective ce_targeted --inks data/reference/printable.csv --ink-count 32"
for s in 101 202; do
  for spec in "ce_ink_segformer_b0:" "ce_pal_snap_segformer_b0:--palette-only" \
              "ce_pal_logit_segformer_b0:--palette-only --level4-param ink_logits"; do
    base="${spec%%:*}"; extra="${spec#*:}"; tag="${base}_s$s"
    if [ ! -f "$OUT/patches/$tag/run.json" ]; then
      echo "=== optimise $tag  $(date +%H:%M:%S)"
      $PD optimize $INK $extra --seed $s --tag "$tag" 2>&1 \
        | grep -E "wrote|Error|Traceback"
    fi
    score "$tag" segformer_b0 ""
  done
done
echo "INK SEEDS DONE $(date +%H:%M:%S)"

# 3. transfer, remaining seeds
for s in 101 202; do
  for v in $ROSTER; do
    score ce_lut_segformer_b0_s$s "$v" printed
    score ce_simple_segformer_b0_s$s "$v" printed
  done
done
echo "ROSTER ALL DONE $(date +%H:%M:%S)"
