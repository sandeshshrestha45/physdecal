#!/bin/sh
# Printable simple patch via the measured printer colour response
# (printchain.PrinterLUT: requested -> scanned colour, 6x6x6 ink chart)
# instead of snapping to a few inks, which cost the attack (run_ce_palette.sh).
#
#   ce_lut_segformer_b0   = ce_simple_segformer_b0 + --printer-lut
#
# Scored "as printed" (--printed), next to ce_simple as printed (the same
# patch optimised without knowing the printer) and a printed grey square.
PD="${PYTHON:-python} -m physdecal"
cd "$(dirname "$0")/.." || exit 1
OUT="${PHYSDECAL_OUT_ROOT:-out}"

tag=ce_lut_segformer_b0
if [ ! -f "$OUT/patches/${tag}/run.json" ]; then
  echo "=== optimise ${tag}  $(date +%H:%M:%S)"
  $PD optimize --task seg --model segformer_b0 --realism 0 \
    --size 2.0 --size-mode fixed --steps 3000 --batch 2 \
    --seg-objective ce_targeted --printer-lut --tag "$tag" 2>&1 \
    | grep -E "^step +[0-9]*(00|99)/|wrote|Error|Traceback"
fi

score() {  # $1 patch tag, $2 optional baseline
  f="$OUT/results/seg_$1__${2:+$2__}printed__segformer_b0.csv"
  if [ ! -f "$f" ]; then
    echo "=== score $1 ${2:-attack} printed  $(date +%H:%M:%S)"
    $PD evaluate --patch "$1" --models segformer_b0 --printed \
      ${2:+--baseline $2} 2>&1 | grep -E "ASR|Error|Traceback"
  fi
}
score "$tag"
score ce_simple_segformer_b0
score "$tag" grey
echo "CE LUT DONE $(date +%H:%M:%S)"
