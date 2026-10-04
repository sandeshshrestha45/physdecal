#!/bin/sh
# Printable counterpart of ce_simple_segformer_b0: the same simple patch (no
# reference design), same CE objective, but at realism level 3 so every pixel
# is snapped to a colour the printer actually produced on the scanned ink
# chart (data/reference/printable.csv, 216 measured colours reduced to 32 real
# chart colours; palette size made no measurable difference, Finding 19).
# Level 3 also band-limits and flattens the patch.
#
# Waits for run_ce.sh to release the GPU, then scores both simple patches
# against a grey square, the right control for a patch with no anchor design.
PD="${PYTHON:-python} -m physdecal"
cd "$(dirname "$0")/.." || exit 1
OUT="${PHYSDECAL_OUT_ROOT:-out}"

until grep -q "CE DONE" $OUT/logs/run_ce.log; do sleep 60; done

tag=ce_ink_segformer_b0
if [ ! -f "$OUT/patches/${tag}/run.json" ]; then
  echo "=== optimise ${tag}  $(date +%H:%M:%S)"
  $PD optimize --task seg --model segformer_b0 --realism 3 \
    --size 2.0 --size-mode fixed --steps 3000 --batch 2 \
    --seg-objective ce_targeted --inks data/reference/printable.csv \
    --ink-count 32 --tag "$tag" 2>&1 \
    | grep -E "inks:|^step +[0-9]*(00|99)/|wrote|Error|Traceback"
fi

for t in "$tag" ce_simple_segformer_b0; do
  for kind in "" grey; do
    f="$OUT/results/seg_${t}__${kind:+grey__}segformer_b0.csv"
    if [ ! -f "$f" ]; then
      echo "=== score ${t} ${kind:-attack}  $(date +%H:%M:%S)"
      $PD evaluate --patch "$t" --models segformer_b0 \
        ${kind:+--baseline $kind} 2>&1 | grep -E "ASR|Error|Traceback"
    fi
  done
done
echo "CE INK DONE $(date +%H:%M:%S)"
