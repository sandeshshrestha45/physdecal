#!/bin/sh
# Supervisor's untargeted form, loss = -CE(pred, gt) (SEG_OBJECTIVE
# "ce_untargeted"), in exactly the setting of ce_lut_segformer_b0: level 0,
# print chain, printer LUT, 2.0 m, 3000 steps, batch 2. Three seeds, scored as
# printed on segformer_b0; the printed grey control already exists
# (seg_ce_lut_segformer_b0__grey__printed__segformer_b0.csv).
PD="${PYTHON:-python} -m physdecal"
cd "$(dirname "$0")/.." || exit 1
OUT="${PHYSDECAL_OUT_ROOT:-out}"

for s in "" 101 202; do
  tag="ce_unt_lut_segformer_b0${s:+_s$s}"
  if [ ! -f "$OUT/patches/$tag/run.json" ]; then
    echo "=== optimise $tag  $(date +%H:%M:%S)"
    $PD optimize --task seg --model segformer_b0 --realism 0 \
      --size 2.0 --size-mode fixed --steps 3000 --batch 2 \
      --seg-objective ce_untargeted --printer-lut ${s:+--seed $s} \
      --tag "$tag" 2>&1 | grep -E "^step +[0-9]*(000|999)/|wrote|Error|Traceback"
  fi
  if [ ! -f "$OUT/results/seg_${tag}__printed__segformer_b0.csv" ]; then
    echo "=== score $tag printed  $(date +%H:%M:%S)"
    $PD evaluate --patch "$tag" --models segformer_b0 --printed 2>&1 \
      | grep -E "ASR|Error|Traceback"
  fi
done
echo "CE UNTARGETED DONE $(date +%H:%M:%S)"
