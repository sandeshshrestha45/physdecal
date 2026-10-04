#!/bin/sh
# Random (unoptimised) patch control for the simple patches: uniform noise
# (evaluate --baseline random), same 2.0 m size, scored through the printer
# response (--printed) like the optimised patches. Segmentation against every
# seg victim (size from ce_lut_segformer_b0), detection against every detector
# (size from dce_lut_retinanet). Resumable.
PD="${PYTHON:-python} -m physdecal"
cd "$(dirname "$0")/.." || exit 1
OUT="${PHYSDECAL_OUT_ROOT:-out}"
for v in segformer_b0 fcn_r50 deeplabv3_r101 upernet_convnext_t upernet_swin_t segformer_b2 clipseg; do
  f="$OUT/results/seg_ce_lut_segformer_b0__random__printed__$v.csv"
  [ "$v" = clipseg ] && f="$OUT/results/seg_ce_lut_segformer_b0__random__printed__clipseg__a_car.csv"
  [ -f "$f" ] && continue
  echo "=== seg $v  $(date +%H:%M:%S)"
  $PD evaluate --patch ce_lut_segformer_b0 --models "$v" --baseline random --printed 2>&1 | grep -E "ASR|Error|Traceback"
done
for d in retinanet fasterrcnn fcos yolo; do
  f="$OUT/results/det_dce_lut_retinanet__random__printed__$d.csv"
  [ -f "$f" ] && continue
  echo "=== det $d  $(date +%H:%M:%S)"
  $PD evaluate --patch dce_lut_retinanet --detectors "$d" --baseline random --printed 2>&1 | grep -E "ASR|Error|Traceback"
done
echo "RANDOM DONE $(date +%H:%M:%S)"
