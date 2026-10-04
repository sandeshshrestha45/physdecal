#!/bin/sh
# Extend the seed replication from two white-box victims to the full roster.
#
# The four seed patches already exist under $OUT/patches/seedtest_*; only
# evaluation is needed, so nothing is re-optimised. Size-matched controls are
# NOT re-run: the anchor is seed-independent and the existing
# task_joint__anchor__<victim> results were verified identical to
# task_joint_ink__anchor__<victim> for every victim.
#
# Sequential on purpose: one GPU, and a concurrent run would thrash or OOM.
PD="${PYTHON:-python} -m physdecal"
cd "$(dirname "$0")/.." || exit 1
OUT="${PHYSDECAL_OUT_ROOT:-out}"

SEG="fcn_r50 deeplabv3_r101 upernet_convnext_t upernet_swin_t segformer_b2 clipseg"
DET="fasterrcnn fcos yolo"

for tag in seedtest_cube_s101 seedtest_cube_s202 seedtest_ink_s101 seedtest_ink_s202; do
  for m in $SEG; do
    if [ -f "$OUT/results/seg_${tag}__${m}.csv" ]; then
      echo "SKIP seg ${tag} ${m} (exists)"
      continue
    fi
    echo "=== seg ${tag} ${m}  $(date +%H:%M:%S)"
    $PD evaluate --patch "$tag" --models "$m" 2>&1 | tail -3
  done
  for d in $DET; do
    if [ -f "$OUT/results/det_${tag}__${d}.csv" ]; then
      echo "SKIP det ${tag} ${d} (exists)"
      continue
    fi
    echo "=== det ${tag} ${d}  $(date +%H:%M:%S)"
    $PD evaluate --patch "$tag" --detectors "$d" 2>&1 | tail -3
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
