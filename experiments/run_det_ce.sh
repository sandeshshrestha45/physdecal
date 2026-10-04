#!/bin/sh
# The detection mirror of paper Sec. res-ce: simple patches (no reference
# design) against RetinaNet (white box), task det, 2.0 m, 3000 steps, batch 2,
# three seeds each. Detection form of the supervisor's objective
# (--det-objective, losses.det_attack_loss).
#
#   dce_lut      targeted CE, through printer LUT, chain   printed, 4 detectors
#   dce_simple   targeted CE, free pixels, chain           printed, 4 detectors
#                                                          + digital, RetinaNet
#   dce_unt_lut  untargeted -CE, through printer LUT       printed, RetinaNet
#   ddig         hinge, free pixels, no chain              digital, RetinaNet
#   dce_dig      targeted CE, free pixels, no chain        digital, RetinaNet
#   dce_ink / dce_pal_snap / dce_pal_logit   32 measured inks   digital, RetinaNet
#   controls: grey square (digital, RetinaNet) and printed grey (4 detectors)
#
# Ordered so the headline comparison lands first. Resumable.
PD="${PYTHON:-python} -m physdecal"
cd "$(dirname "$0")/.." || exit 1
OUT="${PHYSDECAL_OUT_ROOT:-out}"
BASE="--task det --det retinanet --size 2.0 --size-mode fixed --steps 3000 --batch 2"
INK="--realism 3 --inks data/reference/printable.csv --ink-count 32"
OTHERS="fasterrcnn fcos yolo"

opt() {  # $1 tag, rest optimise args
  tag="$1"; shift
  [ -f "$OUT/patches/$tag/run.json" ] && return
  echo "=== optimise $tag  $(date +%H:%M:%S)"
  $PD optimize $BASE --tag "$tag" "$@" 2>&1 | grep -E "wrote|Error|Traceback"
}
score() {  # $1 tag, $2 detector, $3 "printed" or "", $4 optional baseline
  f="$OUT/results/det_$1__${4:+$4__}${3:+printed__}$2.csv"
  [ -f "$f" ] && return
  echo "=== score $1 $2 $3 $4  $(date +%H:%M:%S)"
  $PD evaluate --patch "$1" --detectors "$2" ${3:+--printed} \
    ${4:+--baseline $4} 2>&1 | grep -E "ASR|Error|Traceback"
}
sfx() { [ -n "$1" ] && echo "_s$1"; }

# 1. headline: through LUT vs free pixels, printed, white box
for s in "" 101 202; do
  t=dce_lut_retinanet$(sfx $s)
  opt $t --realism 0 --det-objective ce_targeted --printer-lut ${s:+--seed $s}
  score $t retinanet printed
  t=dce_simple_retinanet$(sfx $s)
  opt $t --realism 0 --det-objective ce_targeted ${s:+--seed $s}
  score $t retinanet printed
  score $t retinanet ""
done
score dce_lut_retinanet retinanet printed grey
echo "DET HEADLINE DONE $(date +%H:%M:%S)"

# 2. untargeted, hinge and CE digital
for s in "" 101 202; do
  t=dce_unt_lut_retinanet$(sfx $s)
  opt $t --realism 0 --det-objective ce_untargeted --printer-lut ${s:+--seed $s}
  score $t retinanet printed
  t=ddig_retinanet$(sfx $s)
  opt $t --realism 0 --det-objective hinge --no-print-chain ${s:+--seed $s}
  score $t retinanet ""
  t=dce_dig_retinanet$(sfx $s)
  opt $t --realism 0 --det-objective ce_targeted --no-print-chain ${s:+--seed $s}
  score $t retinanet ""
done
score ddig_retinanet retinanet "" grey
echo "DET OBJECTIVES DONE $(date +%H:%M:%S)"

# 3. transfer of the two printed patches, all seeds, plus printed controls
for d in $OTHERS; do
  score dce_lut_retinanet "$d" printed grey
  for s in "" 101 202; do
    score dce_lut_retinanet$(sfx $s) "$d" printed
    score dce_simple_retinanet$(sfx $s) "$d" printed
  done
done
echo "DET TRANSFER DONE $(date +%H:%M:%S)"

# 4. 32 measured inks
for s in "" 101 202; do
  t=dce_ink_retinanet$(sfx $s)
  opt $t $INK --det-objective ce_targeted ${s:+--seed $s}
  score $t retinanet ""
  t=dce_pal_snap_retinanet$(sfx $s)
  opt $t $INK --palette-only --det-objective ce_targeted ${s:+--seed $s}
  score $t retinanet ""
  t=dce_pal_logit_retinanet$(sfx $s)
  opt $t $INK --palette-only --level4-param ink_logits --det-objective ce_targeted ${s:+--seed $s}
  score $t retinanet ""
done
echo "DET ALL DONE $(date +%H:%M:%S)"
