#!/bin/sh
# Seed replication for the two claims of paper Sec. res-ce that rest on single
# runs (seed 20260911, the config default):
#   1. optimising through the printer response helps:
#        ce_lut_segformer_b0 vs ce_simple_segformer_b0, both scored --printed
#   2. the cross-entropy objective adds little over the hinge:
#        ce_dig_segformer_b0 vs dig_segformer_b0
# Seeds 101 and 202, as the seedtest_* runs of Finding 18. Every setting is
# copied from the original run.json; only the seed differs. The grey-square
# controls do not depend on the seed and are not re-run.
PD="${PYTHON:-python} -m physdecal"
cd "$(dirname "$0")/.." || exit 1
OUT="${PHYSDECAL_OUT_ROOT:-out}"
COMMON="--task seg --model segformer_b0 --realism 0 --size 2.0 --size-mode fixed --steps 3000 --batch 2"

run() {  # $1 tag, $2 "printed" or "", rest optimise args
  tag="$1"; printed="$2"; shift 2
  if [ ! -f "$OUT/patches/${tag}/run.json" ]; then
    echo "=== optimise ${tag}  $(date +%H:%M:%S)"
    $PD optimize $COMMON --tag "$tag" "$@" 2>&1 \
      | grep -E "wrote|Error|Traceback"
  fi
  f="$OUT/results/seg_${tag}__${printed:+printed__}segformer_b0.csv"
  if [ ! -f "$f" ]; then
    echo "=== score ${tag} ${printed}  $(date +%H:%M:%S)"
    $PD evaluate --patch "$tag" --models segformer_b0 \
      ${printed:+--printed} 2>&1 | grep -E "ASR|Error|Traceback"
  fi
}

for s in 101 202; do
  run ce_lut_segformer_b0_s$s    printed --seed $s --seg-objective ce_targeted --printer-lut
  run ce_simple_segformer_b0_s$s printed --seed $s --seg-objective ce_targeted
  run ce_dig_segformer_b0_s$s    ""      --seed $s --seg-objective ce_targeted --no-print-chain
  run dig_segformer_b0_s$s       ""      --seed $s --seg-objective vanish_prob --no-print-chain
done
echo "CE SEEDS DONE $(date +%H:%M:%S)"
