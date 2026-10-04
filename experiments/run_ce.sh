#!/bin/sh
# Supervisor's cross-entropy objective (SEG_OBJECTIVE "ce_targeted": target
# label map is background everywhere, minimise CE(pred, target)) on a simple
# level-0 patch -- free pixels, no reference design -- universal over every
# nadir train instance.
#
#   ce_dig_segformer_b0     no print chain; identical to dig_segformer_b0
#                           except the loss, so it isolates the loss change
#   ce_simple_segformer_b0  print chain on, the candidate for real photographs
#
# Each patch is scored on its white-box victim, attack then anchor control.
PD="${PYTHON:-python} -m physdecal"
cd "$(dirname "$0")/.." || exit 1
OUT="${PHYSDECAL_OUT_ROOT:-out}"

run() {  # $1 tag, rest extra optimise args
  tag="$1"; shift
  if [ ! -f "$OUT/patches/${tag}/run.json" ]; then
    echo "=== optimise ${tag}  $(date +%H:%M:%S)"
    $PD optimize --task seg --model segformer_b0 --realism 0 \
      --size 2.0 --size-mode fixed --steps 3000 --batch 2 \
      --seg-objective ce_targeted --tag "$tag" "$@" 2>&1 \
      | grep -E "^step +[0-9]*(00|99)/|wrote|Error|Traceback"
  fi
  for kind in "" anchor; do
    f="$OUT/results/seg_${tag}__${kind:+anchor__}segformer_b0.csv"
    if [ ! -f "$f" ]; then
      echo "=== score ${tag} ${kind:-attack}  $(date +%H:%M:%S)"
      $PD evaluate --patch "$tag" --models segformer_b0 \
        ${kind:+--baseline anchor} 2>&1 | grep -E "ASR|Error|Traceback"
    fi
  done
}

run ce_dig_segformer_b0 --no-print-chain
run ce_simple_segformer_b0
echo "CE DONE $(date +%H:%M:%S)"
