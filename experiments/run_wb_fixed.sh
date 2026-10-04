#!/bin/sh
# Re-run the RQ6 controlled pair (Finding 15) with the objective-verification
# fixes of Finding 20: log-odds hinge (SEG_OBJECTIVE "vanish") and one logit
# per anchor region and ink (LEVEL4_PARAM "ink_logits").
#
# The original wb_upernet_swin_t diverged, so its ASR of 0.000 measured the
# optimiser, not Swin. Both halves of the pair get the fixes, so the
# ConvNeXt-vs-Swin comparison still differs only in the backbone. Every other
# setting matches the originals: 2.0 m fixed, realism 4, cargo, supercell 16,
# 3000 steps, batch 2, print chain on. New tags; the originals are untouched.
#
# Evaluation runs in priority order: each patch against its own white-box
# victim first (the RQ6 number), then the rest of the roster. One victim per
# call, skipped when its CSV exists, so the script resumes after an
# interruption and never scores a victim twice. Sequential on purpose: one GPU.
PD="${PYTHON:-python} -m physdecal"
cd "$(dirname "$0")/.." || exit 1
OUT="${PHYSDECAL_OUT_ROOT:-out}"
PAIR="upernet_convnext_t upernet_swin_t"
REST="upernet_convnext_t upernet_swin_t fcn_r50 deeplabv3_r101 segformer_b0 segformer_b2 clipseg"

for m in $PAIR; do
  tag="wb_${m}_fixed"
  if [ ! -f "$OUT/patches/${tag}/run.json" ]; then
    echo "=== optimise ${tag}  $(date +%H:%M:%S)"
    $PD optimize --task seg --model "$m" --realism 4 --design cargo \
      --size 2.0 --size-mode fixed --steps 3000 --batch 2 --supercell 16 \
      --seg-objective vanish --level4-param ink_logits --tag "$tag" 2>&1 \
      | grep -E "^step +[0-9]*(00|99)/|wrote|Error|Traceback"
  else
    echo "SKIP optimise ${tag} (exists)"
  fi
done

# $1 tag, $2 victim, $3 optional "anchor". clipseg writes one CSV per prompt,
# so its done-marker is the first prompt's file.
score() {
  pre="$OUT/results/seg_$1__"; [ -n "$3" ] && pre="${pre}anchor__"
  done_f="${pre}$2.csv"; [ "$2" = clipseg ] && done_f="${pre}clipseg__a_car.csv"
  if [ -f "$done_f" ]; then echo "SKIP $1 $2 $3"; return; fi
  echo "=== $1 $2 ${3:-attack}  $(date +%H:%M:%S)"
  $PD evaluate --patch "$1" --models "$2" ${3:+--baseline $3} 2>&1 \
    | grep -E "ASR|Error|Traceback"
}

# 1. the RQ6 numbers
for m in $PAIR; do score "wb_${m}_fixed" "$m"; done
for m in $PAIR; do score "wb_${m}_fixed" "$m" anchor; done
echo "RQ6 SCORED $(date +%H:%M:%S)"

# 2. the rest of the roster, attack then control, then the detector
for m in $PAIR; do
  tag="wb_${m}_fixed"
  for v in $REST; do score "$tag" "$v"; done
  for v in $REST; do score "$tag" "$v" anchor; done
  for kind in "" anchor; do
    f="$OUT/results/det_${tag}__${kind:+anchor__}retinanet.csv"
    if [ ! -f "$f" ]; then
      echo "=== ${tag} retinanet ${kind:-attack}  $(date +%H:%M:%S)"
      $PD evaluate --patch "$tag" --detectors retinanet \
        ${kind:+--baseline anchor} 2>&1 | grep -E "ASR|Error|Traceback"
    fi
  done
  echo "=== figures ${tag}  $(date +%H:%M:%S)"
  $PD allfigs --patch "$tag" 2>&1 | tail -30
done
echo "WB FIXED DONE $(date +%H:%M:%S)"
