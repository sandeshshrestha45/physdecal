#!/bin/sh
# Test hypothesis H1: does the measured gamut help because of its CHROMATIC
# EXTENT, its CARDINALITY, or merely because it is a smaller search space?
#
# Every run below matches task_joint_ink in all seventeen recorded settings
# except the ink set: joint task, segformer_b0 + retinanet, realism 4, cargo
# anchor, 2.0 m fixed, 256 px, 3000 steps, batch 1, lr 0.02, supercell 16,
# print chain on. Only --inks changes. Seed is the default, as for the
# controlled pair.
#
#   chroma_g*  cardinality fixed at 8, mean C* 16.4 / 32.8 / 47.7 / 58.6
#              (g100 is the measured set and reproduces task_joint_ink)
#   card_k*    chroma fixed at the measured level, k = 3 / 5 / 16 / 32
#   rand_s*    k, L* and C* matched to the measured set, hues redrawn
PD="${PYTHON:-python} -m physdecal"
cd "$(dirname "$0")/.." || exit 1
OUT="${PHYSDECAL_OUT_ROOT:-out}"
P="data/reference/palettes"

for name in chroma_g050 chroma_g100 chroma_g150 chroma_g200 \
            card_k03 card_k05 card_k16 card_k32 \
            rand_s11 rand_s22 rand_s33; do
  tag="h1_${name}"
  if [ ! -d "$OUT/patches/${tag}" ]; then
    echo "=== optimise ${tag}  $(date +%H:%M:%S)"
    $PD optimize --task joint --model segformer_b0 --det retinanet \
      --realism 4 --design cargo --size 2.0 --size-mode fixed \
      --steps 3000 --batch 1 --lr 0.02 --supercell 16 \
      --inks "${P}/${name}.csv" --tag "$tag" 2>&1 | tail -4
  else
    echo "SKIP optimise ${tag} (exists)"
  fi
  if [ ! -f "$OUT/results/seg_${tag}__segformer_b0.csv" ]; then
    echo "=== evaluate ${tag}  $(date +%H:%M:%S)"
    $PD evaluate --patch "$tag" --models segformer_b0 2>&1 | tail -2
  else
    echo "SKIP evaluate ${tag} (exists)"
  fi
done
echo "H1 DONE $(date +%H:%M:%S)"
