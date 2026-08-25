#!/bin/bash
# Rebuild soup42_last4.pth: the uniform weight average of the last four
# released ink_9um seed-42 checkpoints (steps 40k, 50k, 60k, 75k).
#
# Usage: ./build_soup.sh [checkpoint_dir] [out.pth]
#   checkpoint_dir  where the four .pth files live or get downloaded
#                   (default: ./checkpoints)
#   out.pth         output path (default: ./soup42_last4.pth)
#
# Checkpoints already present are not re-downloaded, so pointing
# checkpoint_dir at an existing local copy skips the download entirely.
# Needs: curl, python with torch.
set -euo pipefail
DIR=${1:-checkpoints}
OUT=${2:-soup42_last4.pth}
BASE=https://huggingface.co/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed42
PY=${PY:-python}

mkdir -p "$DIR"
CKPTS=()
for s in 040000 050000 060000 075000; do
  f=$DIR/step-$s.pth
  if [ ! -s "$f" ]; then
    echo "downloading step-$s.pth"
    curl -L --fail -o "$f" "$BASE/step-$s.pth"
  fi
  CKPTS+=("$f")
done

$PY "$(dirname "$0")/make_soup.py" "$OUT" "${CKPTS[@]}"
