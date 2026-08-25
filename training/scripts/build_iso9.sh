#!/bin/bash
# build_iso9.sh <seg> [<seg> ...] — official-recipe 21-slice ~9.6um training input
# (villa scripts/prepare_9um_isotropic_input.py: level-2 XY, centered 84 z planes,
# 4x rint-mean -> 21) from the local level-2 mirror (scripts/fetch_s3_level2.py) or, with
# REMOTE=1, straight from the public S3 zarr URL (no local mirror; rental use).
# Output: $OUT/<seg>.zarr (group with array "0", (21,H,W) u1 zstd-5 bitshuffle).
set -u
W=${W:-.}
OUT=${OUT:-$W/data/volumes/aligned-iso9}
PY=${PY:-python}
PREP=${PREP:?set PREP to villa ink-detection/scripts/prepare_9um_isotropic_input.py}
WORKERS=${WORKERS:-8}
mkdir -p "$OUT"
for seg in "$@"; do
  dst=$OUT/$seg.zarr
  [[ -f $dst/BUILD_OK ]] && { echo "$seg: iso9 present"; continue; }
  rm -rf "$dst" "$dst.partial"
  if [[ -n ${REMOTE:-} ]]; then
    src=$(awk -F'\t' -v s="$seg" '$1==s{print $3}' $W/data/aligned_sources.tsv); lvl=2
  else
    src=$W/data/volumes/level2/${seg}_level2.zarr; lvl=2
    [[ -f $src/FETCH_OK || -f $src/.zarray ]] || { echo "$seg: no level-2 mirror at $src"; continue; }
  fi
  t0=$(date +%s)
  if [[ -n ${REMOTE:-} ]]; then
    $PY "$PREP" "$src" "$dst" --level 2 --workers "$WORKERS"
  else
    $PY "$PREP" "$src" "$dst" --level 2 --workers "$WORKERS"   # bare array: --level ignored by open_source_array
  fi
  rc=$?
  if [[ $rc -eq 0 ]]; then
    $PY - "$dst" <<'PYEOF' && touch "$dst/BUILD_OK"
import sys, zarr, numpy as np
a = zarr.open(sys.argv[1], mode="r")["0"]
assert a.shape[0] == 21, a.shape
bad = [z for z in range(21) if not np.any(np.asarray(a[z, ::16, ::16]))]
nz10 = float((np.asarray(a[10]) > 0).mean())
print(f"iso9 {sys.argv[1]} shape={a.shape} empty_planes={bad} z10_nonzero={nz10:.3f}")
assert not bad, f"all-zero planes {bad}"
PYEOF
  fi
  echo "$seg: rc=$rc $(( $(date +%s) - t0 ))s $(du -sh $dst 2>/dev/null | cut -f1)"
done
