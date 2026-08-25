#!/bin/bash
# z-window ensemble inference: run the same checkpoint over several 17-slice
# z windows of the input volume and average the prediction maps per pixel.
#
# Usage: ./zavg_infer.sh <input.zarr> <checkpoint.pth> <out_prefix> <ls:le> [<ls:le> ...]
#
# Window recipes (layer-start:layer-end):
#   native 28-slice surface volumes:  0:20 3:23 5:25 8:27
#   aligned 21-slice pooled inputs:   0:17 1:18 2:19 3:20 4:21
#
# Run from villa's ink-detection directory (branch merge-ink-pipelines), e.g.
#   PY="uv run python" /path/to/zavg_infer.sh ...
# or set PYTHONPATH so `python -m koine_machines.inference.infer` resolves.
#
# Outputs: <out_prefix>_z<ls>-<le>.tif per window (kept, so reruns resume),
#          <out_prefix>_zavg_prob.npy (float [0,1]) and _zavg_prob.tif.
set -euo pipefail
IN=$1; CKPT=$2; PREFIX=$3; shift 3
[ $# -ge 1 ] || { echo "need at least one ls:le window" >&2; exit 1; }
PY=${PY:-python}

TIFS=()
for win in "$@"; do
  ls=${win%%:*}; le=${win##*:}
  tif=${PREFIX}_z${ls}-${le}.tif
  if [ ! -s "$tif" ]; then
    echo "[zavg] window $ls:$le"
    $PY -m koine_machines.inference.infer "$IN" "$CKPT" "$tif" \
      --overlap 0.5 --blend-mode hann --direction forward --no-compile \
      --layer-start "$ls" --layer-end "$le"
  fi
  TIFS+=("$tif")
done

$PY - "$PREFIX" "${TIFS[@]}" <<'PYEOF'
import sys
import numpy as np
import tifffile

prefix, tifs = sys.argv[1], sys.argv[2:]
maps = []
for t in tifs:
    a = tifffile.imread(t).astype(np.float32)
    # model-card scaling of the uint8 outputs: prob = (value - 64) / 128
    maps.append(np.clip((a - 64.0) / 128.0, 0.0, 1.0))
avg = np.mean(maps, axis=0).astype(np.float32)
np.save(prefix + "_zavg_prob.npy", avg)
tifffile.imwrite(prefix + "_zavg_prob.tif", (avg * 255).astype(np.uint8))
print(f"{prefix}_zavg_prob.npy  (mean of {len(maps)} windows)")
PYEOF
