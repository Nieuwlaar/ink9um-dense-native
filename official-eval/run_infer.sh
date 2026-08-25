#!/bin/bash
# Inference on the three official ink_9um validation-mask segments,
# released checkpoint vs soup, default window plus the other four 17-of-21
# z windows for the z-window ensemble.
#
# Inputs (under data/input/): <seg>_input9um.zarr (full pooled input, from
# villa's scripts/prepare_9um_isotropic_input.py) or <seg>_crop9um.zarr
# (validation-region crop from scripts/build_crop_direct.py; in-mask
# predictions are identical, see README).
#
# Environment:
#   PY     python able to import koine_machines (villa branch
#          merge-ink-pipelines, ink-detection/), e.g. PY="uv run python"
#          with cwd = villa/ink-detection
#   MDIR   directory holding step-075000.pth and soup42_last4.pth
# Run from villa/ink-detection; EVAL resolves to this script's directory.
set -u
EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python}
MDIR=${MDIR:?set MDIR to the checkpoint directory}
mkdir -p "$EVAL/logs" "$EVAL/preds"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$EVAL/logs/infer.log"; }

SEGS=${SEGS:-"pherc0814-46527 pherc0139-w016 pherc1667-w029"}

infer(){ # seg ckpt name extra_args...
  local seg=$1 ckpt=$2 name=$3; shift 3
  local out="$EVAL/preds/${seg}__${name}.tif"
  [ -s "$out" ] && { log "skip $seg $name (exists)"; return 0; }
  local in="$EVAL/data/input/${seg}_input9um.zarr"
  [ -d "$in" ] || in="$EVAL/data/input/${seg}_crop9um.zarr"
  [ -d "$in" ] || { log "MISSING-INPUT $seg"; return 1; }
  log "infer $seg $name"
  $PY -m koine_machines.inference.infer "$in" "$ckpt" "$out" \
    --overlap 0.5 --blend-mode hann --direction forward --no-compile "$@" \
    >> "$EVAL/logs/infer.log" 2>&1 || { log "INFER-FAIL $seg $name"; rm -f "$out"; return 1; }
}

# Pass 1: default center window (layers 2..18 of 21), both checkpoints
for seg in $SEGS; do
  infer "$seg" "$MDIR/step-075000.pth" released
  infer "$seg" "$MDIR/soup42_last4.pth" soup42
done

# Pass 2: remaining 17-of-21 windows for the z-window ensemble
for seg in $SEGS; do
  for k in 0 1 3 4; do
    infer "$seg" "$MDIR/step-075000.pth" "released_z$k" --layer-start "$k" --layer-end $((k+17))
    infer "$seg" "$MDIR/soup42_last4.pth" "soup42_z$k" --layer-start "$k" --layer-end $((k+17))
  done
done

log "INFER-ALL-DONE"
