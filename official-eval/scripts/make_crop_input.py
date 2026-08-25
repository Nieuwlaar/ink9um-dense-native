#!/usr/bin/env python3
"""Cut a 64-aligned crop around the validation mask from a pooled input zarr.

Usage: make_crop_input.py <segment>
Reads  data/input/<seg>_input9um.zarr   (21,H,W)
       data/labels/<seg>/<seg>_validation_mask.zarr (bbox source, slice Z=10)
Writes data/input/<seg>_crop9um.zarr  and records the offset in data/crops.json.

Margin 512 px, crop origin snapped down to a multiple of 64 (the inference
stride), so the sliding-window patch grid inside the crop coincides with the
full-frame grid and hann-blended predictions over the mask are identical.
"""
import json
import sys
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

ROOT = Path(__file__).resolve().parent.parent
MARGIN = 512
ALIGN = 64


def main():
    seg = sys.argv[1]
    inp = zarr.open(str(ROOT / "data" / "input" / f"{seg}_input9um.zarr"), mode="r")["0"]
    v = np.asarray(
        zarr.open(str(ROOT / "data" / "labels" / seg / f"{seg}_validation_mask.zarr"), mode="r")["0"][10]
    )
    assert v.shape == inp.shape[1:], (v.shape, inp.shape)
    ys, xs = np.nonzero(v)
    y0 = max(0, int(ys.min()) - MARGIN) // ALIGN * ALIGN
    x0 = max(0, int(xs.min()) - MARGIN) // ALIGN * ALIGN
    y1 = min(inp.shape[1], int(ys.max()) + 1 + MARGIN)
    x1 = min(inp.shape[2], int(xs.max()) + 1 + MARGIN)
    block = np.asarray(inp[:, y0:y1, x0:x1])

    out_path = ROOT / "data" / "input" / f"{seg}_crop9um.zarr"
    group = zarr.open_group(str(out_path), mode="w")
    group.attrs.update({"crop_of": f"{seg}_input9um.zarr", "y0": y0, "x0": x0,
                        "y1": int(y1), "x1": int(x1), "margin": MARGIN, "align": ALIGN})
    arr = group.create_dataset(
        "0", shape=block.shape, chunks=(block.shape[0], 128, 128), dtype=block.dtype,
        compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE), fill_value=0,
    )
    arr[:] = block

    crops_file = ROOT / "data" / "crops.json"
    crops = json.loads(crops_file.read_text()) if crops_file.exists() else {}
    crops[seg] = {"y0": y0, "x0": x0, "y1": int(y1), "x1": int(x1),
                  "full_shape": [int(s) for s in inp.shape[1:]]}
    crops_file.write_text(json.dumps(crops, indent=2))
    print(f"{seg}: crop [{y0}:{y1}, {x0}:{x1}] of {inp.shape} -> {block.shape}")


if __name__ == "__main__":
    main()
