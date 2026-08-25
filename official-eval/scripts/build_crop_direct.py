#!/usr/bin/env python3
"""Build the pooled crop input directly from the (possibly partial) level-2 mirror.

Usage: build_crop_direct.py <segment>
Requires targeted_fetch.py to have completed for the segment (all crop-region
chunks present locally or absent on S3). Pooling math is identical to villa's
prepare_9um_isotropic_input.py (z 13:97 centered 84 planes, 4x rint-mean),
applied per-pixel, so the crop equals the same crop of the full pooled input.
Writes data/input/<seg>_crop9um.zarr and records the offset in data/crops.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crop_bounds import crop_bounds, ROOT

OUTPUT_Z = 21
POOL_Z = 4
INPUT_Z = OUTPUT_Z * POOL_Z  # 84


def main():
    seg = sys.argv[1]
    src = zarr.open(str(ROOT / "data" / "src" / f"{seg}_level2.zarr"), mode="r")
    shape = src.shape
    assert shape[0] >= INPUT_Z
    z0 = -(-(shape[0] - INPUT_Z) // 2)  # ceil, matches prepare script
    y0, x0, y1, x1, H, W = crop_bounds(seg)
    assert (H, W) == tuple(shape[1:]), (H, W, shape)

    block = np.asarray(src[z0:z0 + INPUT_Z, y0:y1, x0:x1], dtype=np.float32)
    pooled = np.rint(
        block.reshape(OUTPUT_Z, POOL_Z, y1 - y0, x1 - x0).mean(axis=1)
    ).astype(np.uint8)

    out_path = ROOT / "data" / "input" / f"{seg}_crop9um.zarr"
    group = zarr.open_group(str(out_path), mode="w")
    group.attrs.update({
        "format": "level2-zmean4-21slice-v1 (validation crop)",
        "source": f"{seg}_level2.zarr", "source_z_slice": [int(z0), int(z0 + INPUT_Z)],
        "y0": int(y0), "x0": int(x0), "y1": int(y1), "x1": int(x1),
    })
    arr = group.create_dataset(
        "0", shape=pooled.shape, chunks=(OUTPUT_Z, 128, 128), dtype=np.uint8,
        compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE), fill_value=0,
    )
    arr[:] = pooled

    crops_file = ROOT / "data" / "crops.json"
    crops = json.loads(crops_file.read_text()) if crops_file.exists() else {}
    crops[seg] = {"y0": int(y0), "x0": int(x0), "y1": int(y1), "x1": int(x1),
                  "full_shape": [int(H), int(W)]}
    crops_file.write_text(json.dumps(crops, indent=2))
    print(f"{seg}: crop [{y0}:{y1},{x0}:{x1}] pooled -> {pooled.shape} (z {z0}:{z0+INPUT_Z})")


if __name__ == "__main__":
    main()
