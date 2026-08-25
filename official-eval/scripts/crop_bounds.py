#!/usr/bin/env python3
"""Shared crop-bound computation (same formula as make_crop_input.py)."""
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent.parent
MARGIN = 512
ALIGN = 64


def crop_bounds(seg: str):
    v = np.asarray(
        zarr.open(str(ROOT / "data" / "labels" / seg / f"{seg}_validation_mask.zarr"), mode="r")["0"][10]
    )
    H, W = v.shape
    ys, xs = np.nonzero(v)
    y0 = max(0, int(ys.min()) - MARGIN) // ALIGN * ALIGN
    x0 = max(0, int(xs.min()) - MARGIN) // ALIGN * ALIGN
    y1 = min(H, int(ys.max()) + 1 + MARGIN)
    x1 = min(W, int(xs.max()) + 1 + MARGIN)
    return y0, x0, y1, x1, H, W
