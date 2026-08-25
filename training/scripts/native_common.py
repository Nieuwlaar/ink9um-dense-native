"""Shared helpers for the native-113keV dense pseudo-label build.

Conventions (the frozen native-benchmark data sheet):
  - 113 keV surface volume = zarr2 (28,H,W) uint8 ("zarr113" in each segment dir, or the
    9.362um-1.2m-113keV-volume-20250728140407.zarr S3 mirror).
  - valid = max_z(volume) > 0 (cached valid113.npy where present).
  - 78 keV teacher = the org's canonical-2.4um "new_canon_autoresearch_recipe" ink map
    (uint8 prob*255 on the 2.399 um grid), resized cv2.INTER_AREA onto the 113 grid
    (H,W): pixel-grid-correct (IoU 0.96-0.99 against manual labels, zero shift).
  - Label zarr layout = the official native9 release: zarr v2 group, array "0",
    shape (28,H,W), chunks (28,128,128), u1, fill 0, blosc zstd clevel 3 bitshuffle,
    values 0/255, content on the middle plane z=14 only (the flat-mode loader reads
    amax over the 17-slice window, so z=14 is always included; train.py binarizes
    with > 0).
"""
import glob
import json
import os
import shutil

import numpy as np

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "109951162777600")
import cv2  # noqa: E402
import tifffile  # noqa: E402
import zarr  # noqa: E402
from numcodecs import Blosc  # noqa: E402

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Data locations — override via environment for your machine.
#   NATIVE113_SEGMENTS   one dir per PHerc0139 segment (zarr113/ + *78keV*autoresearch*.tif,
#                        pulled from the public S3 bucket; see native-eval/native_bench.py)
#   NATIVE113_SPLIT      json {"segments": [{"wname": ..., "segment_dir": ...}, ...]}
#                        mapping winding names to segment dirs
#   NATIVE113_THRESHOLDS optional lo/hi context table; leave unset if you don't have one
#   NATIVE9_LABELS       the official native9 sparse label release (HF bucket)
#   NATIVE9_VOLUMES      native 28-slice 113 keV volumes, one <w>.zarr per winding
DATA_R2B = os.environ.get("NATIVE113_SEGMENTS", os.path.join(W, "data", "segments"))
R2B_SPLIT = os.environ.get("NATIVE113_SPLIT", os.path.join(W, "data", "split.json"))
R2B_THRESH = os.environ.get("NATIVE113_THRESHOLDS", "")
NATIVE9_LABELS = os.environ.get(
    "NATIVE9_LABELS", os.path.join(W, "data", "labels", "native9-scrollprizeorg-21slices"))
NATIVE9_VOLUMES = os.environ.get(
    "NATIVE9_VOLUMES", os.path.join(W, "data", "volumes", "native113"))

HELD_OUT = ("title", "w024", "w047", "w053")   # NEVER in training
NATIVE_Z = 14                                   # annotated plane of a 28-deep native label zarr
D_NATIVE = 28

NATIVE_CHUNKS = (28, 128, 128)
NATIVE_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def log(msg):
    import datetime
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def r2b_segments():
    """{wname: segment_dir} for every segment in data_r2b (split.json + retry pool)."""
    split = json.load(open(R2B_SPLIT))
    out = {}
    for e in split["segments"] + split.get("retry_pool", []):
        if os.path.isdir(os.path.join(DATA_R2B, e["segment_dir"])):
            out[e["wname"]] = e["segment_dir"]
    return out


def training_wnames():
    segs = r2b_segments()
    train = sorted(w for w in segs if w not in HELD_OUT)
    for h in HELD_OUT:
        assert h not in train
    return train, segs


def open_zarr_array(path):
    z = zarr.open(path, mode="r")
    return z["0"] if hasattr(z, "keys") and "0" in z else z


def load_valid(sdir_or_zarr, cache_path=None):
    """valid = max_z(vol) > 0; cached .npy when cache_path is given."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    a = open_zarr_array(sdir_or_zarr)
    d, h, w = a.shape
    valid = np.zeros((h, w), bool)
    step = 1024
    for y in range(0, h, step):
        valid[y:y + step] = np.asarray(a[:, y:y + step, :]).max(axis=0) > 0
    if cache_path:
        np.save(cache_path, valid)
    return valid


def teacher_on_grid(tif_path, shape_hw):
    """Teacher tif (uint8/uint16 at the 2.399um grid) -> INTER_AREA onto (H,W)."""
    t = tifffile.imread(tif_path)
    if t.dtype == np.uint16:
        t = (t >> 8).astype(np.uint8)
    assert t.dtype == np.uint8, t.dtype
    h, w = shape_hw
    return cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA)


def balanced_accuracy_curve(score_u8, label_bool, mask_bool):
    """BA(t) for t in 1..255 of (score>=t) vs label over mask; returns (t*, BA*, curve)."""
    s = score_u8[mask_bool]
    y = label_bool[mask_bool]
    n_pos = int(y.sum()); n_neg = int(y.size - n_pos)
    assert n_pos > 0 and n_neg > 0, (n_pos, n_neg)
    hp = np.bincount(s[y], minlength=256).astype(np.float64)
    hn = np.bincount(s[~y], minlength=256).astype(np.float64)
    # TP(t) = #pos with s>=t ; TN(t) = #neg with s<t
    tp = hp[::-1].cumsum()[::-1]          # tp[t] = sum_{v>=t} hp[v]
    tn = np.concatenate([[0.0], hn.cumsum()[:-1]])  # tn[t] = sum_{v<t} hn[v]
    ba = 0.5 * (tp / n_pos + tn / n_neg)
    ts = np.arange(1, 256)
    t_star = int(ts[np.argmax(ba[1:])])
    return t_star, float(ba[t_star]), ba, n_pos, n_neg


def write_native_label_zarr(dst_path, plane_u8, shape_zyx, attrs=None):
    """Official native9 layout: group/'0', (28,H,W), chunks (28,128,128), u1, zstd-3
    bitshuffle, fill 0; only z=14 written (other planes are fill chunks)."""
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    d, h, w = shape_zyx
    assert plane_u8.shape == (h, w) and plane_u8.dtype == np.uint8
    g = zarr.open_group(dst_path, mode="w")
    arr = g.create_dataset("0", shape=(d, h, w), chunks=NATIVE_CHUNKS, dtype=np.uint8,
                           fill_value=0, compressor=NATIVE_COMPRESSOR, overwrite=True)
    arr.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]
    band = 1024
    for y0 in range(0, h, band):
        y1 = min(h, y0 + band)
        arr[NATIVE_Z, y0:y1, :] = plane_u8[y0:y1, :]
    if attrs:
        g.attrs.update(attrs)
    return arr
