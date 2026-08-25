#!/usr/bin/env python
"""Recompute the Figure 1 caption metrics from the archived prediction maps.

Region A_titleline_left (x1050 y900 w1300 h400, frozen 2026-08-05) on the
title segment's native 113 keV grid (1780x5360, 9.362 um/px). Classes: ink =
the segment's published 78 keV ink map, INTER_AREA-resized onto the 113 keV
grid, >= 128 on valid pixels; background = all other valid pixels (no ignore
band — this differs from the frozen benchmark's lo/hi construction). Teacher
components: connected components of the ink class >= 1000 px; a component is
hit when >= 30% of its pixels clear the model's 5%-false-positive threshold.

Usage:
    python fig1_metrics.py --seg-dir <title_dir> \
        --pred out/dense_native_prob.npy --pred out/control_prob.npy

<title_dir> holds the published `*78keV*autoresearch*.tif` and `valid113.npy`
(see the top-level README's Reproduce section for the S3 sources).
"""
import argparse
import glob
import os

import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

REGION = (1050, 900, 1300, 400)  # x, y, w, h — frozen fig1 region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--pred", action="append", required=True,
                    help="prediction .npy on the full 113 keV grid (repeatable)")
    args = ap.parse_args()

    tifs = glob.glob(os.path.join(args.seg_dir, "*78keV*autoresearch*.tif"))
    assert len(tifs) == 1, tifs
    import tifffile
    valid = np.load(os.path.join(args.seg_dir, "valid113.npy")).astype(bool)
    hh, ww = valid.shape
    teacher = cv2.resize(tifffile.imread(tifs[0]), (ww, hh),
                         interpolation=cv2.INTER_AREA)
    x, y, w, h = REGION
    sl = np.s_[y:y + h, x:x + w]
    vv = valid[sl]
    ink = (teacher[sl] >= 128) & vv
    bg = (teacher[sl] < 128) & vv
    lab, n = ndimage.label(ink)
    sizes = ndimage.sum(ink, lab, range(1, n + 1))
    comps = [i + 1 for i, s in enumerate(sizes) if s >= 1000]

    for path in args.pred:
        p = np.load(path)[sl].astype(np.float64)
        r = rankdata(np.concatenate([p[ink], p[bg]]))
        n_i, n_b = int(ink.sum()), int(bg.sum())
        auc = (r[:n_i].sum() - n_i * (n_i + 1) / 2) / (n_i * n_b)
        thr = np.quantile(p[bg], 0.95)
        rec = (p[ink] >= thr).mean()
        hit = sum(1 for c in comps if (p[lab == c] >= thr).mean() >= 0.30)
        print(f"{os.path.basename(path)}: AUC {auc:.4f}  "
              f"recall@5%FP {100 * rec:.1f}%  components {hit}/{len(comps)}")


if __name__ == "__main__":
    main()
