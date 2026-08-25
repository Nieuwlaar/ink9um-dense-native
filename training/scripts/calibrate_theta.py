#!/usr/bin/env python
"""Calibrate the dense-label threshold theta for the 78 keV teacher on native 113 keV.

KLAVIS's rule: per segment, t* = argmax balanced accuracy of (teacher >= t) vs the
MANUAL labels, evaluated on the manual supervision region only. Manual native labels
exist for exactly the 5 official native9 segments (w035 w039 w040 w041 w044), so the
script calibrates there and reports: t*, BA(t*), teacher AUC vs manual, ink fraction of
valid at t*, and an optional lo/hi context table. Output: data/theta_calibration.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_common import (DATA_R2B, NATIVE9_LABELS, NATIVE9_VOLUMES, NATIVE_Z,  # noqa: E402
                           R2B_THRESH, W, balanced_accuracy_curve, load_valid, log,
                           open_zarr_array, r2b_segments, teacher_on_grid)
from common0139 import roc_auc  # noqa: E402

CAL = ["w035", "w039", "w040", "w041", "w044"]


def teacher_tif_for(w):
    segs = r2b_segments()
    if w in segs:
        import glob
        hits = glob.glob(os.path.join(DATA_R2B, segs[w], "*78keV*autoresearch*.tif"))
        assert len(hits) == 1, hits
        return hits[0]
    p = os.path.join(W, "data", "teacher", f"{w}.tif")
    assert os.path.exists(p), p
    return p


def main():
    out = {}
    # optional lo/hi context table (NATIVE113_THRESHOLDS); absent -> r2b_lo_hi = None
    th = json.load(open(R2B_THRESH)) if R2B_THRESH and os.path.exists(R2B_THRESH) else {}
    for w in CAL:
        vol = os.path.join(NATIVE9_VOLUMES, f"{w}.zarr")
        valid = load_valid(vol, cache_path=os.path.join(W, "data", f"valid113_{w}.npy"))
        h, wd = valid.shape
        t = teacher_on_grid(teacher_tif_for(w), (h, wd))
        ink = np.asarray(open_zarr_array(os.path.join(NATIVE9_LABELS, w, f"{w}_inklabels.zarr"))[NATIVE_Z]) > 0
        sup = np.asarray(open_zarr_array(os.path.join(NATIVE9_LABELS, w, f"{w}_supervision_mask.zarr"))[NATIVE_Z]) > 0
        assert ink.shape == (h, wd), (ink.shape, (h, wd))
        dom = sup & valid
        t_star, ba_star, ba, n_pos, n_neg = balanced_accuracy_curve(t, ink, dom)
        auc = roc_auc(ink[dom], t[dom].astype(np.float64))
        tv = t[valid]
        rec = {
            "t_star": t_star, "ba_at_t_star": round(ba_star, 4),
            "ba_at_128": round(float(ba[128]), 4), "ba_at_64": round(float(ba[64]), 4),
            "teacher_auc_vs_manual_on_supervision": round(auc, 4),
            "n_pos": n_pos, "n_neg": n_neg, "sup_frac_canvas": round(float(sup.mean()), 4),
            "valid_frac_canvas": round(float(valid.mean()), 4),
            "ink_frac_valid_at_t_star": round(float((tv >= t_star).mean()), 4),
            "ink_frac_valid_at_128": round(float((tv >= 128).mean()), 4),
            "manual_ink_frac_of_sup": round(float(ink[dom].mean()), 4),
            "r2b_lo_hi": ([th[w]["teachers"]["78"]["lo"], th[w]["teachers"]["78"]["hi"]] if w in th else None),
            "teacher_tif": teacher_tif_for(w), "shape_hw": [h, wd],
        }
        out[w] = rec
        log(f"{w}: t*={t_star} BA*={ba_star:.4f} (BA@128 {ba[128]:.4f}) AUC={auc:.4f} "
            f"ink@t* {rec['ink_frac_valid_at_t_star']:.3f} ink@128 {rec['ink_frac_valid_at_128']:.3f} "
            f"manual ink/sup {rec['manual_ink_frac_of_sup']:.3f} r2b lo/hi {rec['r2b_lo_hi']}")
    ts = [out[w]["t_star"] for w in CAL]
    out["_summary"] = {"t_star_values": ts, "median_t_star": int(np.median(ts)),
                       "mean_t_star": round(float(np.mean(ts)), 1)}
    json.dump(out, open(os.path.join(W, "data", "theta_calibration.json"), "w"), indent=1)
    log(f"summary: t* {ts} median {out['_summary']['median_t_star']}")


if __name__ == "__main__":
    main()
