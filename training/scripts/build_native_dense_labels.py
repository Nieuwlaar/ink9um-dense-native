#!/usr/bin/env python
"""Build dense pseudo-labels for the native-113keV PHerc0139 training segments from the
org's 78 keV teacher maps, in the official ink_9um native9 label layout.

Per segment w (all data_r2b segments EXCEPT the held-out benchmark title/w024/w047/w053):
  teacher  = 78 keV canonical-2.4um ink map, INTER_AREA onto the zarr113 (H,W) grid
  valid    = max_z(zarr113) > 0                                  (valid113.npy)
  inklabels        = teacher >= theta(w)        (255/0, plane z=14 only)
                     [manual native9 labels win where manual supervision exists:
                      w040/w041 carry official sparse labels -> KLAVIS's rule]
  supervision_mask = valid & (teacher > 0)      (255/0, plane z=14 only)
                     = the teacher-valid region (~85% of canvas, 99.99% of valid)
  no validation_mask (validation is the held-out native benchmark, native-eval/native_bench.py)

theta(w): --theta-policy calibrated (default) uses data/theta_calibration.json:
  per-segment t* where the segment has manual labels (w040, w041), else the median t*
  over the 5 native9 calibration segments. --theta N forces one value everywhere.

Output: data/labels/native-dense-0139-78keV/<w>/<w>_{inklabels,supervision_mask}.zarr
        data/native_dense_report.json (coverage / ink fractions per segment)
Idempotent per segment (skips when both zarrs + report entry exist; --force rebuilds).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_common import (DATA_R2B, D_NATIVE, HELD_OUT, NATIVE9_LABELS, NATIVE_Z, W,  # noqa: E402
                           load_valid, log, open_zarr_array, teacher_on_grid,
                           training_wnames, write_native_label_zarr)

OUT_ROOT = os.path.join(W, "data", "labels", "native-dense-0139-78keV")
REPORT = os.path.join(W, "data", "native_dense_report.json")
CALIB = os.path.join(W, "data", "theta_calibration.json")


def theta_for(w, args, calib):
    if args.theta is not None:
        return int(args.theta), "forced"
    if w in calib and "t_star" in calib[w]:
        return int(calib[w]["t_star"]), "own-calibration"
    return int(calib["_summary"]["median_t_star"]), "median-of-native9-calibration"


def build(w, sdir, args, calib, report):
    out_dir = os.path.join(OUT_ROOT, w)
    ink_p = os.path.join(out_dir, f"{w}_inklabels.zarr")
    sup_p = os.path.join(out_dir, f"{w}_supervision_mask.zarr")
    if not args.force and os.path.isdir(ink_p) and os.path.isdir(sup_p) and w in report:
        log(f"[{w}] exists, skipping")
        return
    os.makedirs(out_dir, exist_ok=True)
    zpath = os.path.join(sdir, "zarr113")
    vol = open_zarr_array(zpath)
    d, h, wd = vol.shape
    assert d == D_NATIVE, (w, vol.shape)
    valid = load_valid(zpath, cache_path=os.path.join(sdir, "valid113.npy"))
    assert valid.shape == (h, wd)
    hits = glob.glob(os.path.join(sdir, "*78keV*autoresearch*.tif"))
    assert len(hits) == 1, (w, hits)
    t = teacher_on_grid(hits[0], (h, wd))
    theta, how = theta_for(w, args, calib)

    teacher_ink = t >= theta
    sup = valid & (t > 0)
    ink = teacher_ink & sup
    manual_note = None
    man_dir = os.path.join(NATIVE9_LABELS, w)
    if os.path.isdir(man_dir):
        m_ink = np.asarray(open_zarr_array(os.path.join(man_dir, f"{w}_inklabels.zarr"))[NATIVE_Z]) > 0
        m_sup = np.asarray(open_zarr_array(os.path.join(man_dir, f"{w}_supervision_mask.zarr"))[NATIVE_Z]) > 0
        assert m_ink.shape == (h, wd), (m_ink.shape, (h, wd))
        agree = float((teacher_ink[m_sup] == m_ink[m_sup]).mean())
        ink = np.where(m_sup, m_ink, ink)
        sup = sup | m_sup
        manual_note = {"manual_sup_frac_canvas": round(float(m_sup.mean()), 4),
                       "manual_ink_px": int(m_ink.sum()),
                       "teacher_vs_manual_pixel_agreement_on_manual_sup": round(agree, 4)}

    rec = {
        "segment_dir": os.path.basename(sdir), "shape_zyx": [d, h, wd],
        "theta": theta, "theta_source": how,
        "valid_frac_canvas": round(float(valid.mean()), 4),
        "supervision_frac_canvas": round(float(sup.mean()), 4),
        "supervision_frac_of_valid": round(float(sup.sum() / max(1, valid.sum())), 5),
        "ink_frac_canvas": round(float(ink.mean()), 4),
        "ink_frac_of_supervision": round(float(ink.sum() / max(1, sup.sum())), 4),
        "supervised_px": int(sup.sum()), "ink_px": int(ink.sum()),
        "teacher_tif": os.path.basename(hits[0]), "manual_labels_merged": manual_note,
    }
    attrs = {"source": "native-dense-0139-78keV", "teacher": rec["teacher_tif"],
             "teacher_resample": "cv2.INTER_AREA 2.399um->9.362um grid", "theta": theta,
             "theta_source": how, "annotated_plane": NATIVE_Z, "layout": "official native9 21slices (28-deep, z=14)"}
    write_native_label_zarr(ink_p, (ink * 255).astype(np.uint8), (d, h, wd), attrs)
    write_native_label_zarr(sup_p, (sup * 255).astype(np.uint8), (d, h, wd), attrs)
    # verify round trip
    a = open_zarr_array(ink_p); b = open_zarr_array(sup_p)
    assert tuple(a.shape) == (d, h, wd) == tuple(b.shape)
    assert int((np.asarray(a[NATIVE_Z]) > 0).sum()) == rec["ink_px"]
    assert int((np.asarray(b[NATIVE_Z]) > 0).sum()) == rec["supervised_px"]
    assert int(np.asarray(a[0]).sum()) == 0 and int(np.asarray(b[NATIVE_Z - 3]).sum()) == 0
    report[w] = rec
    json.dump(report, open(REPORT, "w"), indent=1)
    log(f"[{w}] theta={theta} ({how}) sup {rec['supervision_frac_canvas']*100:.1f}% canvas "
        f"ink {rec['ink_frac_of_supervision']*100:.1f}% of sup -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segs", default=None, help="comma list; default = all training wnames")
    ap.add_argument("--theta", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    train, segs = training_wnames()
    todo = args.segs.split(",") if args.segs else train
    for w in todo:
        assert w not in HELD_OUT, f"{w} is HELD OUT"
        assert w in segs, w
    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    if args.theta is None:
        assert "_summary" in calib, "run calibrate_theta.py first (or pass --theta)"
    report = json.load(open(REPORT)) if os.path.exists(REPORT) else {}
    log(f"training wnames: {train}  (held out: {HELD_OUT})")
    for w in todo:
        build(w, os.path.join(DATA_R2B, segs[w]), args, calib, report)
    log("done")


if __name__ == "__main__":
    main()
