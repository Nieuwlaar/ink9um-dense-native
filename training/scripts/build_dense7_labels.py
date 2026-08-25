#!/usr/bin/env python
"""Rebuild KLAVIS's dense pseudo-label set for his 7 aligned segments from PUBLIC data.

The label sets are rebuilt rather than downloaded from his
domenicor046/ink9um-dense-labels tar: re-derived with his published recipe
(DomRusso2/ink9um-dense README s6 + scripts/build_pseudo_labels.py):

  teacher     = canonical-2.4um ink model on the segment's public 2.4um surface volume,
                pooled 4x to the 9.6um level-2 raster.  This uses the org's PUBLISHED run of
                that model (S3 <segment>/ink-detection/*new_canon_autoresearch_recipe*.tif,
                uint8 on the 2.399um L0 grid) and pool it with cv2.INTER_AREA /4 (exact 4x4
                block mean; the L0 grid is exactly 4x the level-2 grid).
  t*          = argmax balanced accuracy of (teacher >= t) vs the manual inklabels over the
                manual supervision region only (his calibration rule; recomputed here since
                his teacher run had a different intensity scaling [clip(0,200)/200]).
  inklabels   = manual label where manual supervision exists, else teacher >= t*
  supervision = render-valid (iso9 plane z=10 > 0) AND NOT validation_mask
  validation  = copied verbatim where one exists (w016, 0814, 1667-w029)
  layout      = same shape/chunks/dtype/fill/compressor as the real v2 label zarrs
                ((21,H,W), (21,128,128), u1, zstd-5 bitshuffle), values 0/1 like the
                bucket, content on z=10 only.

Gates per segment (his): shape identical; sup 30-99% & ink 1-60% at z=10; sup AND val == 0;
planes 0/20 empty.  Output: data/labels/dense7-aligned-rebuilt/<seg>/ + data/dense7_report.json
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "109951162777600")
import cv2  # noqa: E402
import tifffile  # noqa: E402
import zarr  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_common import W, balanced_accuracy_curve, log, open_zarr_array  # noqa: E402
from common0139 import roc_auc  # noqa: E402

REAL = os.path.join(W, "data", "labels", "aligned-scrollprizeorg-21slices")
OUT = os.path.join(W, "data", "labels", "dense7-aligned-rebuilt")
ISO = os.path.join(W, "data", "volumes", "aligned-iso9")
TEACH = os.path.join(W, "data", "teacher")
REPORT = os.path.join(W, "data", "dense7_report.json")
Z = 10
KLAVIS_TSTAR = {  # his README s6 (fraction of his clip(0,200)/200-scaled map)
    "pherc0139-w016": 0.15, "pherc0139-w017": 0.33, "pherc0139-w028": 0.27,
    "pherc0139-w029": 0.25, "pherc0814-46527": 0.53, "pherc1667-w028": 0.45,
    "pherc1667-w029": 0.36,
}
DENSE7 = list(KLAVIS_TSTAR)
BAND = 1024


def write_like(src_path, dst_path, plane_u8):
    src = open_zarr_array(src_path)
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    g = zarr.open_group(dst_path, mode="w")
    dst = g.create_dataset("0", shape=tuple(src.shape), chunks=tuple(src.chunks),
                           dtype=src.dtype, fill_value=src.fill_value,
                           compressor=src.compressor, overwrite=True)
    h = int(src.shape[1])
    for y0 in range(0, h, BAND):
        y1 = min(y0 + BAND, h)
        dst[Z, y0:y1, :] = plane_u8[y0:y1, :]
    try:
        dst.attrs.update(dict(src.attrs))
        g.attrs.update(dict(zarr.open(src_path, mode="r").attrs))
    except Exception:  # noqa: BLE001
        pass
    return dst


def build(seg, force, report):
    d_real, d_out = os.path.join(REAL, seg), os.path.join(OUT, seg)
    os.makedirs(d_out, exist_ok=True)
    real_ink = os.path.join(d_real, f"{seg}_inklabels.zarr")
    real_sup = os.path.join(d_real, f"{seg}_supervision_mask.zarr")
    real_val = os.path.join(d_real, f"{seg}_validation_mask.zarr")
    out_ink = os.path.join(d_out, f"{seg}_inklabels.zarr")
    out_sup = os.path.join(d_out, f"{seg}_supervision_mask.zarr")
    out_val = os.path.join(d_out, f"{seg}_validation_mask.zarr")
    if not force and os.path.isdir(out_ink) and os.path.isdir(out_sup) and seg in report:
        log(f"[{seg}] exists, verifying only")
        return verify(seg, report)
    assert os.path.exists(os.path.join(d_real, "FETCH_OK")), f"{seg}: labels not fetched"
    iso = open_zarr_array(os.path.join(ISO, f"{seg}.zarr"))
    canvas = tuple(int(v) for v in open_zarr_array(real_ink).shape[1:])
    assert tuple(iso.shape[1:]) == canvas, (iso.shape, canvas)

    manual_ink = np.asarray(open_zarr_array(real_ink)[Z]) > 0
    manual_sup = np.asarray(open_zarr_array(real_sup)[Z]) > 0
    val = (np.asarray(open_zarr_array(real_val)[Z]) > 0) if os.path.exists(real_val) else np.zeros(canvas, bool)
    render_valid = np.asarray(iso[Z]) > 0

    t0 = tifffile.imread(os.path.join(TEACH, f"{seg}.tif"))
    if t0.dtype == np.uint16:
        t0 = (t0 >> 8).astype(np.uint8)
    ratio = (t0.shape[0] / canvas[0], t0.shape[1] / canvas[1])
    teacher = cv2.resize(t0, (canvas[1], canvas[0]), interpolation=cv2.INTER_AREA)
    del t0

    cal_dom = manual_sup & render_valid & ~val
    t_star, ba_star, ba, n_pos, n_neg = balanced_accuracy_curve(teacher, manual_ink, cal_dom)
    auc = roc_auc(manual_ink[cal_dom], teacher[cal_dom].astype(np.float64))
    teacher_ink = teacher >= t_star
    pseudo_ink = np.where(manual_sup, manual_ink, teacher_ink) & render_valid
    pseudo_sup = render_valid & ~val

    write_like(real_ink, out_ink, pseudo_ink.astype(np.uint8))
    write_like(real_sup, out_sup, pseudo_sup.astype(np.uint8))
    if os.path.exists(real_val):
        if os.path.exists(out_val):
            shutil.rmtree(out_val)
        shutil.copytree(real_val, out_val)
    rec = {
        "canvas": list(canvas), "teacher_l0_to_l2_ratio": [round(r, 4) for r in ratio],
        "t_star_u8": t_star, "t_star_frac": round(t_star / 255.0, 3), "klavis_t_star": KLAVIS_TSTAR[seg],
        "ba_at_t_star": round(ba_star, 4), "teacher_auc_vs_manual": round(auc, 4),
        "cal_n_pos": n_pos, "cal_n_neg": n_neg,
        "manual_sup_frac": round(float(manual_sup.mean()), 4), "manual_px": int(manual_sup.sum()),
        "pseudo_sup_frac": round(float(pseudo_sup.mean()), 4), "pseudo_px": int(pseudo_sup.sum()),
        "multiplier": round(float(pseudo_sup.sum() / max(1, manual_sup.sum())), 1),
        "teacher_ink_frac_at_t_star_canvas": round(float((teacher_ink & render_valid).mean()), 4),
        "pseudo_ink_frac_canvas": round(float(pseudo_ink.mean()), 4),
        "pseudo_ink_frac_of_sup": round(float(pseudo_ink.sum() / max(1, pseudo_sup.sum())), 4),
        "has_validation": bool(os.path.exists(real_val)), "val_px": int(val.sum()),
    }
    report[seg] = rec
    json.dump(report, open(REPORT, "w"), indent=1)
    log(f"[{seg}] t*={t_star} ({t_star/255:.3f}; his {KLAVIS_TSTAR[seg]}) BA*={ba_star:.4f} AUC={auc:.4f} "
        f"sup {rec['manual_sup_frac']*100:.2f}% -> {rec['pseudo_sup_frac']*100:.1f}% ({rec['multiplier']}x) "
        f"ink {rec['pseudo_ink_frac_of_sup']*100:.1f}% of sup")
    return verify(seg, report)


def verify(seg, report):
    d_real, d_out = os.path.join(REAL, seg), os.path.join(OUT, seg)
    real_ink = open_zarr_array(os.path.join(d_real, f"{seg}_inklabels.zarr"))
    ink = open_zarr_array(os.path.join(d_out, f"{seg}_inklabels.zarr"))
    sup = open_zarr_array(os.path.join(d_out, f"{seg}_supervision_mask.zarr"))
    ok = tuple(ink.shape) == tuple(real_ink.shape) == tuple(sup.shape)
    ok &= tuple(ink.chunks) == tuple(real_ink.chunks) and ink.dtype == real_ink.dtype
    ink_p = np.asarray(ink[Z]) > 0; sup_p = np.asarray(sup[Z]) > 0
    ink_f, sup_f = float(ink_p.mean()), float(sup_p.mean())
    ok &= (0.30 <= sup_f <= 0.99) and (0.01 <= ink_f <= 0.60)
    real_val = os.path.join(d_real, f"{seg}_validation_mask.zarr")
    overlap = None
    if os.path.exists(real_val):
        val_p = np.asarray(open_zarr_array(real_val)[Z]) > 0
        overlap = int((sup_p & val_p).sum()); ok &= overlap == 0
        ok &= os.path.isdir(os.path.join(d_out, f"{seg}_validation_mask.zarr"))
    other = int(np.asarray(ink[0]).sum()) + int(np.asarray(ink[20]).sum()) + int(np.asarray(sup[0]).sum())
    ok &= other == 0
    report.setdefault(seg, {})["verify"] = {"pass": bool(ok), "sup_frac_z10": round(sup_f, 4),
                                            "ink_frac_z10": round(ink_f, 4), "sup_and_val_px": overlap,
                                            "other_planes_nonzero": other}
    json.dump(report, open(REPORT, "w"), indent=1)
    log(f"[{seg}] verify: {'PASS' if ok else 'FAIL'} sup {sup_f:.3f} ink {ink_f:.3f} sup&val {overlap} other {other}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    report = json.load(open(REPORT)) if os.path.exists(REPORT) else {}
    res = {}
    for seg in (a.only or DENSE7):
        res[seg] = verify(seg, report) if a.verify_only else build(seg, a.force, report)
    log("SUMMARY " + " ".join(f"{s}:{'PASS' if ok else 'FAIL'}" for s, ok in res.items()))
    sys.exit(0 if all(res.values()) else 1)


if __name__ == "__main__":
    main()
