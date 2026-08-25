#!/usr/bin/env python3
"""Native-113keV benchmark for PHerc0139 ink maps, scored against the
organization's released 78 keV ink-detection maps.

PHerc0139 was scanned at several energies. At 78 keV / 2.399 um the ink is
strong enough that the organization's released ink-detection maps
(`ink-detection/*78keV*autoresearch*.tif` on the public S3 bucket) show the
text clearly; at 113 keV / 9.362 um it is much fainter. That gap makes a
clean benchmark: run a model on the native 113 keV surface volume of a
segment it was never trained on, and score it against confident ink /
confident background classes taken from the public 78 keV map of the same
segment.

Class construction per segment (frozen before any model comparison):
  domain  = valid & (teacher > 0)          valid = max over z of the volume > 0
  no-ink  = teacher <= lo                  lo = smallest uint8 value holding
                                                >= 25% of the domain
  ink     = teacher >= hi                  hi = max(128, P85 of the domain)
The band between lo and hi is ignored. AUC is tie-aware rank AUC
(Mann-Whitney), subsampled to at most 20M pixels with a fixed seed.

Modes:
  metrics     AUC of one prediction map on one segment (+ misregistration
              control: the same AUC with the classes shifted ~2 mm and
              rotated 90 degrees must collapse toward 0.5, otherwise the
              score rides on region/texture shortcuts, not ink):
      native_bench.py metrics --seg-dir DIR --pred PROB.npy [--out out.json]

  fp-control  false-positive control on a known-blank segment: pick the
              threshold theta* that recalls 80% of the teacher's ink
              components (>= 1597 px = 0.14 mm^2) on an ink-bearing segment,
              then report the fraction of a blank segment's valid area
              flagged at theta*:
      native_bench.py fp-control --ink-seg-dir DIR47 --ink-pred P47.npy \
          --blank-seg-dir DIR53 --blank-pred P53.npy [--out out.json]

Each segment directory must contain:
  zarr113/                                the segment's 113 keV surface-volume
                                          zarr (9.362um-1.2m-113keV-*.zarr
                                          from the public S3 bucket); a cached
                                          valid113.npy is written next to it
  *78keV*autoresearch*.tif                the org's released 78 keV ink map
                                          for the same segment (same bucket,
                                          ink-detection/)

Predictions: .npy float maps in [0,1] on the 113 keV pixel grid, or the
uint8 .tif written by the ink_9um inference code (rescaled (v-64)/128).

Requires: numpy, opencv-python, tifffile, zarr.
"""
import argparse
import glob
import json
import os

import numpy as np

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "109951162777600")
import cv2  # noqa: E402
import tifffile  # noqa: E402

LO_FRAC = 0.25          # no-ink class holds >= 25% of the domain
HI_PCT = 85             # hi = max(128, P85)
HI_MIN = 128
SEED_EVAL = 20260803    # subsampling seed (only used above MAX_N pixels)
MAX_N = 20_000_000
MISREG_SHIFT_PX = 214   # ~2 mm at 9.362 um/px
FP_COMP_MIN_PX = 1597   # 0.14 mm^2 at 9.362 um/px
FP_RECALL_FRAC = 0.10   # component recalled iff >=10% of its px >= theta
FP_TEACHER_RECALL = 0.80


# ---------------------------------------------------------------- data

def load_valid(seg_dir):
    """valid = max over z of the 113 keV volume > 0; cached as valid113.npy."""
    cpath = os.path.join(seg_dir, "valid113.npy")
    if os.path.exists(cpath):
        return np.load(cpath) > 0
    import zarr
    z = zarr.open(os.path.join(seg_dir, "zarr113"), mode="r")
    a = z["0"] if hasattr(z, "keys") and "0" in z else z
    d, h, w = a.shape
    valid = np.zeros((h, w), bool)
    step = 1024
    for y in range(0, h, step):
        valid[y:y + step] = np.asarray(a[:, y:y + step, :]).max(axis=0) > 0
    np.save(cpath, valid)
    return valid


def load_teacher(seg_dir, shape_hw):
    """The org's released 78 keV ink map, resized onto the 113 keV grid."""
    hits = glob.glob(os.path.join(seg_dir, "*78keV*autoresearch*.tif"))
    assert len(hits) == 1, f"need exactly one 78keV autoresearch tif in {seg_dir}: {hits}"
    t = tifffile.imread(hits[0])
    if t.ndim == 3:
        t = t[..., 0]
    if t.dtype == np.uint16:
        t = (t >> 8).astype(np.uint8)
    h, w = shape_hw
    return cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA)


def load_pred(path, shape_hw=None):
    if path.endswith(".npy"):
        p = np.load(path).astype(np.float32)
    else:
        a = tifffile.imread(path).astype(np.float32)
        p = np.clip((a - 64.0) / 128.0, 0.0, 1.0)   # model-card uint8 scaling
    if shape_hw is not None:
        assert p.shape == tuple(shape_hw), f"pred {p.shape} != segment {shape_hw}"
    return p


# ---------------------------------------------------------------- metric

def freeze_thresholds(t, valid):
    dom = valid & (t > 0)
    vals = t[dom]
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    cdf = np.cumsum(hist) / max(vals.size, 1)
    lo = int(np.searchsorted(cdf, LO_FRAC))
    hi = max(HI_MIN, int(np.searchsorted(cdf, HI_PCT / 100.0)))
    return lo, hi


def classes(t, valid, lo, hi):
    dom = valid & (t > 0)
    return (t >= hi) & dom, (t <= lo) & dom


def roc_auc(y_true, scores, max_n=MAX_N, seed=SEED_EVAL):
    """Rank-based ROC-AUC (Mann-Whitney U, tie-corrected). numpy only."""
    y = np.asarray(y_true).ravel().astype(bool)
    s = np.asarray(scores).ravel().astype(np.float64)
    if y.size > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(y.size, size=max_n, replace=False)
        y, s = y[idx], s[idx]
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    s_sorted = s[order]
    uniq, first = np.unique(s_sorted, return_index=True)
    counts = np.diff(np.append(first, s_sorted.size))
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def misreg_auc(scores, ink, noink, valid):
    """Shift the classes ~2 mm / rotate 90 degrees; real ink signal collapses
    toward 0.5, region/texture shortcuts keep scoring."""
    out = {}
    for name, tf in (
            ("shift_y", lambda m: np.roll(m, MISREG_SHIFT_PX, axis=0)),
            ("shift_x", lambda m: np.roll(m, MISREG_SHIFT_PX, axis=1)),
            ("rot90", lambda m: cv2.resize(np.rot90(m).astype(np.uint8),
                                           (m.shape[1], m.shape[0]),
                                           interpolation=cv2.INTER_NEAREST).astype(bool))):
        ink_t = tf(ink) & valid
        no_t = tf(noink) & valid & ~ink_t
        sel = ink_t | no_t
        if ink_t.sum() < 1000 or no_t.sum() < 1000:
            out[name] = float("nan")
            continue
        out[name] = roc_auc(ink_t[sel], np.asarray(scores)[sel])
    vals = [v for v in out.values() if v == v]
    out["max"] = max(vals) if vals else float("nan")
    return out


# ---------------------------------------------------------------- modes

def metrics(a):
    valid = load_valid(a.seg_dir)
    t = load_teacher(a.seg_dir, valid.shape)
    lo, hi = freeze_thresholds(t, valid)
    p = load_pred(a.pred, valid.shape)
    ink, noink = classes(t, valid, lo, hi)
    sel = ink | noink
    auc = roc_auc(ink[sel], p[sel])
    out = {"seg_dir": a.seg_dir, "pred": a.pred, "lo": lo, "hi": hi,
           "n_ink": int(ink.sum()), "n_noink": int(noink.sum()),
           "auc_vs_78keV": auc,
           "misreg": misreg_auc(p, ink, noink, valid)}
    print(f"AUC vs 78 keV classes: {auc:.4f}  "
          f"(lo={lo} hi={hi} n_ink={out['n_ink']} n_noink={out['n_noink']})")
    print(f"misregistration control max: {out['misreg']['max']:.4f} "
          f"(must sit well below the real AUC)")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return out


def fp_control(a):
    v_ink = load_valid(a.ink_seg_dir)
    t = load_teacher(a.ink_seg_dir, v_ink.shape)
    lo, hi = freeze_thresholds(t, v_ink)
    ink, _ = classes(t, v_ink, lo, hi)
    n_lab, cc = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    sizes = np.bincount(cc.ravel())
    keep = [i for i in range(1, n_lab) if sizes[i] >= FP_COMP_MIN_PX]
    p_ink = load_pred(a.ink_pred, v_ink.shape)
    comp_scores = np.array([np.percentile(p_ink[cc == i],
                                          100 * (1 - FP_RECALL_FRAC))
                            for i in keep])
    theta = float(np.percentile(comp_scores, 100 * (1 - FP_TEACHER_RECALL)))
    recall = float((comp_scores >= theta).mean())

    v_blank = load_valid(a.blank_seg_dir)
    p_blank = load_pred(a.blank_pred, v_blank.shape)
    flag_frac = float((p_blank[v_blank] >= theta).mean())
    out = {"theta_star": theta, "component_recall": recall,
           "n_components": len(keep),
           "blank_flagged_valid_frac": flag_frac}
    print(f"theta*={theta:.4f} recall={recall:.3f} "
          f"({len(keep)} teacher components >= {FP_COMP_MIN_PX} px)")
    print(f"blank segment flagged at theta*: {flag_frac:.4f} of valid area")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("metrics")
    p.add_argument("--seg-dir", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--out")
    p = sub.add_parser("fp-control")
    p.add_argument("--ink-seg-dir", required=True)
    p.add_argument("--ink-pred", required=True)
    p.add_argument("--blank-seg-dir", required=True)
    p.add_argument("--blank-pred", required=True)
    p.add_argument("--out")
    a = ap.parse_args()
    {"metrics": metrics, "fp-control": fp_control}[a.mode](a)


if __name__ == "__main__":
    main()
