#!/usr/bin/env python3
"""ROC-AUC of an ink prediction on the official validation mask (minus supervision).

Usage: eval_auc.py <segment> <pred_tif> [<pred_tif2> ...]
Multiple prediction TIFFs are averaged (z-window ensemble) before scoring.
Pixels scored: validation_mask[10] > 0 AND supervision_mask[10] == 0.
Positives: inklabels[10] > 0. AUC is tie-aware (Mann-Whitney via average ranks).
Prints: segment auc n_pos n_neg [extra diagnostics to stderr].
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr

ROOT = Path(__file__).resolve().parent.parent
Z = 10  # annotated slice of the aligned 21-slice arrays


def expand_if_crop(seg: str, pred: np.ndarray, full_shape) -> np.ndarray:
    """Crop-run predictions are pasted back at their recorded offset."""
    if tuple(pred.shape) == tuple(full_shape):
        return pred
    crops_file = ROOT / "data" / "crops.json"
    crops = json.loads(crops_file.read_text())
    c = crops[seg]
    assert pred.shape == (c["y1"] - c["y0"], c["x1"] - c["x0"]), (
        f"pred shape {pred.shape} matches neither full {full_shape} nor crop {c}"
    )
    full = np.zeros(tuple(full_shape), dtype=pred.dtype)
    full[c["y0"]:c["y1"], c["x0"]:c["x1"]] = pred
    return full


def auc_mannwhitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware ROC-AUC via average ranks."""
    scores = scores.astype(np.float64)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    # average ranks for ties
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [scores.size]))
    for s, e in zip(starts, ends):
        ranks[order[s:e]] = 0.5 * (s + e - 1) + 1.0
    rank_sum_pos = ranks[labels].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def load_plane(seg: str, kind: str) -> np.ndarray:
    path = ROOT / "data" / "labels" / seg / f"{seg}_{kind}.zarr"
    arr = zarr.open(str(path), mode="r")["0"]
    return np.asarray(arr[Z])


def main():
    seg = sys.argv[1]
    preds = []
    for tif in sys.argv[2:]:
        preds.append(tifffile.imread(tif).astype(np.float32))
    pred = np.mean(preds, axis=0)

    labels = load_plane(seg, "inklabels")
    valid = load_plane(seg, "validation_mask")
    sup = load_plane(seg, "supervision_mask")
    pred = expand_if_crop(seg, pred, labels.shape)
    assert pred.shape == labels.shape, f"shape mismatch pred={pred.shape} labels={labels.shape}"

    eval_mask = (valid > 0) & (sup == 0)
    overlap = int(((valid > 0) & (sup > 0)).sum())
    y = (labels[eval_mask] > 0)
    s = pred[eval_mask]
    auc = auc_mannwhitney(s, y)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    print(f"{seg} auc={auc:.4f} n_pos={n_pos} n_neg={n_neg}")
    print(
        f"  [diag] valid_px={int((valid>0).sum())} sup_px={int((sup>0).sum())} "
        f"valid&sup_overlap={overlap} label_vals={np.unique(labels)[:5].tolist()} "
        f"valid_vals={np.unique(valid).tolist()} sup_vals={np.unique(sup).tolist()}",
        file=sys.stderr,
    )
    return auc


if __name__ == "__main__":
    main()
