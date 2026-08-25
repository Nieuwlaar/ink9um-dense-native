#!/usr/bin/env python3
"""Compute all AUCs and write official_eval.csv.

Rows: segment, checkpoint, auc, n_pos, n_neg
Checkpoints: released, soup42 (default center window) and released_zavg5,
soup42_zavg5 (mean of the five 17-of-21 z windows) when all five exist.
"""
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEGS = ["pherc0139-w016", "pherc0814-46527", "pherc1667-w029"]
PY = sys.executable


def run_eval(seg, tifs):
    out = subprocess.run(
        [PY, str(ROOT / "scripts" / "eval_auc.py"), seg] + [str(t) for t in tifs],
        capture_output=True, text=True, check=True,
    )
    line = out.stdout.strip().splitlines()[-1]
    parts = dict(kv.split("=") for kv in line.split()[1:])
    sys.stderr.write(out.stderr)
    return float(parts["auc"]), int(parts["n_pos"]), int(parts["n_neg"])


def main():
    rows = []
    for seg in SEGS:
        for name in ["released", "soup42"]:
            tif = ROOT / "preds" / f"{seg}__{name}.tif"
            if not tif.exists():
                print(f"missing {tif}", file=sys.stderr)
                continue
            auc, npos, nneg = run_eval(seg, [tif])
            rows.append([seg, name, f"{auc:.4f}", npos, nneg])
            print(f"{seg} {name} auc={auc:.4f}")
        for name in ["released", "soup42"]:
            tifs = [ROOT / "preds" / f"{seg}__{name}.tif"] + [
                ROOT / "preds" / f"{seg}__{name}_z{k}.tif" for k in [0, 1, 3, 4]
            ]
            if not all(t.exists() for t in tifs):
                continue
            auc, npos, nneg = run_eval(seg, tifs)
            rows.append([seg, f"{name}_zavg5", f"{auc:.4f}", npos, nneg])
            print(f"{seg} {name}_zavg5 auc={auc:.4f}")

    with open(ROOT / "official_eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "checkpoint", "auc", "n_pos", "n_neg"])
        w.writerows(rows)
    print(f"wrote {ROOT / 'official_eval.csv'} ({len(rows)} rows)")

    # means per checkpoint over the three segments
    for name in ["released", "soup42", "released_zavg5", "soup42_zavg5"]:
        vals = [float(r[2]) for r in rows if r[1] == name]
        if len(vals) == len(SEGS):
            print(f"mean {name}: {sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    main()
