# Weights: provenance and post-strip verification

Both files are **bare safetensors state_dicts**: 508 tensors, 68,175,426
parameters, float32, no optimizer state, no scheduler, no training config,
no metadata block. The safetensors header was inspected after writing: it
contains only tensor names (`network.*`, `depth_fusion.*`), dtypes, shapes
and byte offsets. Nothing else.

| file | bytes | sha256 |
|---|---|---|
| `soup42_last4.safetensors` | 272,767,904 | `ab3f1bdefa53733f4d4c46da6c7713c142bc8c703df5098fe8641757d764849c` |
| `dense_native_016000.safetensors` | 272,767,904 | `d3d95dc41dcb24ed76b84b5e12ca5dae25d7fb34edf1dfbb93e0186f91e6fc6d` |

## What they are

- **`soup42_last4.safetensors`**: the uniform weight average of the last
  four released `ink_9um` seed-42 checkpoints (steps 40k/50k/60k/75k).
  Verified **bit-identical**, tensor for tensor, to the `soup42_last4.pth`
  that produced every soup number in this repo's score files, the same file
  `./build_soup.sh` rebuilds from the released checkpoints. Parents are MIT
  (scrollprize/ink_9um); this average is MIT too.
- **`dense_native_016000.safetensors`**: the treatment run's step-16 000
  checkpoint (the best on the held-out native benchmark), trained as
  described in `training/REPRODUCE.md`, initialized from KLAVIS's
  MIT-licensed `ink9um-dense` weights.

## Post-strip re-verification (dense-native, title segment)

The stripped file was re-run end to end: safetensors → `.pth` payload
rebuilt with the shipped config (the three-line snippet in the README) →
`koine_machines.inference.infer` on the title segment (default window,
forward, hann blending) → `native-eval/native_bench.py metrics`.

```
AUC vs 78 keV classes: 0.9548  (0.95475465; lo=31 hi=128 n_ink=995114 n_noink=2115012)
misregistration control max: 0.7050
```

This matches `training/BENCHMARK.csv` (`dense_native-016000`, title:
0.9548 / 0.705) exactly. The AUC agrees to full float precision with
scoring the original run's archived prediction map, so nothing was lost in
the strip.

## Re-verify yourself

```bash
python - <<'PY'
import json, torch
from safetensors.torch import load_file
torch.save({"model": load_file("weights/dense_native_016000.safetensors"),
            "config": json.load(open("training/configs/train_dense_native.json")),
            "step": 16000}, "dense_native_016000.pth")
PY
# from villa/ink-detection (branch merge-ink-pipelines):
python -m koine_machines.inference.infer <title_dir>/zarr113 dense_native_016000.pth title.tif \
    --overlap 0.5 --blend-mode hann --direction forward --no-compile
python native-eval/native_bench.py metrics --seg-dir <title_dir> --pred title.tif
```

Segment data comes from the public S3 bucket as described in the README's
Reproduce section.
