# Dense native-scan pseudo-label training: how to reproduce it

This directory holds the training experiment behind the entry's headline
number: extending the dense pseudo-label recipe published by
**DomRusso2 / `ink9um-dense` (KLAVIS)** with dense labels on **native
113 keV PHerc0139 surface volumes**, measured against a matched control on a
frozen held-out benchmark.

Everything below runs on public data: the MIT-licensed `scrollprize/ink_9um`
labels, KLAVIS's public `domenicor046/ink9um-dense` checkpoints, and the
open-data S3 bucket. Two runs of 20 000 steps take about five hours on two
cloud 4090s, roughly $4 of compute.

The raw training checkpoints (415 MB each with optimizer state) are not in
this repo; the stripped step-16 000 state_dict ships in `weights/`. Everything
needed to rebuild from scratch is here.

---

## 0. Result you should get

`BENCHMARK.csv` is the harness log of all 20 evaluations (5 checkpoints x 2
runs x 2 segments), with the independent-reference fields for w024 merged in
from the per-run metrics JSONs.

| run | title AUC (5 ckpts) | w024 AUC (5 ckpts) | w024 vs 59 keV @16k |
|---|---|---|---|
| treatment (`train_dense_native.json`) | 0.9485 – 0.9548 | 0.9678 – 0.9694 | 0.9149 |
| control (`train_control.json`) | 0.9023 – 0.9149 | 0.9523 – 0.9545 | 0.9085 |

Best treatment checkpoint is step 16 000: **0.9548 / 0.9685**. The two ranges
never overlap on either segment. The initialization (KLAVIS's shipped dense
checkpoint) scores 0.9147 / 0.9538 on the same benchmark, which is where the
control stays. So the gain is the native-dense data, not 20 000 extra steps.

The `smoke-*` row is a 60-step local smoke test on a partial data mirror; it
is not part of the comparison, only proof that the harness runs end to end.

## 1. Environment

> Branch tip used: `3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e` (2026-08-14). The pipeline has since been merged to villa main as `vesuvius/src/vesuvius/ink_detection` (PR #1456); these instructions use the pre-merge module name `koine_machines`, which resolves only on this branch.

Training and inference use villa's `ink-detection` tree (branch
`merge-ink-pipelines`, so `koine_machines` resolves), on torch 2.4.1+cu121.
The cloud image was `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel` plus:

```
numpy==2.2.6 zarr==2.18.7 numcodecs==0.15.1 albumentations==2.0.8
opencv-contrib-python-headless==4.11.0.86 timm==1.0.25 einops==0.8.2
accelerate==1.14.0 diffusers==0.39.0 pytorch-optimizer==3.10.1
scipy tifffile imagecodecs numba connected-components-3d boto3 huggingface_hub
```

villa's top-level `vesuvius/__init__.py` is heavy; the runs used a one-line
shim so the submodules import on their own:

```bash
mkdir -p shim/vesuvius
echo "__path__ = ['<villa>/vesuvius/src/vesuvius']" > shim/vesuvius/__init__.py
export PYTHONPATH=<villa>/ink-detection:$PWD/shim
```

## 2. Inputs

Fixed data root layout. The configs and every script assume it:

```
<root>/labels/aligned-scrollprizeorg-21slices/<seg>/   official labels (24 aligned segments)
<root>/labels/dense7-aligned-rebuilt/<seg>/            KLAVIS's 7 dense segments, rebuilt (step 3)
<root>/labels/native9-scrollprizeorg-21slices/<w>/     official native sparse labels (5)
<root>/labels/native-dense-0139-78keV/<w>/             the new native-113 keV dense labels (step 4)
<root>/volumes/aligned-iso9/<seg>.zarr                 21-slice ~9.6 um pooled inputs (24)
<root>/volumes/native113/<w>.zarr                      native 28-slice 113 keV volumes (15)
```

```bash
# official labels (HF bucket, level-0 arrays; the 2026-08-18 corrected upload)
python scripts/fetch_aligned_labels.py

# aligned 9.6 um model inputs: level-2 mirror from S3, then villa's pooling script
python scripts/fetch_s3_level2.py <zarr_url> <root>/volumes/level2/<seg>.zarr
./scripts/build_iso9.sh <seg>       # wraps prepare_9um_isotropic_input.py

# native 113 keV surface volumes: the 0/ array of
#   s3://vesuvius-challenge-open-data/PHerc0139/segments/<seg>/surface-volumes/
#     *9.362um*113keV*.zarr        (~1.3 GB each, 15 segments)
```

`aligned_sources.tsv` and `teacher_sources.tsv` list the exact S3 object for
every input and every teacher map, so none of this has to be guessed.

## 3. Rebuild KLAVIS's seven dense label sets

The labels are re-derived from his published recipe and the org's **public**
canonical-2.4 µm ink maps
(`*new_canon_autoresearch_recipe*.tif` under each segment's `ink-detection/`),
rather than taken from his `domenicor046/ink9um-dense-labels` tar archive:
pool the L0 map 4x with `INTER_AREA` onto the level-2 canvas, re-calibrate the
ink threshold per segment by his balanced-accuracy rule on the manual
supervision region, let manual labels win wherever they exist, set supervision
= render-valid minus validation, and copy the validation masks verbatim.

```bash
python scripts/build_dense7_labels.py            # --verify-only to re-run the gates
```

Canvas coverage reproduces his published table to the decimal on six of seven
segments (83.9 / 85.0 / 88.8 / 88.2 / 56.7 / 81.8 / 78.6 %; the last,
pherc1667-w029, rebuilds 0.1 below his published 78.7), which is also the
evidence that the
org's published maps are his teacher family. Per-segment output:
`dense7_report.json`.

His threshold values are not transferable 1:1 because they refer to his own
`clip(0,200)/200` teacher scaling; the recalibrated values are in the report.


**Label-version note.** The org re-ran the native9 label transfer on
2026-08-19 (the `rerun_provenance` block in the current `.zattrs`). The
per-segment threshold calibration was re-run against that label version:
t* is identical for all five segments (39/47/48/46/61, median 47 = 0.184),
balanced accuracy unchanged to three decimals. So the calibration shipped
here is valid for both label versions.

## 4. Build the native-113 keV dense labels (the new ingredient)

The teacher is the org's public 78 keV companion ink map for the same segment,
resized onto the 113 keV grid with `INTER_AREA`; valid = `max_z(volume) > 0`.
The threshold is calibrated with the same balanced-accuracy rule, against the
five manual native label sets:

```bash
python scripts/calibrate_theta.py                # -> theta_calibration.json
python scripts/build_native_dense_labels.py      # -> labels/native-dense-0139-78keV/
```

Per-segment `t*` comes out at 39–61/255; segments with their own manual labels
(w040, w041) use their own value, the other eight use the **median, 47
(= 0.184)**. Manual labels win wherever manual supervision exists. Output
layout is byte-compatible with the official native label zarrs (28 x H x W,
u1, blosc zstd-3 bitshuffle, content on z = 14).

Ten training segments: **w030, w032, w033, w034, w040, w041, w043, w046,
w049, w050**. Coverage lands at 84–88 % of canvas, the same regime as the
aligned dense set, giving 415 k training patches against 5.8 k from all the
sparse native labels (72x the sparse density). Details:
`native_dense_report.json`.

**The four benchmark segments (title, w024, w047, w053) are excluded here
and asserted absent from every dataset entry in `make_configs.py`
(`HELD_OUT`), which fails loudly rather than silently training on them.**

## 5. Generate and check the configs

```bash
python scripts/make_configs.py --data-root <root> \
    --init <KLAVIS dense-ex016 checkpoint>.pth \
    --out-root <runs> --config-dir configs/
python scripts/check_config.py configs/train_dense_native.json
python scripts/check_config.py configs/train_control.json
```

The shipped `configs/*.json` are the exact files the runs consumed (paths
under `/workspace`, the training box's data root). Both are KLAVIS's
unmodified `aligned21_hybrid_3d2d` contract except for: init from his best
public checkpoint (`weights_only`, fresh optimizer, step 0), AdamW 2e-5, wd
3e-5, warmup 500, cosine to 0, 20 000 steps, batch 64, fp16, checkpoint every
4 000.

The two configs differ in exactly four keys. Verify it yourself:

```bash
diff <(python -m json.tool configs/train_dense_native.json) \
     <(python -m json.tool configs/train_control.json)
# description, out_dir, fixed_scroll_prior.target_batch_counts, and the one extra datasets[] entry
```

Treatment batch quotas (of 64): 0139 20 / 1667 16 / Paris4 8 / 0814 1 /
**0139nat 19** (29.7 %). Control: KLAVIS's original 29 / 22 / 11 / 2. The
native-dense representations sit under their own sampler scroll key so the
control's 29 representations keep their relative weights. The treatment adds
data; it does not re-weight the corpus. A sampler audit is written every 200
steps; the smoke run's audit confirmed 0139nat at 34.4 % of samples with every
physical segment inside a scroll sampled uniformly.

## 6. Train

```bash
python -m koine_machines.training.train configs/train_dense_native.json   # GPU 0
python -m koine_machines.training.train configs/train_control.json        # GPU 1
```

Two 4090s run the arms in parallel; on one A100-80 GB run them back to back.
Throughput is ~3.5 it/s at batch 64 (~8 GB VRAM), so 20 000 steps is about
1.6 h per arm. Staging the ~250 GB of S3 reads dominates wall clock. Budget
4–5 h and about $4 in total.

Known-cosmetic: the persistent dataloader workers abort at interpreter exit
*after* all checkpoints are written. Treat the presence of
`ckpt_020000.pth` as the success condition, not the exit code.

## 7. Evaluate on the held-out benchmark

Use the repo's own harness, one level up, the same one that produced every
other number in this entry:

```bash
# from villa/ink-detection, with the shim on PYTHONPATH
python -m koine_machines.inference.infer <seg>/zarr113 <ckpt>.pth out/<name>.tif \
    --overlap 0.5 --blend-mode hann --direction forward --no-compile
# tif -> probability: np.clip((a - 64.) / 128., 0, 1)

python ../native-eval/native_bench.py metrics --seg-dir <seg_dir> --pred out/<name>_prob.npy
```

Run it on `ckpt_004000` … `ckpt_020000` of both arms, on both benchmark
segments, and you should land on `BENCHMARK.csv`. Segment IDs and the S3
paths for title / w024 / w047 / w053 are in the top-level `README.md`.

## 8. Caveats to keep attached to these numbers

- **The teacher is in the benchmark's family.** The native-dense pseudo-labels
  come from the org's 78 keV maps, and the benchmark's main reference is the
  same 78 keV family. So AUC-vs-78 gains are an **upper bound**. This
  is why the benchmark also carries an independent 59 keV / 1.1 µm reference
  from a different model on a different scan (treatment 0.9149 vs control
  0.9085 at 16k; same direction, smaller margin) and a cloning check:
  `student_over_ceiling` = how much better the model agrees with the 78 keV
  reference than the independent 59 keV reference does. It sits at 1.018 at
  16k (1.018–1.020 across the five treatment checkpoints), under the 1.02
  flag. The model is not collapsing onto its teacher, but it is close enough
  to the line that the vs-78 number should never be quoted alone.
- **Misregistration control**: 0.70–0.77 for the treatment, above the 0.60
  gate but higher than the official checkpoints' 0.60–0.69, consistent with
  broader supervision carrying more low-frequency region prior.
- **One seed, one initialization.** Both arms start from KLAVIS's dense
  checkpoint with seed 42. A soup42-initialized replicate in a different basin
  is the obvious follow-up and has not been run.
- Better AUC is not readable text. These maps show letter-shaped strokes on
  native 113 keV data; they are not transcribable.

## 9. Paths in the scripts

The scripts are the ones that actually ran, absolute paths and all. Before
running them, edit the constants at the top of `scripts/native_common.py`
(segment data root, the official native label and volume directories) to point
at your copies. `scripts/common0139.py` is a small shim: the two calibration
scripts import a tie-corrected ROC-AUC helper from my benchmark tree, and it
is the identical function that ships in `native-eval/native_bench.py`, so the
shim just re-exports it.
