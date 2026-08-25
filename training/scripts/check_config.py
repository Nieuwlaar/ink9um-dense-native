#!/usr/bin/env python
"""Static + on-disk validation of a training config before a (paid) run.

  check_config.py <config.json> [<config.json> ...] [--no-disk]

Checks: sampler contract (quotas sum to batch_size, every dataset has sampling_scroll +
both key maps for every segment, a physical key never crosses scroll keys), held-out
integrity (title/w024/w047/w053 never present), init checkpoint loads with 'model', and
(unless --no-disk) every label dir / volume exists, label zarr '0' shape == volume '0'
shape, the annotated plane (shape[0]//2) of every supervision mask is non-empty, and
other planes are empty (spot-check z=0).
"""
import json
import os
import sys

HELD_OUT = ("title", "w024", "w047", "w053")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    disk = "--no-disk" not in sys.argv
    import zarr
    import numpy as np
    ok_all = True
    for cfg_path in args:
        c = json.load(open(cfg_path))
        errs = []
        q = c["fixed_scroll_prior"]["target_batch_counts"]
        if sum(q.values()) != c["batch_size"]:
            errs.append(f"quotas {q} sum != batch_size {c['batch_size']}")
        phys_scroll = {}
        scrolls = set()
        n_reps = 0
        for i, d in enumerate(c["datasets"]):
            sc = d.get("sampling_scroll")
            scrolls.add(sc)
            for s in d["segments"]:
                n_reps += 1
                if s in HELD_OUT or any(s.endswith("-" + h) or s == h for h in HELD_OUT):
                    errs.append(f"HELD-OUT segment in training: {s}")
                for k in ("surface_volume_paths", "sampling_physical_segment_keys", "sampling_representation_keys"):
                    if s not in d.get(k, {}):
                        errs.append(f"datasets[{i}] {s} missing {k}")
                p = d["sampling_physical_segment_keys"].get(s)
                if p in phys_scroll and phys_scroll[p] != sc:
                    errs.append(f"physical key {p} crosses scrolls {phys_scroll[p]} / {sc}")
                phys_scroll[p] = sc
                if disk:
                    vol = d["surface_volume_paths"][s]
                    lab = os.path.join(d["segments_path"], s)
                    if not os.path.isdir(vol):
                        errs.append(f"missing volume {vol}"); continue
                    if not os.path.isdir(lab):
                        errs.append(f"missing label dir {lab}"); continue
                    try:
                        v = zarr.open(vol, mode="r")["0"]
                        ink = zarr.open(os.path.join(lab, f"{s}_inklabels.zarr"), mode="r")["0"]
                        sup = zarr.open(os.path.join(lab, f"{s}_supervision_mask.zarr"), mode="r")["0"]
                    except Exception as e:  # noqa: BLE001
                        errs.append(f"{s}: open failed: {e}"); continue
                    if tuple(ink.shape) != tuple(v.shape) or tuple(sup.shape) != tuple(v.shape):
                        errs.append(f"{s}: label shape {ink.shape}/{sup.shape} != volume {v.shape}")
                        continue
                    z = v.shape[0] // 2
                    sp = np.asarray(sup[z, ::4, ::4])
                    if not np.any(sp):
                        errs.append(f"{s}: supervision plane z={z} empty")
                    if np.any(np.asarray(sup[0, ::8, ::8])):
                        errs.append(f"{s}: supervision plane z=0 NOT empty (layout?)")
                    vp = np.asarray(v[z, ::8, ::8])
                    print(f"  {s:<18} vol {tuple(v.shape)} sup@z{z} {float((sp>0).mean()):.3f} "
                          f"ink@z{z} {float((np.asarray(ink[z, ::4, ::4])>0).mean()):.3f} vol@z{z}>0 {float((vp>0).mean()):.3f}")
        if set(q) != scrolls:
            errs.append(f"quota keys {sorted(q)} != dataset scrolls {sorted(scrolls)}")
        ck = c.get("checkpoint")
        if ck and disk:
            if not os.path.isfile(ck):
                errs.append(f"init checkpoint missing {ck}")
            else:
                import torch
                sd = torch.load(ck, map_location="cpu", weights_only=False)
                if "model" not in sd:
                    errs.append("init checkpoint has no 'model'")
                elif not c.get("weights_only"):
                    errs.append("checkpoint given but weights_only is not true (full resume would KeyError)")
        status = "OK" if not errs else "FAIL"
        print(f"{cfg_path}: {status}  reps={n_reps} quotas={q} iters={c['num_iterations']} lr={c['learning_rate']} "
              f"opt={c['optimizer']} out={c['out_dir']}")
        for e in errs:
            print("   !!", e)
        ok_all &= not errs
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
