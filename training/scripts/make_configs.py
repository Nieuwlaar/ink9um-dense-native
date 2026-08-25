#!/usr/bin/env python
"""Generate the training configs (treatment + matched control + smoke variants) for a
given data root / init checkpoint / output root.

  INK9_BASE_CONFIG=/path/to/villa/ink-detection/configs/aligned21_hybrid_3d2d.json \
  make_configs.py --data-root /data \
                  --init models/dense9um-w016excluded-step075000.pth \
                  --out-root runs --config-dir configs/local

Data-root layout (identical locally and on the rental, see training/REPRODUCE.md section 2):
  <root>/labels/aligned-scrollprizeorg-21slices/<seg>/   official v2 labels (24 aligned)
  <root>/labels/dense7-aligned-rebuilt/<seg>/            KLAVIS's 7 dense segments, rebuilt
  <root>/labels/native9-scrollprizeorg-21slices/<w>/     official native9 sparse labels (5)
  <root>/labels/native-dense-0139-78keV/<w>/             the new native-113keV dense labels (10)
  <root>/volumes/aligned-iso9/<seg>.zarr                 21-slice ~9.6um inputs (24)
  <root>/volumes/native113/<w>.zarr                      native 28-slice 113keV volumes (15)

Recipe = aligned21_hybrid_3d2d.json (KLAVIS's unmodified contract) except:
  checkpoint/weights_only (init from dense-ex016), optimizer adamw, lr 2e-5, warmup 500,
  num_iterations 20000, save/val every 4000, out_dir, datasets, and (treatment only)
  fixed_scroll_prior quotas with a 5th key '0139nat' = 19/64 (29.7%) for the native-dense set.
"""
import argparse
import copy
import json
import os

ALIGNED_BY_SCROLL = {
    "0139": ["pherc0139-w016", "pherc0139-w017", "pherc0139-w028", "pherc0139-w029", "pherc0139-w035",
             "pherc0139-w039", "pherc0139-w040", "pherc0139-w041", "pherc0139-w043"],
    "0814": ["pherc0814-46527"],
    "1667": ["pherc1667-w013", "pherc1667-w018", "pherc1667-w023", "pherc1667-w028", "pherc1667-w029",
             "pherc1667-w031"],
    "Paris4": ["phercparis4-w00", "phercparis4-w01", "phercparis4-w02", "phercparis4-w03", "phercparis4-w05",
               "phercparis4-w06", "phercparis4-w07", "phercparis4-w09"],
}
# KLAVIS's shipped "w016-excluded" dense set: these 6 read dense labels, w016 stays manual.
DENSE6 = ["pherc0139-w017", "pherc0139-w028", "pherc0139-w029", "pherc0814-46527", "pherc1667-w028",
          "pherc1667-w029"]
NATIVE9 = ["w035", "w039", "w040", "w041", "w044"]
NATIVE_DENSE = ["w030", "w032", "w033", "w034", "w040", "w041", "w043", "w046", "w049", "w050"]
HELD_OUT = ["title", "w024", "w047", "w053"]

# KLAVIS's unmodified recipe contract, from villa's ink-detection configs.
BASE_CONFIG = os.environ.get("INK9_BASE_CONFIG", "aligned21_hybrid_3d2d.json")
BASE = json.load(open(BASE_CONFIG))


def physical(seg):
    # official convention: "<scroll>:<suffix after the dash>"
    scroll = next(s for s, l in ALIGNED_BY_SCROLL.items() if seg in l)
    return f"{scroll}:{seg.split('-', 1)[1]}"


def entry(root, scroll, segs, labels_sub, family, phys_prefix, rep_prefix, volume_of):
    return {
        "segments_path": f"{root}/labels/{labels_sub}",
        "segments": list(segs),
        "surface_volume_paths": {s: volume_of(s) for s in segs},
        "volume_scale": 0,
        "source_family": family,
        "sampling_scroll": scroll,
        "sampling_physical_segment_keys": {s: phys_prefix(s) for s in segs},
        "sampling_representation_keys": {s: f"{rep_prefix}:{s}" for s in segs},
    }


def datasets(root, with_native_dense):
    iso = lambda s: f"{root}/volumes/aligned-iso9/{s}.zarr"  # noqa: E731
    nat = lambda w: f"{root}/volumes/native113/{w}.zarr"     # noqa: E731
    out = []
    for scroll, segs in ALIGNED_BY_SCROLL.items():
        manual = [s for s in segs if s not in DENSE6]
        dense = [s for s in segs if s in DENSE6]
        if manual:
            out.append(entry(root, scroll, manual, "aligned-scrollprizeorg-21slices",
                             "public_2p4_level2_zmean4", physical, "public_2p4_level2_zmean4", iso))
        if dense:
            out.append(entry(root, scroll, dense, "dense7-aligned-rebuilt",
                             "public_2p4_level2_zmean4", physical, "public_2p4_level2_zmean4", iso))
    out.append(entry(root, "0139", NATIVE9, "native9-scrollprizeorg-21slices", "native_9p362_level0",
                     lambda w: f"0139:{w}", "native_9p362_level0", nat))
    if with_native_dense:
        out.append(entry(root, "0139nat", NATIVE_DENSE, "native-dense-0139-78keV",
                         "native_9p362_level0_dense78keV", lambda w: f"0139nat:{w}",
                         "native_9p362_level0_dense78keV", nat))
    return out


def _available(root, d):
    """Drop segments whose volume or label dir is missing (local smoke on a partial mirror)."""
    keep = [s for s in d["segments"]
            if os.path.isdir(d["surface_volume_paths"][s]) and os.path.isdir(os.path.join(d["segments_path"], s))]
    d = copy.deepcopy(d)
    d["segments"] = keep
    for k in ("surface_volume_paths", "sampling_physical_segment_keys", "sampling_representation_keys"):
        d[k] = {s: d[k][s] for s in keep}
    return d


def make(root, init, out_dir, with_native_dense, *, iters=20000, save_every=4000, warmup=500,
         lr=2e-5, workers=12, batch=64, desc="", available_only=False):
    c = copy.deepcopy(BASE)
    c["description"] = desc
    c["checkpoint"] = init
    c["weights_only"] = True
    c["optimizer"] = "adamw"
    c["learning_rate"] = lr
    c["warmup_steps"] = warmup
    c["num_iterations"] = iters
    c["save_every"] = save_every
    c["val_every"] = save_every
    c["batch_size"] = batch
    c["dataloader_workers"] = workers
    c["out_dir"] = out_dir
    if with_native_dense:
        c["fixed_scroll_prior"] = {"seed": 42, "target_batch_counts": {"0139": 20, "1667": 16, "Paris4": 8,
                                                                       "0814": 1, "0139nat": 19}}
    c["datasets"] = datasets(root, with_native_dense)
    if available_only:
        ds = [_available(root, d) for d in c["datasets"]]
        c["datasets"] = [d for d in ds if d["segments"]]
        present = {d["sampling_scroll"] for d in c["datasets"]}
        q = {k: v for k, v in c["fixed_scroll_prior"]["target_batch_counts"].items() if k in present}
        # rescale the surviving quotas to the batch size (largest-remainder)
        tot = sum(q.values())
        scaled = {k: batch * v / tot for k, v in q.items()}
        q2 = {k: int(v) for k, v in scaled.items()}
        for k in sorted(scaled, key=lambda k: scaled[k] - int(scaled[k]), reverse=True)[: batch - sum(q2.values())]:
            q2[k] += 1
        c["fixed_scroll_prior"]["target_batch_counts"] = q2
        c["description"] += " [AVAILABLE-ONLY subset: partial local mirror; quotas rescaled]"
    assert sum(c["fixed_scroll_prior"]["target_batch_counts"].values()) == c["batch_size"]
    # held-out integrity: no benchmark segment anywhere
    for d in c["datasets"]:
        for s in d["segments"]:
            assert not any(h == s or s.endswith(h) for h in HELD_OUT), s
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out-root", required=True, help="run output root (checkpoints go to <out-root>/<name>)")
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke-iters", type=int, default=60)
    ap.add_argument("--smoke-available-only", action="store_true",
                    help="smoke configs drop segments missing from the local mirror")
    a = ap.parse_args()
    os.makedirs(a.config_dir, exist_ok=True)
    R, I, O = a.data_root.rstrip("/"), a.init, a.out_root.rstrip("/")
    common = dict(workers=a.workers)
    cfgs = {
        "train_dense_native.json": make(R, I, f"{O}/dense_native", True, desc=(
            "TREATMENT: KLAVIS dense recipe (aligned21_hybrid_3d2d, his ex016 label set rebuilt from public "
            "data) + OUR 10 native-113keV PHerc0139 segments with dense 78keV-teacher pseudo-labels at "
            "~30% of every batch (quota key 0139nat=19/64). Init dense-ex016, AdamW 2e-5, 20k steps. "
            "Held out forever: title w024 w047 w053."), **common),
        "train_control.json": make(R, I, f"{O}/control", False, desc=(
            "MATCHED CONTROL: identical to train_dense_native.json except the native-dense dataset entry is "
            "absent and the scroll quotas are KLAVIS's original 29/22/11/2."), **common),
        "smoke_dense_native.json": make(R, I, f"{O}/smoke_dense_native", True, iters=a.smoke_iters,
                                        save_every=a.smoke_iters // 2, warmup=10,
                                        desc="SMOKE of train_dense_native.json (short schedule)",
                                        available_only=a.smoke_available_only, **common),
        "smoke_control.json": make(R, I, f"{O}/smoke_control", False, iters=a.smoke_iters,
                                   save_every=a.smoke_iters // 2, warmup=10,
                                   desc="SMOKE of train_control.json (short schedule)",
                                   available_only=a.smoke_available_only, **common),
    }
    for name, c in cfgs.items():
        p = os.path.join(a.config_dir, name)
        json.dump(c, open(p, "w"), indent=1)
        n_reps = sum(len(d["segments"]) for d in c["datasets"])
        print(f"wrote {p}: {n_reps} representations, quotas {c['fixed_scroll_prior']['target_batch_counts']}")


if __name__ == "__main__":
    main()
