#!/usr/bin/env python3
"""Fetch only the level-2 chunks needed for a segment's validation crop.

Usage: targeted_fetch.py <segment> <zarr_base_url>
Chunks intersecting [z all, y0:y1, x0:x1] of the crop go to the same local
mirror the full downloader uses (data/src/<seg>_level2.zarr). 404 = chunk
absent on S3 (fill value) and is fine. Unique tmp suffix avoids clashing with
the concurrently running full mirror.
"""
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crop_bounds import crop_bounds, ROOT


def http_get(url, tries=6):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == tries - 1:
                raise
        except Exception:
            if attempt == tries - 1:
                raise
        time.sleep(2 * (attempt + 1))


def main():
    seg, base = sys.argv[1], sys.argv[2].rstrip("/")
    dest = ROOT / "data" / "src" / f"{seg}_level2.zarr"
    meta = json.loads((dest / ".zarray").read_bytes()) if (dest / ".zarray").exists() else None
    if meta is None:
        raw = http_get(base + "/2/.zarray")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / ".zarray").write_bytes(raw)
        meta = json.loads(raw)
    cz, cy, cx = meta["chunks"]
    chunk_bytes = cz * cy * cx
    y0, x0, y1, x1, H, W = crop_bounds(seg)
    keys = [(0, yy, xx)
            for yy in range(y0 // cy, math.ceil(y1 / cy))
            for xx in range(x0 // cx, math.ceil(x1 / cx))]
    print(f"{seg}: crop [{y0}:{y1},{x0}:{x1}] -> {len(keys)} chunks", flush=True)

    absent = 0

    def fetch(key):
        nonlocal absent
        z, yy, xx = key
        out = dest / str(z) / str(yy) / str(xx)
        if out.exists() and out.stat().st_size == chunk_bytes:
            return 0
        body = http_get(f"{base}/2/{z}/{yy}/{xx}")
        if body is None:
            absent += 1
            return 0
        assert len(body) == chunk_bytes
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + f".tmp{os.getpid()}")
        tmp.write_bytes(body)
        tmp.replace(out)
        return len(body)

    done = 0
    new = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        for n in pool.map(fetch, keys):
            done += 1
            new += n
            if done % 100 == 0:
                print(f"{seg}: {done}/{len(keys)} chunks, {new/1e9:.2f} GB new", flush=True)
    print(f"{seg}: TARGETED-DONE {done} chunks ({absent} absent on S3), {new/1e9:.2f} GB new", flush=True)


if __name__ == "__main__":
    main()
