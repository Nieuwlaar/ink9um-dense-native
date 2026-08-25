#!/usr/bin/env python3
"""Mirror pyramid level 2 of a public S3 surface-volume zarr as a local bare zarr array.

Usage: fetch_s3_level2.py <zarr_base_url> <dest_dir>
Writes <dest_dir>/.zarray (verbatim remote 2/.zarray) and chunks <dest_dir>/<z>/<y>/<x>.
Missing remote chunks (404) are fill-value chunks and are skipped. Idempotent:
existing full-size chunk files are not re-downloaded (compressor is null, so every
stored chunk has the exact uncompressed size).
"""
import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


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
    base, dest = sys.argv[1].rstrip("/"), Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    zarray = http_get(base + "/2/.zarray")
    meta = json.loads(zarray)
    assert meta["compressor"] is None, "expected uncompressed chunks"
    (dest / ".zarray").write_bytes(zarray)
    shape, chunks = meta["shape"], meta["chunks"]
    chunk_bytes = chunks[0] * chunks[1] * chunks[2]
    grid = [math.ceil(s / c) for s, c in zip(shape, chunks)]
    keys = [(z, y, x) for z in range(grid[0]) for y in range(grid[1]) for x in range(grid[2])]
    print(f"{dest.name}: shape={shape} grid={grid} -> {len(keys)} chunks", flush=True)

    def fetch(key):
        z, y, x = key
        out = dest / str(z) / str(y) / str(x)
        if out.exists() and out.stat().st_size == chunk_bytes:
            return 0
        body = http_get(f"{base}/2/{z}/{y}/{x}")
        if body is None:
            return 0  # fill-value chunk, not stored
        assert len(body) == chunk_bytes, f"bad chunk size {len(body)} for {key}"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_bytes(body)
        tmp.replace(out)
        return len(body)

    done = 0
    total = 0
    with ThreadPoolExecutor(max_workers=48) as pool:
        for n in pool.map(fetch, keys):
            done += 1
            total += n
            if done % 500 == 0:
                print(f"{dest.name}: {done}/{len(keys)} chunks, {total/1e9:.2f} GB new", flush=True)
    print(f"{dest.name}: DONE {done} chunks, {total/1e9:.2f} GB new", flush=True)


if __name__ == "__main__":
    main()
