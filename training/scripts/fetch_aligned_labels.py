#!/usr/bin/env python3
"""Mirror the level-0 label arrays of ink_9um aligned segments from the public
scrollprize HF bucket (anonymous, Retry-After aware, resumable).

Usage: fetch_aligned_labels.py <dest_root> <segment> [<segment> ...]

Writes <dest_root>/<segment>/<segment>_{inklabels,supervision_mask,validation_mask}.zarr/
with .zgroup, .zattrs and the full level "0" array (the koine loader opens
resolution "0" only; pyramid levels 1-5 are skipped to cut the file count ~4x).
Derived from official-eval/scripts/fetch_bucket_labels.py (xetHash dedup).
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET_API = "https://huggingface.co/api/buckets/scrollprize/datasets/tree/"
BUCKET_RESOLVE = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
PREFIX = "ink_9um/labels/aligned-scrollprizeorg-21slices/"
KEEP_RE = re.compile(r"\.zarr/(\.zgroup|\.zattrs|0/.*)$")
WORKERS = 5


def http_get(url, tries=12):
    delay = 2.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read(), r.headers
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                time.sleep(min(wait, 120))
                delay = min(delay * 2, 120)
                continue
            if attempt == tries - 1:
                raise
            time.sleep(delay)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(delay)
    raise RuntimeError(f"exhausted retries for {url}")


def list_segment(segment):
    base = BUCKET_API + urllib.parse.quote(PREFIX + segment)
    cursor = None
    files = []
    while True:
        url = base + (f"?cursor={urllib.parse.quote(cursor)}" if cursor else "")
        body, headers = http_get(url)
        data = json.loads(body)
        if isinstance(data, dict):
            data = data.get("items", data)
        files += [
            (d["path"], int(d.get("size", -1)), d.get("xetHash") or d["path"])
            for d in data if d.get("type") == "file"
        ]
        link = headers.get("Link", "") or ""
        m = re.search(r"[?&]cursor=([^&>]+)", link)
        if m and 'rel="next"' in link:
            cursor = urllib.parse.unquote(m.group(1))
        else:
            break
    return files


def fetch_segment(segment, dest_root):
    files = [f for f in list_segment(segment) if KEEP_RE.search(f[0])]

    def dest_of(path):
        return dest_root / path[len(PREFIX):]

    by_hash = {}
    for path, size, h in files:
        by_hash.setdefault(h, []).append((path, size))
    todo = []
    for h, members in by_hash.items():
        missing = [(p, s) for p, s in members
                   if not (dest_of(p).exists() and (s < 0 or dest_of(p).stat().st_size == s))]
        if not missing:
            continue
        donor = next((p for p, s in members if dest_of(p).exists()
                      and (s < 0 or dest_of(p).stat().st_size == s)), None)
        todo.append((h, donor, missing))
    print(f"{segment}: {len(files)} level-0 files, {len(by_hash)} unique blobs, "
          f"{len(todo)} blobs to fetch/replicate", flush=True)
    done = [0]

    def handle(item):
        h, donor, missing = item
        if donor is not None:
            data = dest_of(donor).read_bytes()
        else:
            data, _ = http_get(BUCKET_RESOLVE + urllib.parse.quote(missing[0][0]))
        for p, s in missing:
            d = dest_of(p)
            d.parent.mkdir(parents=True, exist_ok=True)
            tmp = d.with_name(d.name + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(d)
        done[0] += 1
        if done[0] % 500 == 0:
            print(f"{segment}: {done[0]}/{len(todo)} blobs", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(handle, todo))
    # completeness check
    missing = [p for p, s, _ in files
               if not (dest_of(p).exists() and (s < 0 or dest_of(p).stat().st_size == s))]
    status = "LABELS-DONE" if not missing else f"INCOMPLETE ({len(missing)} missing)"
    print(f"{segment}: {status} ({len(todo)} blobs handled, {len(files)} files)", flush=True)
    (dest_root / segment / "FETCH_OK").write_text(
        json.dumps({"files": len(files), "missing": len(missing),
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n"
    ) if not missing else None
    return not missing


def main():
    dest_root = Path(sys.argv[1])
    ok = True
    for seg in sys.argv[2:]:
        ok &= fetch_segment(seg, dest_root)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
