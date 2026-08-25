#!/usr/bin/env python3
"""Mirror one segment's label directory from the scrollprize HF bucket (anonymous).

Usage: fetch_bucket_labels.py <segment> <dest_root>
Downloads ink_9um/labels/aligned-scrollprizeorg-21slices/<segment>/** to
<dest_root>/<segment>/**. Deduplicates by xetHash (identical chunk files are
fetched once and copied), honors 429 Retry-After, resumes (skips files already
present with the right size).
"""
import json
import re
import shutil
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


def http_get(url, tries=10):
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


def main():
    segment, dest_root = sys.argv[1], Path(sys.argv[2])
    files = list_segment(segment)

    def dest_of(path):
        return dest_root / path[len(PREFIX):]

    # group by content hash
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
    print(f"{segment}: {len(files)} files, {len(by_hash)} unique blobs, "
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
        if done[0] % 200 == 0:
            print(f"{segment}: {done[0]}/{len(todo)} blobs", flush=True)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(handle, todo))
    print(f"{segment}: LABELS-DONE ({len(todo)} blobs handled)", flush=True)


if __name__ == "__main__":
    main()
