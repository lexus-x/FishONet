"""Parallel download of GBIF media listed in external_image_urls.json."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

OUT = '/home/ubuntu/onet/outputs'
urls = json.load(open(f'{OUT}/external_image_urls.json'))
dl = os.path.join(OUT, 'external_images')
os.makedirs(dl, exist_ok=True)
name_map = {c.replace(' ', '_').replace('/', '_'): c for c in urls}
jobs = []
for c, ulist in urls.items():
    safe = c.replace(' ', '_').replace('/', '_')
    for i, u in enumerate(ulist[:2]):
        ext = os.path.splitext(urlparse(u).path)[1] or '.jpg'
        if len(ext) > 5:
            ext = '.jpg'
        dest = os.path.join(dl, f'{safe}__{i}{ext}')
        if not os.path.exists(dest) or os.path.getsize(dest) < 1000:
            jobs.append((c, u, dest))
print('to download', len(jobs), 'existing', len(os.listdir(dl)), flush=True)


def one(job):
    c, u, dest = job
    try:
        r = requests.get(u, timeout=15, headers={'User-Agent': 'onet-research/1.0'})
        if r.status_code == 200 and len(r.content) > 1000:
            open(dest, 'wb').write(r.content)
            return c, dest, True
    except Exception:
        pass
    return c, dest, False


meta = {}
for fn in os.listdir(dl):
    base = fn.rsplit('__', 1)[0]
    c = name_map.get(base)
    path = os.path.join(dl, fn)
    if c and os.path.getsize(path) >= 1000:
        meta.setdefault(c, []).append(path)

ok = fail = 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=32) as ex:
    futs = [ex.submit(one, j) for j in jobs]
    for i, f in enumerate(as_completed(futs), 1):
        c, dest, success = f.result()
        if success:
            ok += 1
            meta.setdefault(c, []).append(dest)
        else:
            fail += 1
        if i % 200 == 0 or i == len(jobs):
            print(f'{i}/{len(jobs)} ok={ok} fail={fail} [{time.time()-t0:.0f}s]', flush=True)

json.dump(meta, open(f'{OUT}/external_image_files.json', 'w'), indent=1)
print('meta classes', len(meta), 'files', sum(len(v) for v in meta.values()), flush=True)
