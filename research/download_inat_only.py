"""Download iNat photos from already-cached URLs (S3/CDN, no API rate limit).

Reads outputs/inat_image_urls.json, downloads to outputs/inat_images/, writes
outputs/inat_image_files.json. Does NOT touch the rate-limited iNat API.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT  # noqa: E402

UA = {'User-Agent': 'onet-cv4e-research/1.0'}
urls = json.load(open(os.path.join(OUT, 'inat_image_urls.json')))
dl_dir = os.path.join(OUT, 'inat_images')
os.makedirs(dl_dir, exist_ok=True)
files_out = os.path.join(OUT, 'inat_image_files.json')
files = json.load(open(files_out)) if os.path.exists(files_out) else {}

jobs = []
for c, us in urls.items():
    for i, u in enumerate(us or []):
        safe = c.replace(' ', '_').replace('/', '_')
        dest = os.path.join(dl_dir, f'{safe}__{i}.jpg')
        jobs.append((c, u, dest))
print(f'{len(jobs)} image jobs over {sum(1 for c in urls if urls[c])} classes', flush=True)


def dl(job):
    c, u, dest = job
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return c, dest
    try:
        r = requests.get(u, timeout=40, headers=UA)
        if r.status_code == 200 and len(r.content) > 1500:
            open(dest, 'wb').write(r.content)
            return c, dest
    except Exception:
        return c, None
    return c, None


t0 = time.time()
n_ok = 0
with ThreadPoolExecutor(max_workers=32) as ex:
    futs = [ex.submit(dl, j) for j in jobs]
    for k, f in enumerate(as_completed(futs)):
        c, dest = f.result()
        if dest:
            files.setdefault(c, [])
            if dest not in files[c]:
                files[c].append(dest)
            n_ok += 1
        if (k + 1) % 2000 == 0:
            json.dump(files, open(files_out + '.tmp', 'w'))
            os.replace(files_out + '.tmp', files_out)
            print(f'  {k+1}/{len(jobs)} ok={n_ok} [{time.time()-t0:.0f}s]', flush=True)
json.dump(files, open(files_out, 'w'), indent=1)
cov = sum(1 for c in files if files[c])
print(f'done: {n_ok} files, {cov} classes covered', flush=True)
