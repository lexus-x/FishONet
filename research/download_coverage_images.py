"""Download images from coverage URL JSON (GBIF/Wikimedia/Openverse/etc.)."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
UA = {'User-Agent': 'onet-cv4e-research/1.0 (academic; contact cv4e.workshop@gmail.com)'}


def load_merged_urls(base_path: str, extra_path: str | None) -> dict:
    urls = json.load(open(base_path))
    if extra_path and os.path.exists(extra_path):
        for c, us in json.load(open(extra_path)).items():
            prev = list(urls.get(c) or [])
            seen = set(prev)
            for u in us or []:
                if u and u not in seen:
                    prev.append(u)
                    seen.add(u)
            urls[c] = prev
    return urls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urls', default=os.path.join(OUT, 'coverage_urls.json'))
    ap.add_argument('--extra_urls', default='', help='Optional second JSON merged at read time')
    ap.add_argument('--dl_dir', default=os.path.join(OUT, 'coverage_images'))
    ap.add_argument('--files_out', default='')
    a = ap.parse_args()

    extra = a.extra_urls or None
    files_out = a.files_out or os.path.join(
        OUT,
        'coverage_image_files_extra.json'
        if a.dl_dir.rstrip('/').endswith('_extra')
        else 'coverage_image_files.json',
    )
    os.makedirs(a.dl_dir, exist_ok=True)

    urls = load_merged_urls(a.urls, extra)
    files = json.load(open(files_out)) if os.path.exists(files_out) else {}

    jobs = []
    for c, us in urls.items():
        if not us:
            continue
        safe = c.replace(' ', '_').replace('/', '_')
        for i, u in enumerate(us):
            dest = os.path.join(a.dl_dir, f'{safe}__{i}.jpg')
            jobs.append((c, u, dest))
    print(f'{len(jobs)} jobs, {sum(1 for v in urls.values() if v)} classes -> {a.dl_dir}', flush=True)

    def dl(job):
        c, u, dest = job
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            return c, dest
        try:
            r = requests.get(u, timeout=45, headers=UA, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 1500:
                ct = (r.headers.get('content-type') or '').lower()
                if 'html' in ct and len(r.content) < 50000:
                    return c, None
                open(dest, 'wb').write(r.content)
                return c, dest
        except Exception:
            pass
        return c, None

    t0 = time.time()
    n_ok = 0
    workers = int(os.environ.get('DL_WORKERS', '32'))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(dl, j) for j in jobs]
        for k, f in enumerate(as_completed(futs)):
            c, dest = f.result()
            if dest:
                files.setdefault(c, [])
                if dest not in files[c]:
                    files[c].append(dest)
                n_ok += 1
            if (k + 1) % 500 == 0 or k + 1 == len(jobs):
                json.dump(files, open(files_out + '.tmp', 'w'))
                os.replace(files_out + '.tmp', files_out)
                cov = sum(1 for v in files.values() if v)
                print(f'  {k+1}/{len(jobs)} ok={n_ok} classes={cov} [{time.time()-t0:.0f}s]', flush=True)

    json.dump(files, open(files_out, 'w'), indent=1)
    cov = sum(1 for v in files.values() if v)
    print(f'DONE: {n_ok} files, {cov} classes', flush=True)


if __name__ == '__main__':
    main()
