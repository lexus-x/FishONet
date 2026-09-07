"""Fetch iNaturalist research-grade observation photos as external image prototypes.

External data is allowed with disclosure (COMPETITION_RULES §3.2). iNaturalist is a
general biodiversity platform (NOT a fish-specific model/dataset). We fetch by
scientific name, quality_grade=research, ordered by community votes, and use the
photos only as *visual class prototypes* (never matched to eval images, never used to
identify eval-split membership).

Stage 1: fetch photo URLs per class  ->  outputs/inat_image_urls.json
Stage 2: download medium-res photos  ->  outputs/inat_images/  + outputs/inat_image_files.json

  python research/fetch_inat_images.py --which cand            # fetch URLs for full cand
  python research/fetch_inat_images.py --which pseudo --download --max_per_class 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, OUT  # noqa: E402

API = 'https://api.inaturalist.org/v1/observations'
UA = {'User-Agent': 'onet-cv4e-research/1.0 (biodiversity zero-shot prototypes)'}


class RateLimiter:
    """Global min-interval limiter (thread-safe). rate = requests/min."""

    def __init__(self, rate_per_min: float):
        self.interval = 60.0 / max(1e-6, rate_per_min)
        self.lock = threading.Lock()
        self.next_t = time.time()

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        sleep = t - now
        if sleep > 0:
            time.sleep(sleep)


LIMITER: RateLimiter | None = None


def photo_medium(url: str) -> str:
    # iNat returns square (75px) thumbnails; swap to medium (500px).
    for sz in ('square', 'thumb', 'small'):
        if f'/{sz}.' in url:
            return url.replace(f'/{sz}.', '/medium.')
    return url


BLOCK = threading.Event()  # set while iNat is throttling us


def fetch_urls(name: str, sess: requests.Session, max_per_class: int):
    """Returns list[str] on success (possibly empty=genuine), or None if blocked (429)."""
    for attempt in range(6):
        # honor a global cooldown if another thread hit a block
        while BLOCK.is_set():
            time.sleep(2.0)
        if LIMITER:
            LIMITER.wait()
        try:
            r = sess.get(API, params={
                'taxon_name': name,
                'quality_grade': 'research',
                'photos': 'true',
                'per_page': min(60, max_per_class * 3),
                'order_by': 'votes',
                'order': 'desc',
            }, headers=UA, timeout=30)
        except Exception:
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 429:
            # cooldown: let the rolling window clear before anyone retries
            if not BLOCK.is_set():
                BLOCK.set()
                time.sleep(180.0)
                BLOCK.clear()
            else:
                time.sleep(5.0)
            continue
        if r.status_code != 200:
            return []
        try:
            return _parse(r.json(), max_per_class)
        except Exception:
            return []
    return None  # persistent block -> caller must NOT cache as empty


def _parse(d, max_per_class) -> list[str]:
    urls: list[str] = []
    seen = set()
    for o in d.get('results', []):
        # one photo per observation preferred (diversity), fall back to more
        for p in o.get('photos', []):
            u = p.get('url')
            if not u:
                continue
            u = photo_medium(u)
            if u in seen:
                continue
            seen.add(u)
            urls.append(u)
            break
        if len(urls) >= max_per_class:
            break
    # if too few observations, allow multiple photos per obs
    if len(urls) < max_per_class:
        for o in d.get('results', []):
            for p in o.get('photos', []):
                u = photo_medium(p.get('url') or '')
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
                    if len(urls) >= max_per_class:
                        break
            if len(urls) >= max_per_class:
                break
    return urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', choices=['pseudo', 'cand', 'distract'], default='cand')
    ap.add_argument('--n_distract', type=int, default=1200)
    ap.add_argument('--urls', default=os.path.join(OUT, 'inat_image_urls.json'))
    ap.add_argument('--max_per_class', type=int, default=20)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--dl_dir', default=os.path.join(OUT, 'inat_images'))
    ap.add_argument('--files_out', default=os.path.join(OUT, 'inat_image_files.json'))
    ap.add_argument('--dl_workers', type=int, default=24)
    ap.add_argument('--rate', type=float, default=85.0, help='max API requests/min (iNat cap ~100)')
    a = ap.parse_args()

    global LIMITER
    LIMITER = RateLimiter(a.rate)

    D = FishData()
    cand_classes = [D.classes[int(i)] for i in D.cand]
    if a.which == 'pseudo':
        classes = list(D.pseudo)
    elif a.which == 'cand':
        classes = cand_classes
    else:  # distract: pseudo + random distractors
        import random
        rng = random.Random(0)
        pool = [c for c in cand_classes if c not in D.pseudo_set]
        classes = list(D.pseudo) + rng.sample(pool, min(a.n_distract, len(pool)))
    print(f'which={a.which}: {len(classes)} classes', flush=True)

    urls = {}
    if os.path.exists(a.urls):
        urls = json.load(open(a.urls))
    todo = [c for c in classes if c not in urls]
    print(f'fetch URLs for {len(todo)} classes (have {len(urls)})', flush=True)

    def one(name):
        s = requests.Session()
        return name, fetch_urls(name, s, a.max_per_class)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, c) for c in todo]
        for f in as_completed(futs):
            name, u = f.result()
            if u is None:
                continue  # blocked; leave uncached so a later pass retries
            urls[name] = u
            done += 1
            if done % 200 == 0 or done == len(todo):
                json.dump(urls, open(a.urls + '.tmp', 'w'))
                os.replace(a.urls + '.tmp', a.urls)
                hit = sum(1 for c in classes if urls.get(c))
                nim = sum(len(urls.get(c) or []) for c in classes)
                print(f'  {done}/{len(todo)} cover={hit}/{len(classes)} imgs={nim} [{time.time()-t0:.0f}s]', flush=True)
    json.dump(urls, open(a.urls, 'w'), indent=1)
    hit = sum(1 for c in classes if urls.get(c))
    nim = sum(len(urls.get(c) or []) for c in classes)
    print(f'URL coverage {hit}/{len(classes)} ({100*hit/len(classes):.1f}%), total imgs {nim}', flush=True)

    if not a.download:
        return

    os.makedirs(a.dl_dir, exist_ok=True)
    files = {}
    if os.path.exists(a.files_out):
        files = json.load(open(a.files_out))

    jobs = []
    for c in classes:
        for i, u in enumerate(urls.get(c) or []):
            safe = c.replace(' ', '_').replace('/', '_')
            dest = os.path.join(a.dl_dir, f'{safe}__{i}.jpg')
            jobs.append((c, u, dest))

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
    with ThreadPoolExecutor(max_workers=a.dl_workers) as ex:
        futs = [ex.submit(dl, j) for j in jobs]
        for k, f in enumerate(as_completed(futs)):
            c, dest = f.result()
            if dest:
                files.setdefault(c, [])
                if dest not in files[c]:
                    files[c].append(dest)
                n_ok += 1
            if (k + 1) % 1000 == 0:
                json.dump(files, open(a.files_out + '.tmp', 'w'))
                os.replace(a.files_out + '.tmp', a.files_out)
                print(f'  dl {k+1}/{len(jobs)} ok={n_ok} [{time.time()-t0:.0f}s]', flush=True)
    json.dump(files, open(a.files_out, 'w'), indent=1)
    cov = sum(1 for c in classes if files.get(c))
    print(f'downloaded {n_ok} files; class coverage {cov}/{len(classes)} ({100*cov/len(classes):.1f}%)', flush=True)


if __name__ == '__main__':
    main()
