"""Fetch public GBIF occurrence media for each species (external image prototypes).

Single-pipeline loophole B: external data OK with disclosure. Excludes competition
Drive / known FishNet-like hosts. Writes outputs/external_image_urls.json then
optionally downloads a small set for proxy classes.

  python research/fetch_external_images.py --limit_classes 500
  python research/fetch_external_images.py --download --max_per_class 2
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
D = os.path.join(ROOT, 'data', 'dl')

# Block hosts that may overlap competition distribution / private mirrors
BLOCK_HOSTS = {
    'drive.google.com', 'docs.google.com', 'googleusercontent.com',
    'fishnet-2023.github.io',
}


def host_ok(url: str) -> bool:
    try:
        h = urlparse(url).netloc.lower()
    except Exception:
        return False
    if not h:
        return False
    return not any(b in h for b in BLOCK_HOSTS)


def gbif_usage_key(name: str, sess: requests.Session) -> int | None:
    r = sess.get('https://api.gbif.org/v1/species/match',
                 params={'name': name, 'kingdom': 'Animalia'}, timeout=30).json()
    return r.get('usageKey') or r.get('speciesKey')


def gbif_media(usage_key: int, sess: requests.Session, limit: int = 5) -> list[str]:
    urls = []
    r = sess.get('https://api.gbif.org/v1/species',
                 params={'limit': 1}, timeout=30)  # warm
    # occurrence search with media
    off = 0
    while len(urls) < limit and off < 200:
        rr = sess.get(
            'https://api.gbif.org/v1/occurrence/search',
            params={
                'taxonKey': usage_key,
                'mediaType': 'StillImage',
                'limit': 50,
                'offset': off,
            },
            timeout=45,
        )
        if rr.status_code != 200:
            break
        data = rr.json()
        results = data.get('results') or []
        if not results:
            break
        for occ in results:
            for m in occ.get('media') or []:
                u = m.get('identifier') or m.get('references')
                if u and host_ok(u) and u not in urls:
                    urls.append(u)
                    if len(urls) >= limit:
                        return urls
        off += 50
        if off >= data.get('count', 0):
            break
    return urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(OUT, 'external_image_urls.json'))
    ap.add_argument('--limit_classes', type=int, default=0)
    ap.add_argument('--max_per_class', type=int, default=3)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--dl_dir', default=os.path.join(OUT, 'external_images'))
    ap.add_argument('--pseudo_only', action='store_true',
                    help='only classes needed for pseudo-unseen proxy candidates')
    a = ap.parse_args()

    classes = list(pickle.load(open(os.path.join(D, 'all_classes.pkl'), 'rb')))
    if a.pseudo_only:
        # rarest 20% seen + all non-kept ≈ cand set from FishData logic
        lab = json.load(open(os.path.join(D, 'label_train.json')))
        from collections import defaultdict
        by = defaultdict(list)
        for fn, c in lab.items():
            by[c].append(fn)
        seen = sorted(by.keys())
        order = sorted(seen, key=lambda c: len(by[c]))
        n_pseudo = int(len(seen) * 0.2)
        kept = set(order[n_pseudo:])
        classes = [c for c in classes if c not in kept]
        print(f'pseudo_only cand classes: {len(classes)}', flush=True)

    if a.limit_classes:
        classes = classes[: a.limit_classes]

    out = {}
    if a.resume and os.path.exists(a.out):
        out = json.load(open(a.out))

    todo = [c for c in classes if c not in out]
    print(f'fetch URLs for {len(todo)} classes (have {len(out)})', flush=True)
    sess_local = requests.Session()

    def one(name):
        s = requests.Session()
        try:
            key = gbif_usage_key(name, s)
            if not key:
                return name, []
            return name, gbif_media(key, s, a.max_per_class)
        except Exception:
            return name, []

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, c) for c in todo]
        for f in as_completed(futs):
            name, urls = f.result()
            out[name] = urls
            done += 1
            if done % 200 == 0 or done == len(todo):
                json.dump(out, open(a.out + '.tmp', 'w'))
                os.replace(a.out + '.tmp', a.out)
                hit = sum(1 for c in out if out[c])
                print(f'  {done}/{len(todo)} hit={hit} [{time.time()-t0:.0f}s]', flush=True)

    json.dump(out, open(a.out, 'w'), indent=1)
    hit = sum(1 for c in classes if out.get(c))
    print(f'wrote {a.out} coverage {hit}/{len(classes)}', flush=True)

    if a.download:
        os.makedirs(a.dl_dir, exist_ok=True)
        meta = {}
        n_dl = 0
        for c, urls in out.items():
            for i, u in enumerate(urls[: a.max_per_class]):
                ext = os.path.splitext(urlparse(u).path)[1] or '.jpg'
                if len(ext) > 5:
                    ext = '.jpg'
                safe = c.replace(' ', '_').replace('/', '_')
                dest = os.path.join(a.dl_dir, f'{safe}__{i}{ext}')
                if os.path.exists(dest):
                    meta.setdefault(c, []).append(dest)
                    continue
                try:
                    r = sess_local.get(u, timeout=40, headers={'User-Agent': 'onet-research/1.0'})
                    if r.status_code == 200 and len(r.content) > 1000:
                        open(dest, 'wb').write(r.content)
                        meta.setdefault(c, []).append(dest)
                        n_dl += 1
                except Exception:
                    continue
            if n_dl and n_dl % 50 == 0:
                print(f'  downloaded {n_dl}', flush=True)
        mp = os.path.join(OUT, 'external_image_files.json')
        json.dump(meta, open(mp, 'w'), indent=1)
        print(f'downloaded {n_dl} files; wrote {mp}', flush=True)


if __name__ == '__main__':
    main()
