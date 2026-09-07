"""Supplement iNat proto URLs: needs_id grade + low-photo densification.

Merges into outputs/inat_image_urls.json (disclosed external iNaturalist photos).
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
from fetch_inat_images import API, UA, RateLimiter, photo_medium, BLOCK  # noqa: E402

LOG = os.path.join(OUT, 'inat_denser_fetch.log')


def fetch_urls_grade(name: str, sess: requests.Session, max_per_class: int, grade: str):
    for attempt in range(8):
        while BLOCK.is_set():
            time.sleep(2.0)
        if LIMITER:
            LIMITER.wait()
        try:
            r = sess.get(API, params={
                'taxon_name': name,
                'quality_grade': grade,
                'photos': 'true',
                'per_page': min(60, max(30, max_per_class * 3)),
                'order_by': 'votes',
                'order': 'desc',
            }, headers=UA, timeout=35)
        except Exception:
            time.sleep(2.0 * (attempt + 1))
            continue
        if r.status_code == 429:
            if not BLOCK.is_set():
                BLOCK.set()
                time.sleep(120.0 + 30 * attempt)
                BLOCK.clear()
            continue
        if r.status_code != 200:
            return []
        urls: list[str] = []
        seen: set[str] = set()
        for o in r.json().get('results', []):
            for p in o.get('photos', []):
                u = photo_medium(p.get('url') or '')
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
                    break
            if len(urls) >= max_per_class:
                break
        if len(urls) < max_per_class:
            for o in r.json().get('results', []):
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
    return None


def merge_urls(existing: list[str], new: list[str], cap: int) -> tuple[list[str], int]:
    out = list(existing or [])
    seen = set(out)
    added = 0
    for u in new:
        if u not in seen:
            out.append(u)
            seen.add(u)
            added += 1
        if len(out) >= cap:
            break
    return out[:cap], added


def pick_targets(D: FishData, urls: dict, files: dict, mode: str, n: int) -> list[str]:
    cand = [D.classes[int(i)] for i in D.cand]
    cand_set = set(cand)
    if mode == 'needs_id_resume':
        uncov = [c for c in cand_set if not files.get(c)]
        return [c for _, c in sorted((len(urls.get(c) or []), c) for c in uncov)[:n]]
    if mode == 'low_photo':
        have = [c for c in files if files.get(c) and c in cand_set]
        return [c for _, c in sorted((len(files.get(c) or []), c) for c in have)[:n]]
    if mode == 'urls_no_files':
        return [c for c in urls if urls.get(c) and not files.get(c)][:n]
    raise ValueError(mode)


LIMITER: RateLimiter | None = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['needs_id_resume', 'low_photo', 'urls_no_files'], required=True)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--max_per_class', type=int, default=30)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--rate', type=float, default=35.0)
    ap.add_argument('--urls', default=os.path.join(OUT, 'inat_image_urls.json'))
    ap.add_argument('--files', default=os.path.join(OUT, 'inat_image_files.json'))
    a = ap.parse_args()

    global LIMITER
    LIMITER = RateLimiter(a.rate)

    D = FishData()
    urls = json.load(open(a.urls)) if os.path.exists(a.urls) else {}
    files = json.load(open(a.files)) if os.path.exists(a.files) else {}
    targets = pick_targets(D, urls, files, a.mode, a.n)
    print(f'mode={a.mode} targets={len(targets)}', flush=True)
    open(LOG, 'a').write(f'\n=== mode={a.mode} n={len(targets)} rate={a.rate} ===\n')

    def one(name):
        sess = requests.Session()
        cur = urls.get(name) or []
        need = max(0, a.max_per_class - len(cur))
        if need <= 0 and a.mode != 'needs_id_resume':
            return name, 0, False
        extra = []
        for grade in ('research,needs_id', 'research,needs_id,casual', 'research'):
            if len(cur) + len(extra) >= a.max_per_class:
                break
            got = fetch_urls_grade(name, sess, a.max_per_class, grade)
            if got is None:
                return name, 0, True
            for u in got:
                if u not in cur and u not in extra:
                    extra.append(u)
                if len(cur) + len(extra) >= a.max_per_class:
                    break
        merged, added = merge_urls(cur, extra, a.max_per_class)
        if added:
            urls[name] = merged
        return name, added, False

    t0 = time.time()
    total_added = 0
    blocked = 0
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, c): c for c in targets}
        for f in as_completed(futs):
            name, added, was_blocked = f.result()
            done += 1
            if was_blocked:
                blocked += 1
            else:
                total_added += added
            if done % 25 == 0 or done == len(targets):
                json.dump(urls, open(a.urls + '.tmp', 'w'))
                os.replace(a.urls + '.tmp', a.urls)
                msg = f'  {done}/{len(targets)} added_total={total_added} blocked={blocked} [{time.time()-t0:.0f}s]'
                print(msg, flush=True)
                open(LOG, 'a').write(msg + '\n')
    json.dump(urls, open(a.urls, 'w'), indent=1)
    summary = f'done mode={a.mode} classes={len(targets)} urls_added={total_added} blocked={blocked}'
    print(summary, flush=True)
    open(LOG, 'a').write(summary + '\n')


if __name__ == '__main__':
    main()
