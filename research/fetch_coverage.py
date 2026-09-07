"""Fetch external image URLs for UNCOVERED unseen-route classes (disclosed sources).

Sources (DISCLOSURE.md):
  - GBIF occurrence StillImage (api.gbif.org), incl. synonym / variant name matches
  - Wikimedia Commons MediaSearch (commons.wikimedia.org)
  - Openverse API (api.openverse.org) — CC-licensed web images
  - EOL page media (eol.org) when available

Writes outputs/coverage_urls_extra.json (new hits only) then merge into coverage_urls.json:
  python research/merge_coverage_urls.py

  python research/fetch_coverage.py --only_missing --workers 24
  python research/fetch_coverage.py --probe 200
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')

BLOCK_HOSTS = {
    'drive.google.com', 'docs.google.com', 'googleusercontent.com',
    'fishnet-2023.github.io',
}
UA = {'User-Agent': 'onet-cv4e-research/1.0 (academic; contact cv4e.workshop@gmail.com)'}


def host_ok(url: str) -> bool:
    try:
        h = urlparse(url).netloc.lower()
    except Exception:
        return False
    return bool(h) and not any(b in h for b in BLOCK_HOSTS)


def add_urls(dst: list[str], src: list[str], limit: int) -> None:
    for u in src:
        if len(dst) >= limit:
            return
        if u and host_ok(u) and u not in dst:
            dst.append(u)


def name_variants(scientific_name: str) -> list[str]:
    """Binomial + stripped authority + subspecies→species + genus-only (last resort)."""
    name = scientific_name.strip()
    out: list[str] = []
    seen: set[str] = set()

    def push(n: str) -> None:
        n = re.sub(r'\s+', ' ', n.strip())
        if not n or n in seen:
            return
        seen.add(n)
        out.append(n)

    push(name)
    no_auth = re.sub(r'\s+\([^)]*\)\s*$', '', name).strip()
    push(no_auth)
    no_year = re.sub(r'\s+\d{4}\s*$', '', no_auth).strip()
    push(no_year)
    parts = name.split()
    if len(parts) >= 3:
        push(' '.join(parts[:2]))
    if len(parts) >= 4:
        push(' '.join(parts[:3]))
    return out


def gbif_match_keys(name: str, s: requests.Session) -> list[int]:
    keys: list[int] = []
    seen: set[int] = set()

    def add(k) -> None:
        if k and k not in seen:
            seen.add(int(k))
            keys.append(int(k))

    try:
        r = s.get(
            'https://api.gbif.org/v1/species/match',
            params={'name': name, 'kingdom': 'Animalia'},
            timeout=30,
            headers=UA,
        ).json()
        add(r.get('usageKey') or r.get('speciesKey'))
        add(r.get('acceptedUsageKey'))
    except Exception:
        pass
    try:
        sr = s.get(
            'https://api.gbif.org/v1/species/search',
            params={'q': name, 'rank': 'SPECIES', 'limit': 5, 'kingdom': 'Animalia'},
            timeout=30,
            headers=UA,
        ).json()
        for row in sr.get('results') or []:
            add(row.get('key'))
            add(row.get('acceptedKey'))
    except Exception:
        pass
    for k in list(keys):
        try:
            syn = s.get(
                f'https://api.gbif.org/v1/species/{k}/synonyms',
                params={'limit': 20},
                timeout=25,
                headers=UA,
            ).json()
            for row in syn.get('results') or []:
                add(row.get('key'))
        except Exception:
            pass
    return keys


def gbif_media(key: int, s: requests.Session, limit: int) -> list[str]:
    urls: list[str] = []
    off = 0
    while len(urls) < limit and off < 400:
        try:
            rr = s.get(
                'https://api.gbif.org/v1/occurrence/search',
                params={'taxonKey': key, 'mediaType': 'StillImage', 'limit': 100, 'offset': off},
                timeout=45,
                headers=UA,
            )
            if rr.status_code != 200:
                break
            data = rr.json()
        except Exception:
            break
        results = data.get('results') or []
        if not results:
            break
        for occ in results:
            for m in occ.get('media') or []:
                u = m.get('identifier') or m.get('references')
                if u:
                    add_urls(urls, [u], limit)
        off += 100
        if off >= data.get('count', 0):
            break
    return urls


def gbif_fetch(name: str, s: requests.Session, limit: int) -> list[str]:
    urls: list[str] = []
    for variant in name_variants(name):
        if len(urls) >= limit:
            break
        for key in gbif_match_keys(variant, s):
            add_urls(urls, gbif_media(key, s, limit - len(urls)), limit)
            if len(urls) >= limit:
                break
    return urls


def commons_media(name: str, s: requests.Session, limit: int) -> list[str]:
    urls: list[str] = []
    parts = name.split()
    binomial = ' '.join(parts[:2]) if len(parts) >= 2 else name
    queries = [
        f'"{name}" filetype:bitmap',
        f'{name} fish filetype:bitmap',
        f'{binomial} fish filetype:bitmap',
        f'{binomial} filetype:bitmap',
    ]
    for q in queries:
        if len(urls) >= limit:
            break
        try:
            r = s.get(
                'https://commons.wikimedia.org/w/api.php',
                params={
                    'action': 'query',
                    'format': 'json',
                    'generator': 'search',
                    'gsrsearch': q,
                    'gsrnamespace': 6,
                    'gsrlimit': min(limit, 10),
                    'prop': 'imageinfo',
                    'iiprop': 'url',
                    'iiurlwidth': 800,
                },
                timeout=40,
                headers=UA,
            )
            if r.status_code != 200:
                continue
            pages = (r.json().get('query') or {}).get('pages') or {}
            for p in pages.values():
                for info in p.get('imageinfo') or []:
                    u = info.get('url') or info.get('thumburl')
                    add_urls(urls, [u] if u else [], limit)
        except Exception:
            pass
    return urls


def openverse_media(name: str, s: requests.Session, limit: int) -> list[str]:
    urls: list[str] = []
    for q in name_variants(name)[:3]:
        if len(urls) >= limit:
            break
        try:
            r = s.get(
                'https://api.openverse.org/v1/images/',
                params={'q': q, 'page_size': min(limit, 20), 'license_type': 'commercial,modification'},
                timeout=35,
                headers=UA,
            )
            if r.status_code != 200:
                continue
            for row in r.json().get('results') or []:
                u = row.get('url') or row.get('thumbnail')
                add_urls(urls, [u] if u else [], limit)
        except Exception:
            pass
    return urls


def eol_media(name: str, s: requests.Session, limit: int) -> list[str]:
    urls: list[str] = []
    for q in name_variants(name)[:2]:
        if len(urls) >= limit:
            break
        try:
            r = s.get(
                'https://eol.org/api/search/1.0.json',
                params={'q': q, 'page': 1},
                timeout=30,
                headers=UA,
            )
            if r.status_code != 200:
                continue
            for res in (r.json().get('results') or [])[:4]:
                pid = res.get('id')
                if not pid:
                    continue
                pr = s.get(
                    f'https://eol.org/api/pages/1.0/{pid}.json',
                    params={
                        'images_per_page': limit,
                        'videos_per_page': 0,
                        'sounds_per_page': 0,
                        'maps_per_page': 0,
                        'texts_per_page': 0,
                    },
                    timeout=30,
                    headers=UA,
                )
                if pr.status_code != 200:
                    continue
                for im in pr.json().get('media') or []:
                    u = im.get('eolMediaURL') or im.get('originalUrl') or im.get('source')
                    add_urls(urls, [u] if u else [], limit)
        except Exception:
            pass
    return urls


def fetch_one(name: str, max_per_class: int, use_gbif: bool, use_commons: bool,
              use_openverse: bool, use_eol: bool) -> tuple[str, list[str]]:
    s = requests.Session()
    urls: list[str] = []
    if use_gbif:
        add_urls(urls, gbif_fetch(name, s, max_per_class), max_per_class)
    if use_commons and len(urls) < max_per_class:
        add_urls(urls, commons_media(name, s, max_per_class - len(urls)), max_per_class)
    if use_openverse and len(urls) < max_per_class:
        add_urls(urls, openverse_media(name, s, max_per_class - len(urls)), max_per_class)
    if use_eol and len(urls) < max_per_class:
        add_urls(urls, eol_media(name, s, max_per_class - len(urls)), max_per_class)
    return name, urls[:max_per_class]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', default=os.path.join(OUT, 'uncovered_unseen_classes.json'))
    ap.add_argument('--existing', default=os.path.join(OUT, 'coverage_urls.json'))
    ap.add_argument('--out', default=os.path.join(OUT, 'coverage_urls_extra.json'))
    ap.add_argument('--max_per_class', type=int, default=12)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--probe', type=int, default=0)
    ap.add_argument('--only_missing', action='store_true',
                    help='Only classes with no URLs in --existing')
    ap.add_argument('--no-gbif', action='store_true')
    ap.add_argument('--no-commons', action='store_true')
    ap.add_argument('--no-openverse', action='store_true')
    ap.add_argument('--no-eol', action='store_true')
    a = ap.parse_args()

    classes = json.load(open(a.classes))
    existing: dict = {}
    if os.path.exists(a.existing):
        existing = json.load(open(a.existing))

    if a.only_missing:
        todo = [c for c in classes if not existing.get(c)]
    else:
        extra_prev = json.load(open(a.out)) if os.path.exists(a.out) and not a.probe else {}
        todo = [c for c in classes if c not in extra_prev and not existing.get(c)]

    if a.probe:
        todo = todo[: a.probe]

    print(
        f'fetch {len(todo)} classes (existing hits {sum(1 for c in classes if existing.get(c))}/'
        f'{len(classes)}); writing {a.out}',
        flush=True,
    )

    extra: dict = {}
    if os.path.exists(a.out) and not a.probe and not a.only_missing:
        extra = json.load(open(a.out))
    elif a.only_missing and os.path.exists(a.out) and not a.probe:
        extra = json.load(open(a.out))

    use_gbif = not a.no_gbif
    use_commons = not a.no_commons
    use_openverse = not a.no_openverse
    use_eol = not a.no_eol

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [
            ex.submit(
                fetch_one, c, a.max_per_class, use_gbif, use_commons, use_openverse, use_eol,
            )
            for c in todo
        ]
        for f in as_completed(futs):
            name, urls = f.result()
            extra[name] = urls
            done += 1
            if done % 100 == 0 or done == len(todo):
                new_hit = sum(1 for c in todo[:done] if extra.get(c))
                if not a.probe:
                    json.dump(extra, open(a.out + '.tmp', 'w'))
                    os.replace(a.out + '.tmp', a.out)
                merged_hit = sum(
                    1 for c in classes if existing.get(c) or extra.get(c)
                )
                print(
                    f'  {done}/{len(todo)} batch_new_hit={new_hit} '
                    f'merged_hit={merged_hit}/{len(classes)} [{time.time()-t0:.0f}s]',
                    flush=True,
                )

    if not a.probe:
        json.dump(extra, open(a.out, 'w'))
    new_hit = sum(1 for c in todo if extra.get(c))
    merged = sum(1 for c in classes if existing.get(c) or extra.get(c))
    print(
        f'DONE extra new_hits={new_hit}/{len(todo)}; merged {merged}/{len(classes)} '
        f'({100 * merged / max(len(classes), 1):.1f}%)',
        flush=True,
    )


if __name__ == '__main__':
    main()
