"""Extra URL discovery for uncovered unseen-route classes (GBIF/Wikimedia pass missed).

Adds: GBIF multi-key + scientificName/genus-species occurrence search, synonym keys,
Wikimedia query variants (incl. GBIF vernacular/common names), Openverse, iNaturalist
(all quality grades, rate-limited), Wikipedia REST thumbnails.

Writes only NEW hits to outputs/coverage_urls_extra.json (does not touch coverage_urls.json).

  python research/fetch_coverage_extra.py --probe 300
  python research/fetch_coverage_extra.py --workers 32
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote, urlparse

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


def strip_binomial(name: str) -> str:
    """Genus species from binomial + optional infraspecific epithet."""
    parts = name.strip().split()
    if len(parts) >= 2:
        return f'{parts[0]} {parts[1]}'
    return name.strip()


def name_variants(name: str, common: str | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str):
        s = re.sub(r'\s+', ' ', s.strip())
        if not s or s.lower() in seen:
            return
        seen.add(s.lower())
        out.append(s)

    add(name)
    add(strip_binomial(name))
    g, sp = (name.split() + ['', ''])[:2]
    if g and sp:
        add(f'{g} {sp}')
        add(f'"{g} {sp}"')
    if common:
        add(common)
        if g and sp:
            add(f'{common} {g}')
    return out


class InatLimiter:
    def __init__(self, rpm: float = 45.0):
        self.interval = 60.0 / max(rpm, 1.0)
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        if t > now:
            time.sleep(t - now)


INAT_LIMITER = InatLimiter(50.0)


def gbif_match(name: str, s: requests.Session) -> dict:
    try:
        return s.get(
            'https://api.gbif.org/v1/species/match',
            params={'name': name, 'kingdom': 'Animalia', 'verbose': True},
            timeout=35,
            headers=UA,
        ).json()
    except Exception:
        return {}


def gbif_taxon_keys(name: str, s: requests.Session) -> list[int]:
    keys: list[int] = []
    seen: set[int] = set()

    def add(k):
        if k and k not in seen:
            seen.add(int(k))
            keys.append(int(k))

    m = gbif_match(name, s)
    add(m.get('usageKey'))
    add(m.get('speciesKey'))
    add(m.get('acceptedUsageKey'))
    canon = m.get('canonicalName') or m.get('scientificName')
    if canon and canon != name:
        m2 = gbif_match(canon, s)
        add(m2.get('usageKey'))

    try:
        sr = s.get(
            'https://api.gbif.org/v1/species/search',
            params={'q': strip_binomial(name), 'rank': 'SPECIES', 'limit': 8,
                    'highertaxon_key': 2},  # Animalia
            timeout=35,
            headers=UA,
        ).json()
        for hit in sr.get('results') or []:
            if hit.get('kingdom') not in (None, 'Animalia'):
                continue
            sn = hit.get('scientificName') or ''
            if strip_binomial(sn).lower() != strip_binomial(name).lower():
                continue
            add(hit.get('key') or hit.get('usageKey'))
    except Exception:
        pass

    uk = m.get('usageKey')
    if uk:
        try:
            sy = s.get(
                f'https://api.gbif.org/v1/species/{uk}/synonyms',
                params={'limit': 40},
                timeout=35,
                headers=UA,
            ).json()
            for row in sy.get('results') or []:
                add(row.get('key') or row.get('usageKey'))
        except Exception:
            pass
    return keys


def gbif_occurrence_urls(s: requests.Session, params: dict, limit: int) -> list[str]:
    urls: list[str] = []
    off = 0
    while len(urls) < limit and off < 250:
        try:
            p = dict(params)
            p.update({'mediaType': 'StillImage', 'limit': 50, 'offset': off})
            rr = s.get('https://api.gbif.org/v1/occurrence/search', params=p, timeout=45, headers=UA)
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
                if u and host_ok(u) and u not in urls:
                    urls.append(u)
                    if len(urls) >= limit:
                        return urls
        off += 50
        if off >= data.get('count', 0):
            break
    return urls


def gbif_media_for_name(name: str, s: requests.Session, limit: int) -> list[str]:
    urls: list[str] = []
    parts = name.split()
    for key in gbif_taxon_keys(name, s):
        for u in gbif_occurrence_urls(s, {'taxonKey': key}, limit - len(urls)):
            if u not in urls:
                urls.append(u)
            if len(urls) >= limit:
                return urls
    if len(urls) < limit and len(parts) >= 2:
        for u in gbif_occurrence_urls(
            s,
            {'genus': parts[0], 'species': parts[1], 'kingdom': 'Animalia', 'phylum': 'Chordata'},
            limit - len(urls),
        ):
            if u not in urls:
                urls.append(u)
    if len(urls) < limit:
        for u in gbif_occurrence_urls(
            s,
            {'scientificName': strip_binomial(name), 'kingdom': 'Animalia', 'phylum': 'Chordata'},
            limit - len(urls),
        ):
            if u not in urls:
                urls.append(u)
    return urls[:limit]


def commons_search(query: str, s: requests.Session, limit: int) -> list[str]:
    urls: list[str] = []
    try:
        r = s.get(
            'https://commons.wikimedia.org/w/api.php',
            params={
                'action': 'query', 'format': 'json', 'generator': 'search',
                'gsrsearch': query, 'gsrnamespace': 6, 'gsrlimit': min(limit, 50),
                'prop': 'imageinfo', 'iiprop': 'url', 'iiurlwidth': 800,
            },
            timeout=40,
            headers=UA,
        ).json()
        pages = (r.get('query') or {}).get('pages') or {}
        for p in pages.values():
            for info in p.get('imageinfo') or []:
                u = info.get('thumburl') or info.get('url')
                if u and host_ok(u) and u not in urls:
                    urls.append(u)
    except Exception:
        pass
    return urls[:limit]


def commons_for_name(name: str, common: str | None, s: requests.Session, limit: int) -> list[str]:
    urls: list[str] = []
    g, sp = (name.split() + ['', ''])[:2]
    queries = [
        f'{name} filetype:bitmap',
        f'intitle:"{strip_binomial(name)}"',
        f'{strip_binomial(name)} fish filetype:bitmap',
    ]
    if g and sp:
        queries.append(f'{g} {sp} filetype:bitmap')
    if common:
        queries.append(f'{common} fish filetype:bitmap')
        queries.append(f'"{common}" {g} filetype:bitmap')
    for q in queries:
        for u in commons_search(q, s, limit - len(urls)):
            if u not in urls:
                urls.append(u)
            if len(urls) >= limit:
                return urls
    return urls


def openverse_for_queries(queries: list[str], s: requests.Session, limit: int) -> list[str]:
    urls: list[str] = []
    for q in queries:
        if len(urls) >= limit:
            break
        try:
            r = s.get(
                'https://api.openverse.org/v1/images/',
                params={'q': q, 'page_size': min(20, limit), 'license_type': 'commercial,modification'},
                timeout=35,
                headers=UA,
            )
            if r.status_code != 200:
                continue
            for hit in r.json().get('results') or []:
                u = hit.get('url') or hit.get('thumbnail') or hit.get('detail_url')
                if u and host_ok(u) and u not in urls:
                    urls.append(u)
                if len(urls) >= limit:
                    return urls
        except Exception:
            pass
    return urls


def inat_photo_url(raw: str) -> str:
    for sz in ('square', 'thumb', 'small'):
        if f'/{sz}.' in raw:
            return raw.replace(f'/{sz}.', '/medium.')
    return raw


def inat_for_name(name: str, s: requests.Session, limit: int) -> list[str]:
    urls: list[str] = []
    param_sets = [
        {'taxon_name': name, 'photos': 'true', 'per_page': 40,
         'quality_grade': 'research,needs_id,casual', 'order_by': 'votes'},
        {'taxon_name': strip_binomial(name), 'photos': 'true', 'per_page': 40,
         'quality_grade': 'research,needs_id,casual'},
        {'q': strip_binomial(name), 'photos': 'true', 'per_page': 30},
    ]
    for params in param_sets:
        if len(urls) >= limit:
            break
        for attempt in range(4):
            INAT_LIMITER.wait()
            try:
                r = s.get('https://api.inaturalist.org/v1/observations', params=params,
                          timeout=35, headers=UA)
            except Exception:
                break
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code != 200:
                break
            for obs in r.json().get('results') or []:
                for p in obs.get('photos') or []:
                    u = inat_photo_url(p.get('url') or '')
                    if u and host_ok(u) and u not in urls:
                        urls.append(u)
                    if len(urls) >= limit:
                        return urls
            break
    return urls


def wiki_thumbnail(name: str, s: requests.Session) -> list[str]:
    title = name.replace(' ', '_')
    try:
        r = s.get(
            f'https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}',
            timeout=25,
            headers=UA,
        )
        if r.status_code != 200:
            return []
        j = r.json()
        if j.get('type') == 'disambiguation':
            return []
        thumb = (j.get('thumbnail') or {}).get('source')
        if thumb and host_ok(thumb):
            return [thumb]
    except Exception:
        pass
    return []


def gbif_vernacular(name: str, s: requests.Session) -> str | None:
    keys = gbif_taxon_keys(name, s)
    if not keys:
        return None
    try:
        vr = s.get(
            f'https://api.gbif.org/v1/species/{keys[0]}/vernacularNames',
            params={'limit': 20},
            timeout=25,
            headers=UA,
        ).json()
        for row in vr.get('results') or []:
            if row.get('language') in ('eng', 'en', None):
                vn = row.get('vernacularName')
                if vn:
                    return vn
        for row in vr.get('results') or []:
            vn = row.get('vernacularName')
            if vn:
                return vn
    except Exception:
        pass
    return None


def fetch_one(name: str, common_names: dict, max_per_class: int, sources: set[str]) -> tuple[str, list[str]]:
    s = requests.Session()
    urls: list[str] = []
    common = common_names.get(name)
    if 'gbif' in sources and len(urls) < max_per_class:
        for u in gbif_media_for_name(name, s, max_per_class - len(urls)):
            if u not in urls:
                urls.append(u)
    if 'commons' in sources and len(urls) < max_per_class:
        if not common and 'gbif' in sources:
            common = gbif_vernacular(name, s) or common
        for u in commons_for_name(name, common, s, max_per_class - len(urls)):
            if u not in urls:
                urls.append(u)
    if 'openverse' in sources and len(urls) < max_per_class:
        qs = name_variants(name, common)
        for u in openverse_for_queries(qs[:6], s, max_per_class - len(urls)):
            if u not in urls:
                urls.append(u)
    if 'inat' in sources and len(urls) < max_per_class:
        for u in inat_for_name(name, s, max_per_class - len(urls)):
            if u not in urls:
                urls.append(u)
    if 'wiki' in sources and len(urls) < max_per_class:
        for u in wiki_thumbnail(name, s):
            if u not in urls:
                urls.append(u)
    return name, urls[:max_per_class]


def load_existing(base_path: str, extra_path: str) -> dict:
    out: dict = {}
    if os.path.exists(base_path):
        out.update(json.load(open(base_path)))
    if os.path.exists(extra_path):
        for k, v in json.load(open(extra_path)).items():
            if v:
                out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', default=os.path.join(OUT, 'uncovered_unseen_classes.json'))
    ap.add_argument('--base_urls', default=os.path.join(OUT, 'coverage_urls.json'))
    ap.add_argument('--out', default=os.path.join(OUT, 'coverage_urls_extra.json'))
    ap.add_argument('--common_names', default=os.path.join(OUT, 'common_names.json'))
    ap.add_argument('--max_per_class', type=int, default=12)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--probe', type=int, default=0)
    ap.add_argument('--sources', default='gbif,commons,openverse,inat,wiki',
                    help='comma-separated: gbif,commons,openverse,inat,wiki')
    a = ap.parse_args()

    sources = set(x.strip() for x in a.sources.split(',') if x.strip())
    classes = json.load(open(a.classes))
    common_names = json.load(open(a.common_names)) if os.path.exists(a.common_names) else {}
    existing = load_existing(a.base_urls, a.out)

    todo = [c for c in classes if not existing.get(c)]
    if a.probe:
        todo = todo[:a.probe]

    extra: dict = {}
    if os.path.exists(a.out) and not a.probe:
        extra = json.load(open(a.out))

    print(f'extra fetch: {len(todo)} classes without URLs (existing hits {sum(1 for c in classes if existing.get(c))}/{len(classes)}) '
          f'sources={sorted(sources)}', flush=True)

    done = 0
    new_hit = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_one, c, common_names, a.max_per_class, sources): c for c in todo}
        for f in as_completed(futs):
            name, urls = f.result()
            extra[name] = urls
            done += 1
            if urls:
                new_hit += 1
            if done % 100 == 0 or done == len(todo):
                if not a.probe:
                    json.dump(extra, open(a.out + '.tmp', 'w'))
                    os.replace(a.out + '.tmp', a.out)
                batch_hit = sum(1 for c in todo[:done] if extra.get(c))
                print(f'  {done}/{len(todo)} batch_hit={batch_hit} new_hit={new_hit} '
                      f'[{time.time()-t0:.0f}s]', flush=True)

    if not a.probe:
        json.dump(extra, open(a.out, 'w'))

    merged = load_existing(a.base_urls, a.out)
    total_hit = sum(1 for c in classes if merged.get(c))
    extra_only = sum(1 for c in classes if not existing.get(c) and merged.get(c))
    print(f'DONE extra-only new hits {extra_only}; merged coverage {total_hit}/{len(classes)} '
          f'({100*total_hit/len(classes):.1f}%) urls={sum(len(v) for v in merged.values() if v)}', flush=True)


if __name__ == '__main__':
    main()
