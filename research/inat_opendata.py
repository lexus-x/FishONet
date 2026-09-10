"""Build iNat photo URLs for competition species from the iNaturalist Open Data
S3 dumps (no API rate limit). Streams observations + photos from S3.

Pipeline:
  1. taxa.csv.gz (local)      -> target name -> {taxon_id}
  2. observations.csv.gz (S3) -> research-grade obs_uuid per target taxon (cap)
  3. photos.csv.gz (S3)       -> photo_id/extension per obs_uuid (cap per class)
  -> outputs/inat_image_urls.json  (merged with existing)

Photo URL: https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/medium.{ext}

  python research/inat_opendata.py --stage taxa
  python research/inat_opendata.py --stage obs
  python research/inat_opendata.py --stage obs_needs_id
  python research/inat_opendata.py --stage photos --photo-cap 24
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, OUT  # noqa: E402

TAXA = os.path.join(OUT, 'inat_taxa.csv.gz')
S3 = 's3://inaturalist-open-data'
URL_BASE = 'https://inaturalist-open-data.s3.amazonaws.com/photos'

TARGET_TAXA = os.path.join(OUT, 'inat_od_target_taxa.json')     # taxon_id -> name
OBS_MAP = os.path.join(OUT, 'inat_od_obs.json')                 # obs_uuid -> name (research)
OBS_MAP_NEEDS = os.path.join(OUT, 'inat_od_obs_needs_id.json')
URLS = os.path.join(OUT, 'inat_image_urls.json')

# INAT_SUFFIX=seen -> separate artifact set (does not touch the cand-pool files).
# INAT_TARGETS_JSON=<json list of names> overrides the D.cand target set.
_SFX = os.environ.get('INAT_SUFFIX')
if _SFX:
    TARGET_TAXA = TARGET_TAXA.replace('.json', f'_{_SFX}.json')
    OBS_MAP = OBS_MAP.replace('.json', f'_{_SFX}.json')
    OBS_MAP_NEEDS = OBS_MAP_NEEDS.replace('.json', f'_{_SFX}.json')
    URLS = URLS.replace('.json', f'_{_SFX}.json')

# Defaults; overridden by CLI
OBS_CAP = 60
PHOTO_CAP = 16
OBS_GRADES = ('research',)
OBS_MAP_OUT = OBS_MAP


def s3_stream(key):
    p = subprocess.Popen(['aws', 's3', 'cp', '--no-sign-request', f'{S3}/{key}', '-'],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1024 * 1024)
    return p


def stage_taxa():
    tj = os.environ.get('INAT_TARGETS_JSON')
    if tj:
        targets = {c.lower(): c for c in json.load(open(tj))}
    else:
        D = FishData()
        targets = {c.lower(): c for c in [D.classes[int(i)] for i in D.cand]}
    print(f'{len(targets)} target names', flush=True)
    tid2name = {}
    name_active = {}  # name_lower -> has an active mapping already
    t0 = time.time()
    with gzip.open(TAXA, 'rt') as f:
        next(f)
        for ln in f:
            parts = ln.rstrip('\n').split('\t')
            if len(parts) < 6:
                continue
            tid, _anc, _rl, rank, name, active = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            nl = name.lower()
            if nl in targets:
                # keep all taxon_ids that map to a target name (synonyms boost coverage)
                tid2name[tid] = targets[nl]
                if active == 'true':
                    name_active[nl] = True
    covered = len(set(tid2name.values()))
    json.dump(tid2name, open(TARGET_TAXA, 'w'))
    print(f'[{time.time()-t0:.0f}s] taxon_ids matched={len(tid2name)} distinct names={covered}/{len(targets)}', flush=True)


def stage_obs():
    global OBS_GRADES, OBS_CAP, OBS_MAP_OUT
    tid2name = json.load(open(TARGET_TAXA))
    grades = set(OBS_GRADES)
    print(f'{len(tid2name)} target taxon_ids grades={sorted(grades)} cap={OBS_CAP}', flush=True)
    per_taxon = {}
    obs2name = {}
    t0 = time.time()
    n = 0
    p = s3_stream('observations.csv.gz')
    with gzip.open(p.stdout, 'rt') as f:
        try:
            next(f)
        except StopIteration:
            pass
        for ln in f:
            n += 1
            if n % 20_000_000 == 0:
                print(f'  obs {n//1_000_000}M kept={len(obs2name)} [{time.time()-t0:.0f}s]', flush=True)
            parts = ln.split('\t')
            if len(parts) < 7:
                continue
            if parts[6] not in grades:
                continue
            tid = parts[5]
            nm = tid2name.get(tid)
            if nm is None:
                continue
            if per_taxon.get(tid, 0) >= OBS_CAP:
                continue
            per_taxon[tid] = per_taxon.get(tid, 0) + 1
            obs2name[parts[0]] = nm
    p.stdout.close()
    json.dump(obs2name, open(OBS_MAP_OUT, 'w'))
    print(f'[{time.time()-t0:.0f}s] obs scanned={n} kept obs_uuids={len(obs2name)} '
          f'distinct classes={len(set(obs2name.values()))} -> {OBS_MAP_OUT}', flush=True)


def _merge_urls(urls: dict, per_class: dict, cap: int, boost_low: int) -> int:
    added = 0
    for nm, lst in per_class.items():
        cap_i = cap
        if boost_low and len(urls.get(nm) or []) < 8:
            cap_i = max(cap_i, boost_low)
        merged = list(urls.get(nm) or [])
        seen = set(merged)
        for u in lst:
            if u not in seen:
                merged.append(u)
                seen.add(u)
            if len(merged) >= cap_i:
                break
        if len(merged) > len(urls.get(nm) or []):
            if not urls.get(nm):
                added += 1
            urls[nm] = merged[:cap_i]
    return added


def stage_photos():
    obs_paths = [OBS_MAP]
    if os.path.exists(OBS_MAP_NEEDS) and OBS_MAP_NEEDS not in obs_paths:
        obs_paths.append(OBS_MAP_NEEDS)
    obs2name = {}
    for op in obs_paths:
        chunk = json.load(open(op))
        obs2name.update(chunk)
    print(f'{len(obs2name)} target obs_uuids (from {len(obs_paths)} maps)', flush=True)
    per_class = {}     # name -> list of urls
    t0 = time.time()
    n = 0
    p = s3_stream('photos.csv.gz')
    with gzip.open(p.stdout, 'rt') as f:
        try:
            next(f)
        except StopIteration:
            pass
        for ln in f:
            n += 1
            if n % 40_000_000 == 0:
                nu = sum(len(v) for v in per_class.values())
                print(f'  photos {n//1_000_000}M classes={len(per_class)} urls={nu} [{time.time()-t0:.0f}s]', flush=True)
            # columns: photo_uuid,photo_id,observation_uuid,observer_id,extension,...
            parts = ln.split('\t')
            if len(parts) < 5:
                continue
            nm = obs2name.get(parts[2])
            if nm is None:
                continue
            lst = per_class.setdefault(nm, [])
            if len(lst) >= PHOTO_CAP:
                continue
            ext = parts[4] or 'jpg'
            lst.append(f'{URL_BASE}/{parts[1]}/medium.{ext}')
    p.stdout.close()

    urls = json.load(open(URLS)) if os.path.exists(URLS) else {}
    added = _merge_urls(urls, per_class, PHOTO_CAP, boost_low=30)
    json.dump(urls, open(URLS, 'w'), indent=1)
    cov = sum(1 for c in urls if urls[c])
    print(f'[{time.time()-t0:.0f}s] photos scanned={n} classes_with_photos={len(per_class)} '
          f'new_classes_added={added}; total url classes={cov}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['taxa', 'obs', 'obs_needs_id', 'photos'], required=True)
    ap.add_argument('--obs-cap', type=int, default=None)
    ap.add_argument('--photo-cap', type=int, default=None)
    a = ap.parse_args()
    if a.stage == 'obs_needs_id':
        OBS_GRADES = ('needs_id',)
        OBS_MAP_OUT = OBS_MAP_NEEDS
        if a.obs_cap is None:
            OBS_CAP = 80
        else:
            OBS_CAP = a.obs_cap
    elif a.stage == 'obs':
        OBS_GRADES = ('research',)
        OBS_MAP_OUT = OBS_MAP
        if a.obs_cap is not None:
            OBS_CAP = a.obs_cap
    if a.photo_cap is not None:
        PHOTO_CAP = a.photo_cap
    if a.stage == 'photos' and a.photo_cap is None:
        PHOTO_CAP = 24
    {'taxa': stage_taxa, 'obs': stage_obs, 'obs_needs_id': stage_obs, 'photos': stage_photos}[a.stage]()
