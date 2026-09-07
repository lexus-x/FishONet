"""Stream TreeOfLife-200M BioCLIP-2 embeddings → mean protos for competition classes.

External disclosed data (imageomics/TreeOfLife-200M-Embeddings, BioCLIP-2 float16).
Downloads one parquet shard at a time, filters to competition scientific_name set,
accumulates per-class mean, deletes shard. Disk-friendly (~0.5GB peak per shard).

  conda activate onet
  python research/tol_bioclip2_protos.py 2>&1 | tee outputs/tol_bioclip2_protos.log
"""
from __future__ import annotations

import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
TMP = os.path.join(OUT, 'tol_shard_tmp')
CKPT = os.path.join(OUT, 'tol_bioclip2_accum.pt')
FINAL = os.path.join(OUT, 'tol_protos_bioclip2.pt')
REPO = 'imageomics/TreeOfLife-200M-Embeddings'
NSHARD = 666
CAP = int(os.environ.get('TOL_CAP', '32'))
DIM = 768


def load_names():
    classes = list(pickle.load(open(os.path.join(ROOT, 'data/dl/all_classes.pkl'), 'rb')))
    # Map scientific_name variants → class index
    name2i = {}
    for i, c in enumerate(classes):
        name2i[c] = i
        name2i[c.replace('_', ' ')] = i
        name2i[c.lower()] = i
        name2i[c.replace('_', ' ').lower()] = i
    return classes, name2i


def load_ckpt(ncls):
    if not os.path.exists(CKPT):
        return {
            'sum': np.zeros((ncls, DIM), dtype=np.float64),
            'cnt': np.zeros(ncls, dtype=np.int64),
            'next_shard': 0,
            'rows_kept': 0,
            'rows_seen': 0,
        }
    d = torch.load(CKPT, weights_only=False)
    return d


def save_ckpt(state):
    torch.save(state, CKPT + '.tmp')
    os.replace(CKPT + '.tmp', CKPT)


def download_shard(i, retries=8):
    name = f'bioclip-2_float16/train-{i:05d}-of-{NSHARD:05d}.parquet'
    os.makedirs(TMP, exist_ok=True)
    wait = 5.0
    for attempt in range(retries):
        try:
            path = hf_hub_download(
                REPO, name, repo_type='dataset', local_dir=TMP,
            )
            return path
        except Exception as e:
            msg = str(e)
            print(f'  download fail shard {i} attempt {attempt}: {msg[:160]}', flush=True)
            if '429' in msg or 'Rate' in msg:
                time.sleep(wait)
                wait = min(wait * 1.7, 180.0)
            else:
                time.sleep(min(wait, 30.0))
    raise RuntimeError(f'failed shard {i} after {retries} retries')


def process_shard(path, name2i, state):
    """Accumulate capped mean embeddings for matching scientific names."""
    pf = pq.ParquetFile(path)
    kept = 0
    seen = 0
    # Stop accepting more for classes already at CAP
    full = set(int(i) for i in np.where(state['cnt'] >= CAP)[0])
    for batch in pf.iter_batches(columns=['scientific_name', 'emb'], batch_size=8192):
        names = batch.column('scientific_name').to_pylist()
        embs = batch.column('emb').to_pylist()
        for nm, emb in zip(names, embs):
            seen += 1
            if nm is None:
                continue
            idx = name2i.get(nm) or name2i.get(nm.lower() if isinstance(nm, str) else None)
            if idx is None:
                continue
            if idx in full:
                continue
            if state['cnt'][idx] >= CAP:
                full.add(idx)
                continue
            v = np.asarray(emb, dtype=np.float32)
            if v.shape[0] != DIM:
                continue
            state['sum'][idx] += v.astype(np.float64)
            state['cnt'][idx] += 1
            kept += 1
            if state['cnt'][idx] >= CAP:
                full.add(idx)
    state['rows_kept'] += kept
    state['rows_seen'] += seen
    return kept, seen


def finalize(classes, state):
    ncls = len(classes)
    protos = torch.zeros(ncls, DIM)
    ncov = 0
    counts = {}
    for i in range(ncls):
        c = int(state['cnt'][i])
        if c <= 0:
            continue
        p = torch.from_numpy((state['sum'][i] / c).astype(np.float32))
        protos[i] = F.normalize(p, dim=0)
        ncov += 1
        counts[classes[i]] = c
    out = {
        'protos': protos,
        'classes': classes,
        'dim': DIM,
        'cap': CAP,
        'ncov': ncov,
        'counts': counts,
        'source': REPO,
        'rows_kept': int(state['rows_kept']),
        'rows_seen': int(state['rows_seen']),
    }
    torch.save(out, FINAL)
    meta = {k: out[k] for k in ('dim', 'cap', 'ncov', 'source', 'rows_kept', 'rows_seen')}
    meta['n_classes'] = ncls
    json.dump(meta, open(os.path.join(OUT, 'tol_protos_bioclip2_meta.json'), 'w'), indent=2)
    print(f'wrote {FINAL} ncov={ncov}/{ncls}', flush=True)
    return out


def main():
    classes, name2i = load_names()
    print(f'classes={len(classes)} name_keys={len(name2i)} cap={CAP}', flush=True)
    state = load_ckpt(len(classes))
    print(
        f'resume next_shard={state["next_shard"]} kept={state["rows_kept"]} '
        f'cov={(state["cnt"] > 0).sum()}',
        flush=True,
    )
    t0 = time.time()
    for i in range(state['next_shard'], NSHARD):
        t1 = time.time()
        path = download_shard(i)
        kept, seen = process_shard(path, name2i, state)
        # delete local shard copy to free disk
        try:
            os.remove(path)
        except OSError:
            pass
        # also clear nested path if hf put it under local_dir/repo structure
        nested = os.path.join(TMP, 'bioclip-2_float16', os.path.basename(path))
        if os.path.exists(nested):
            try:
                os.remove(nested)
            except OSError:
                pass
        state['next_shard'] = i + 1
        if (i + 1) % 5 == 0 or i == 0 or i + 1 == NSHARD:
            save_ckpt(state)
            cov = int((state['cnt'] > 0).sum())
            print(
                f'shard {i + 1}/{NSHARD} kept+={kept} seen={seen} '
                f'total_kept={state["rows_kept"]} cov={cov} '
                f'shard_s={time.time() - t1:.1f} elapsed_m={(time.time() - t0) / 60:.1f}',
                flush=True,
            )
        else:
            print(
                f'shard {i + 1}/{NSHARD} kept+={kept} cov={(state["cnt"] > 0).sum()} '
                f'{time.time() - t1:.1f}s',
                flush=True,
            )
            if (i + 1) % 1 == 0:
                save_ckpt(state)
    save_ckpt(state)
    finalize(classes, state)


if __name__ == '__main__':
    main()
