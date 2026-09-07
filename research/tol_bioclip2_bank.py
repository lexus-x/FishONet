"""Stream ToL BioCLIP-2 shards → per-class photo bank (up to CAP embs) for maxpool.

Only processes shards until MAX_SHARD (fish range ~0-180 from prior run).

  conda activate onet
  MAX_SHARD=180 python research/tol_bioclip2_bank.py 2>&1 | tee outputs/tol_bioclip2_bank.log
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
TMP = os.path.join(OUT, 'tol_shard_tmp')
CKPT = os.path.join(OUT, 'tol_bioclip2_bank_accum.pt')
FINAL = os.path.join(OUT, 'tol_photo_bank_bioclip2.pt')
REPO = 'imageomics/TreeOfLife-200M-Embeddings'
NSHARD = 666
MAX_SHARD = int(os.environ.get('MAX_SHARD', '180'))
CAP = int(os.environ.get('TOL_BANK_CAP', '16'))
DIM = 768


def load_names():
    classes = list(pickle.load(open(os.path.join(ROOT, 'data/dl/all_classes.pkl'), 'rb')))
    name2i = {}
    for i, c in enumerate(classes):
        name2i[c] = i
        name2i[c.replace('_', ' ')] = i
        name2i[c.lower()] = i
        name2i[c.replace('_', ' ').lower()] = i
    return classes, name2i


def load_ckpt(ncls):
    if os.path.exists(CKPT):
        return torch.load(CKPT, weights_only=False)
    return {
        'bank': {},  # ci -> list of np float16 [DIM]
        'next_shard': 0,
        'rows_kept': 0,
    }


def save_ckpt(state):
    torch.save(state, CKPT + '.tmp')
    os.replace(CKPT + '.tmp', CKPT)


def download_shard(i, retries=8):
    name = f'bioclip-2_float16/train-{i:05d}-of-{NSHARD:05d}.parquet'
    os.makedirs(TMP, exist_ok=True)
    wait = 5.0
    for attempt in range(retries):
        try:
            return hf_hub_download(REPO, name, repo_type='dataset', local_dir=TMP)
        except Exception as e:
            msg = str(e)
            print(f'  dl fail {i} try {attempt}: {msg[:140]}', flush=True)
            time.sleep(wait)
            wait = min(wait * 1.7, 180.0)
    raise RuntimeError(f'shard {i} failed')


def process(path, name2i, state):
    bank = state['bank']
    kept = 0
    pf = pq.ParquetFile(path)
    full = {ci for ci, photos in bank.items() if len(photos) >= CAP}
    for batch in pf.iter_batches(columns=['scientific_name', 'emb'], batch_size=8192):
        names = batch.column('scientific_name').to_pylist()
        embs = batch.column('emb').to_pylist()
        for nm, emb in zip(names, embs):
            if nm is None:
                continue
            idx = name2i.get(nm) or name2i.get(nm.lower() if isinstance(nm, str) else None)
            if idx is None or idx in full:
                continue
            lst = bank.setdefault(idx, [])
            if len(lst) >= CAP:
                full.add(idx)
                continue
            v = np.asarray(emb, dtype=np.float16)
            if v.shape[0] != DIM:
                continue
            lst.append(v)
            kept += 1
            if len(lst) >= CAP:
                full.add(idx)
    state['rows_kept'] += kept
    return kept


def finalize(classes, state):
    out_bank = {}
    for ci, photos in state['bank'].items():
        if not photos:
            continue
        t = torch.from_numpy(np.stack(photos).astype(np.float32))
        out_bank[int(ci)] = F.normalize(t, dim=-1)
    torch.save({
        'bank': out_bank,
        'cap': CAP,
        'n_classes': len(out_bank),
        'rows_kept': state['rows_kept'],
        'source': REPO,
        'enc': 'bioclip-2',
    }, FINAL)
    print(f'wrote {FINAL} classes={len(out_bank)} kept={state["rows_kept"]}', flush=True)


def main():
    classes, name2i = load_names()
    state = load_ckpt(len(classes))
    print(f'resume shard={state["next_shard"]} classes={len(state["bank"])} kept={state["rows_kept"]} '
          f'MAX_SHARD={MAX_SHARD} CAP={CAP}', flush=True)
    t0 = time.time()
    for i in range(state['next_shard'], min(MAX_SHARD, NSHARD)):
        t1 = time.time()
        path = download_shard(i)
        kept = process(path, name2i, state)
        for p in (path, os.path.join(TMP, 'bioclip-2_float16', os.path.basename(path))):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        state['next_shard'] = i + 1
        save_ckpt(state)
        print(f'shard {i+1}/{MAX_SHARD} kept+={kept} classes={len(state["bank"])} '
              f'{time.time()-t1:.1f}s elapsed_m={(time.time()-t0)/60:.1f}', flush=True)
        # early stop if nearly all classes at CAP
        nfull = sum(1 for photos in state['bank'].values() if len(photos) >= CAP)
        if nfull >= 10000 and i > 100:
            print(f'early stop: {nfull} classes at CAP', flush=True)
            break
    finalize(classes, state)


if __name__ == '__main__':
    main()
