"""Extract 5 overlapping long-axis strips for eval (holdout best crop set).

Holdout: max(center, squash, ostrip0-4) + 3*336 bank = +1.424 vs v50-like.
center/squash already in emb_*_ctft_cropviews.pt — this file is strips only.

  conda activate onet && python research/extract_eval_overlap.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from crop_v2_views import overlap_strips
from crop_views_proxy import DEV, IMG_ROOT, encode_pil_batch, index_images, load_ctft, to_tensor

SPLITS = ('test', 'unseen')
N_STRIPS = 5
VIEW_KEYS = [f'ostrip{i}' for i in range(N_STRIPS)]


def main():
    print(f'DEV={DEV}', flush=True)
    model = load_ctft()
    imgidx = index_images(IMG_ROOT)
    for split in SPLITS:
        out_path = os.path.join(OUT, f'emb_{split}_ctft_ostrip.pt')
        if os.path.exists(out_path):
            print(f'skip {out_path}', flush=True)
            continue
        files = list(pickle.load(open(f'data/dl/splits/{split}.pkl', 'rb')))
        n = len(files)
        packs = {k: [None] * n for k in VIEW_KEYS}
        kept = []
        buf_t, buf_k, buf_i = [], [], []
        miss = 0
        t0 = time.time()

        def flush():
            if not buf_t:
                return
            f = encode_pil_batch(model, buf_t)
            for row, key, qi in zip(f, buf_k, buf_i):
                packs[key][qi] = row
            buf_t.clear()
            buf_k.clear()
            buf_i.clear()

        print(f'=== {split} n={n} ===', flush=True)
        qi = 0
        for i, fn in enumerate(files):
            p = imgidx.get(fn)
            if p is None:
                miss += 1
                continue
            try:
                img = Image.open(p).convert('RGB')
                crops = overlap_strips(img, n=N_STRIPS)
            except Exception:
                miss += 1
                continue
            while len(crops) < N_STRIPS:
                crops.append(crops[-1])
            kept.append(fn)
            for k, crop in zip(VIEW_KEYS, crops):
                buf_t.append(to_tensor(crop))
                buf_k.append(k)
                buf_i.append(qi)
            qi += 1
            if len(buf_t) >= 80:
                flush()
            if i % 1000 == 0:
                print(f'  {split} {i}/{n} kept={len(kept)} miss={miss} [{time.time()-t0:.0f}s]',
                      flush=True)
        flush()
        tensors = {k: F.normalize(torch.stack(packs[k][:len(kept)]).float(), dim=-1)
                   for k in VIEW_KEYS}
        torch.save({'files': kept, 'views': tensors, 'view_keys': VIEW_KEYS}, out_path)
        print(f'wrote {out_path} n={len(kept)} miss={miss} [{time.time()-t0:.0f}s]', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
