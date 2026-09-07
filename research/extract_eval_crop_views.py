"""Extract eval (test+unseen) ctftshift crop views on AWS GPU.

Views (no flip — holdout said max(all_no_flip) won):
  center, squash, letterbox, strip0, strip1, strip2

  conda activate onet && python research/extract_eval_crop_views.py
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
from crop_views_proxy import (
    DEV, IMG_ROOT, index_images, load_ctft, encode_pil_batch, views_for,
)

SPLITS = ('test', 'unseen')
VIEW_KEYS = ['center', 'squash', 'letterbox', 'strip0', 'strip1', 'strip2']


def main():
    print(f'DEV={DEV}', flush=True)
    model = load_ctft()
    imgidx = index_images(IMG_ROOT)
    for split in SPLITS:
        out_path = os.path.join(OUT, f'emb_{split}_ctft_cropviews.pt')
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
                vs = views_for(img)
            except Exception:
                miss += 1
                continue
            if 'strip1' not in vs:
                vs['strip1'] = vs['strip0']
            if 'strip2' not in vs:
                vs['strip2'] = vs.get('strip1', vs['strip0'])
            kept.append(fn)
            for k in VIEW_KEYS:
                buf_t.append(vs[k])
                buf_k.append(k)
                buf_i.append(qi)
            qi += 1
            if len(buf_t) >= 72:
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
