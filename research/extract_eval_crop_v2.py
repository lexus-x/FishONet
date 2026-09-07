"""Extract extra eval crop views (tight bbox + 5 overlapping strips) if holdout v2 wins.

  conda activate onet && python research/extract_eval_crop_v2.py
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
from crop_chase53_proxy import ALL_KEYS, pad_views
from crop_v2_views import views_for_v2
from crop_views_proxy import DEV, IMG_ROOT, encode_pil_batch, index_images, load_ctft

SPLITS = ('test', 'unseen')


def main():
    print(f'DEV={DEV}', flush=True)
    model = load_ctft()
    imgidx = index_images(IMG_ROOT)
    for split in SPLITS:
        out_path = os.path.join(OUT, f'emb_{split}_ctft_cropviews_v2.pt')
        if os.path.exists(out_path):
            print(f'skip {out_path}', flush=True)
            continue
        files = list(pickle.load(open(f'data/dl/splits/{split}.pkl', 'rb')))
        n = len(files)
        packs = {k: [None] * n for k in ALL_KEYS}
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
                vs = pad_views(views_for_v2(img))
            except Exception:
                miss += 1
                continue
            kept.append(fn)
            for k in ALL_KEYS:
                buf_t.append(vs[k])
                buf_k.append(k)
                buf_i.append(qi)
            qi += 1
            if len(buf_t) >= 64:
                flush()
            if i % 1000 == 0:
                print(f'  {split} {i}/{n} kept={len(kept)} miss={miss} [{time.time()-t0:.0f}s]',
                      flush=True)
        flush()
        tensors = {k: F.normalize(torch.stack(packs[k][:len(kept)]).float(), dim=-1)
                   for k in ALL_KEYS}
        torch.save({'files': kept, 'views': tensors, 'view_keys': ALL_KEYS}, out_path)
        print(f'wrote {out_path} n={len(kept)} miss={miss} [{time.time()-t0:.0f}s]', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
