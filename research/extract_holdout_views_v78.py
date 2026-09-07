"""Extract the 7 v56 crop views for every holdout row (seen-val + pseudo-novel).

The cached emb_holdout_ctft_cropviews_v2.pt covers only the 2,318 pseudo-novel rows, so the
gate could never see a view-disagreement feature. This extracts the same 7 keys v56 uses
(center, squash, ostrip0-4) for all ~14,184 holdout rows with the SAME code path as the eval
side (crop_v2_views.overlap_strips + crop_views_proxy.center_tf/squash_tf, ctft_shift @ 0.4),
so v78's view features are identical in definition on holdout and eval.

  conda activate onet && python research/extract_holdout_views_v78.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from crop_v2_views import overlap_strips
from crop_views_proxy import (
    DEV, IMG_ROOT, center_tf, encode_pil_batch, index_images, load_ctft, squash_tf, to_tensor,
)
from learned_gate_v77 import holdout_split, load_train_embs

VIEW_KEYS = ['center', 'squash'] + [f'ostrip{i}' for i in range(5)]
OUT_PATH = os.path.join(OUT, 'emb_holdout_all_ctft_views7.pt')


def views7(img):
    v = {'center': center_tf(img), 'squash': squash_tf(img)}
    crops = overlap_strips(img, n=5)
    while len(crops) < 5:
        crops.append(crops[-1])
    for i, c in enumerate(crops[:5]):
        v[f'ostrip{i}'] = to_tensor(c)
    return v


def main():
    if os.path.exists(OUT_PATH):
        print(f'exists: {OUT_PATH}', flush=True)
        return
    print(f'DEV={DEV}', flush=True)
    train, tb, b2f, b2l = load_train_embs()
    sp = holdout_split(train, tb, b2f, b2l)
    files = sp['val_files']
    n = len(files)
    print(f'holdout rows: {n} (seen={sp["n_seen"]} pseudo={n - sp["n_seen"]})', flush=True)

    # cross-check the split against the cached pseudo-only artifact
    old = os.path.join(OUT, 'emb_holdout_ctft_cropviews_v2.pt')
    if os.path.exists(old):
        cached = set(torch.load(old, weights_only=False)['files'])
        mine = set(files[sp['n_seen']:])
        print(f'pseudo set vs cached v2: |mine|={len(mine)} |cached|={len(cached)} '
              f'overlap={len(mine & cached)}', flush=True)

    # ctftshift center embeddings as the fallback for unreadable images
    idx, feats, _ = train['ctftshift']
    model = load_ctft()
    imgidx = index_images(IMG_ROOT)
    packs = {k: [None] * n for k in VIEW_KEYS}
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

    for i, fn in enumerate(files):
        p = imgidx.get(fn)
        try:
            if p is None:
                raise FileNotFoundError(fn)
            vs = views7(Image.open(p).convert('RGB'))
        except Exception:
            miss += 1
            for k in VIEW_KEYS:
                packs[k][i] = feats[idx[fn]]
            continue
        for k in VIEW_KEYS:
            buf_t.append(vs[k])
            buf_k.append(k)
            buf_i.append(i)
        if len(buf_t) >= 84:
            flush()
        if i % 1000 == 0:
            print(f'  {i}/{n} miss={miss} [{time.time() - t0:.0f}s]', flush=True)
    flush()

    tensors = {k: F.normalize(torch.stack(rows).float(), dim=-1) for k, rows in packs.items()}
    torch.save({'files': files, 'views': tensors, 'view_keys': VIEW_KEYS}, OUT_PATH)
    print(f'wrote {OUT_PATH} n={n} miss={miss} [{time.time() - t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
