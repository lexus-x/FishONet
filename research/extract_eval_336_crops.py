"""Eval + holdout crop views at 336 with fullft336shift.

v56 won with max(center, squash, 5 overlapping strips) on ctft. This applies
the same recipe to the 336 encoder that moved the iNat bank (+0.777 holdout).

  conda activate onet && python -u research/extract_eval_336_crops.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'
CKPT = os.path.join(OUT, 'fullft336_shift.pt')
RES = 336
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
N_STRIPS = 5
VIEW_KEYS = ['center', 'squash'] + [f'ostrip{i}' for i in range(N_STRIPS)]
FLUSH = 28
IMG_ROOT = 'data/dl/images'
SPLITS = ('test', 'unseen')
HOLD_SRC = os.path.join(OUT, 'emb_holdout_ctft_cropviews_v2.pt')

to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
center_tf = T.Compose([
    T.Resize(RES, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(RES),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
squash_tf = T.Compose([
    T.Resize((RES, RES), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx[f] = os.path.join(dp, f)
    return idx


def overlap_strips(img, n=N_STRIPS, res=RES):
    w, h = img.size
    n = max(3, int(n))
    crops = []
    if w >= h:
        side = h
        span = max(0, w - side)
        if span == 0:
            return [img.resize((res, res), Image.BICUBIC)]
        xs = [int(round(i * span / (n - 1))) for i in range(n)]
        for x in dict.fromkeys(xs):
            crops.append(img.crop((x, 0, x + side, side)))
    else:
        side = w
        span = max(0, h - side)
        if span == 0:
            return [img.resize((res, res), Image.BICUBIC)]
        ys = [int(round(i * span / (n - 1))) for i in range(n)]
        for y in dict.fromkeys(ys):
            crops.append(img.crop((0, y, side, y + side)))
    return [c.resize((res, res), Image.BICUBIC) for c in crops]


def views_for(img):
    v = {'center': center_tf(img), 'squash': squash_tf(img)}
    crops = overlap_strips(img)
    while len(crops) < N_STRIPS:
        crops.append(crops[-1])
    for i, crop in enumerate(crops[:N_STRIPS]):
        v[f'ostrip{i}'] = to_tensor(crop)
    return v


def load_336():
    print(f'loading {CKPT}', flush=True)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    print(f'fullft336_shift res={ck.get("res")} val_acc={ck.get("val_acc")}', flush=True)
    model, _, _ = open_clip.create_model_and_transforms(MODEL, force_image_size=RES)
    model.load_state_dict(ck['model'] if 'model' in ck else ck)
    model = model.to(DEV).eval()
    return model


@torch.no_grad()
def encode_batch(model, tensors):
    x = torch.stack(tensors).to(DEV, non_blocking=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        f = model.encode_image(x).float()
    return F.normalize(f, dim=-1).cpu()


def extract_filelist(model, imgidx, files, out_path, tag):
    if os.path.exists(out_path):
        print(f'skip {out_path}', flush=True)
        return
    n = len(files)
    packs = {k: [None] * n for k in VIEW_KEYS}
    kept = []
    buf_t, buf_k, buf_i = [], [], []
    miss = 0
    t0 = time.time()

    def flush():
        if not buf_t:
            return
        f = encode_batch(model, buf_t)
        for row, key, qi in zip(f, buf_k, buf_i):
            packs[key][qi] = row
        buf_t.clear()
        buf_k.clear()
        buf_i.clear()

    print(f'=== {tag} n={n} ===', flush=True)
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
        kept.append(fn)
        for k in VIEW_KEYS:
            buf_t.append(vs[k])
            buf_k.append(k)
            buf_i.append(qi)
        qi += 1
        if len(buf_t) >= FLUSH:
            flush()
        if i % 500 == 0:
            print(f'  {tag} {i}/{n} kept={len(kept)} miss={miss} [{time.time()-t0:.0f}s]',
                  flush=True)
    flush()
    tensors = {k: F.normalize(torch.stack(packs[k][:len(kept)]).float(), dim=-1)
               for k in VIEW_KEYS}
    torch.save({'files': kept, 'views': tensors, 'view_keys': VIEW_KEYS, 'res': RES}, out_path)
    print(f'wrote {out_path} n={len(kept)} miss={miss} [{time.time()-t0:.0f}s]', flush=True)


def main():
    print(f'DEV={DEV}', flush=True)
    model = load_336()
    imgidx = index_images(IMG_ROOT)
    print(f'indexed {len(imgidx)}', flush=True)
    for split in SPLITS:
        files = list(pickle.load(open(f'data/dl/splits/{split}.pkl', 'rb')))
        extract_filelist(model, imgidx, files, os.path.join(OUT, f'emb_{split}_fullft336_cropviews.pt'), split)
    if os.path.exists(HOLD_SRC):
        hfiles = torch.load(HOLD_SRC, weights_only=False)['files']
        extract_filelist(
            model, imgidx, hfiles,
            os.path.join(OUT, 'emb_holdout_fullft336_cropviews.pt'), 'holdout',
        )
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
