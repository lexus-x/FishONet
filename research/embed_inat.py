"""Embed downloaded iNaturalist images into per-class visual prototypes.

Encoders:
  --enc h          frozen BioCLIP-2.5 ViT-H/14 (hf-hub:imageomics/bioclip-2.5-vith14)
  --enc b2         frozen BioCLIP-2 ViT-L/14 (hf-hub:imageomics/bioclip-2)
  --enc ctft --ckpt outputs/ctft_shift.pt [--squash_tta 1]   LoRA ctft encoder

Prototype = L2-normalized mean of per-image L2-normalized embeddings, indexed over the
full class list (outputs are aligned to research.common.FishData.classes order).

  python research/embed_inat.py --enc h --files outputs/inat_image_files.json \
      --out outputs/inat_protos_h.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, OUT  # noqa: E402

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'
MODEL_B2 = 'hf-hub:imageomics/bioclip-2'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class LoRALinear(nn.Module):
    def __init__(s, base, r=16, alpha=32):
        super().__init__(); s.base = base
        for p in s.base.parameters():
            p.requires_grad_(False)
        s.r = r; s.scaling = alpha / max(r, 1)
        s.A = nn.Parameter(torch.zeros(r, base.in_features))
        s.B = nn.Parameter(torch.zeros(base.out_features, r))
        s.drop = nn.Dropout(0.0); s.lora_scale = 1.0

    def forward(s, x):
        return s.base(x) + (s.drop(x) @ s.A.t() @ s.B.t()) * s.scaling * s.lora_scale


def inject_lora(model, r, alpha, top_k=0):
    blocks = model.visual.transformer.resblocks
    rng = range(len(blocks)) if top_k <= 0 else range(len(blocks) - top_k, len(blocks))
    for i in rng:
        blk = blocks[i]
        blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, r, alpha)
        blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, r, alpha)


def set_scale(model, s):
    for blk in model.visual.transformer.resblocks:
        for m in (blk.mlp.c_fc, blk.mlp.c_proj):
            if isinstance(m, LoRALinear):
                m.lora_scale = s


class DS(Dataset):
    def __init__(s, items, pp, squash):
        s.items = items; s.pp = pp; s.squash = squash
        s.sq = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                          T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(s):
        return len(s.items)

    def __getitem__(s, i):
        ci, path = s.items[i]
        try:
            img = Image.open(path).convert('RGB')
            if s.squash:
                return ci, s.pp(img), s.sq(img)
            return ci, s.pp(img), s.pp(img)
        except Exception:
            z = torch.zeros(3, 224, 224)
            return -1, z, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', choices=['h', 'ctft', 'b2', 'b2lora'], default='h')
    ap.add_argument('--ckpt', default='outputs/ctft_shift.pt')
    ap.add_argument('--scale', type=float, default=0.4)
    ap.add_argument('--squash_tta', type=int, default=0)
    ap.add_argument('--files', default=os.path.join(OUT, 'inat_image_files.json'))
    ap.add_argument('--out', default=os.path.join(OUT, 'inat_protos_h.pt'))
    ap.add_argument('--incremental', action='store_true',
                    help='Load --out if present; only update classes in --files')
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    D = FishData()
    meta = json.load(open(a.files))
    items = []
    for c, paths in meta.items():
        if c not in D.ci:
            continue
        for p in paths:
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                items.append((D.ci[c], p))
    print(f'{len(items)} images over {len(set(i for i, _ in items))} classes', flush=True)

    clip_model = MODEL_B2 if a.enc in ('b2', 'b2lora') else MODEL
    model, _, pp = open_clip.create_model_and_transforms(clip_model)
    if a.enc in ('ctft', 'b2lora'):
        ck = torch.load(a.ckpt, weights_only=False)
        inject_lora(model, ck['rank'], ck['alpha'], ck['top_k'])
        own = dict(model.named_parameters())
        for n, v in ck['state'].items():
            own[n].data.copy_(v)
        model = model.to(DEV).eval()
        set_scale(model, a.scale)
        print(f'ctft ckpt {a.ckpt} rank={ck["rank"]} squash_tta={a.squash_tta}', flush=True)
    else:
        model = model.to(DEV).eval()

    dim = model.visual.output_dim if hasattr(model.visual, 'output_dim') else 1024
    C = len(D.classes)
    acc = torch.zeros(C, dim)
    cnt = torch.zeros(C)
    if a.incremental and os.path.exists(a.out):
        prev = torch.load(a.out, weights_only=False)
        acc = prev['protos'].float().clone()
        cnt = prev['cnt'].float().clone()
        print(f'incremental base: {int((cnt>0).sum())} classes from {a.out}', flush=True)
    dl = DataLoader(DS(items, pp, a.squash_tta), batch_size=a.bs, num_workers=a.workers,
                    pin_memory=True)
    t0 = time.time()
    with torch.no_grad():
        for bi, (cis, x, xsq) in enumerate(dl):
            x = x.to(DEV); xsq = xsq.to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = F.normalize(model.encode_image(x).float(), dim=-1)
                f = f + F.normalize(model.encode_image(torch.flip(x, dims=[-1])).float(), dim=-1)
                if a.squash_tta:
                    f = f + F.normalize(model.encode_image(xsq).float(), dim=-1)
                    f = f + F.normalize(model.encode_image(torch.flip(xsq, dims=[-1])).float(), dim=-1)
                f = F.normalize(f, dim=-1).cpu()
            for j, ci in enumerate(cis.tolist()):
                if ci < 0:
                    continue
                acc[ci] += f[j]
                cnt[ci] += 1
            if bi % 20 == 0:
                print(f'  batch {bi} [{time.time()-t0:.0f}s]', flush=True)

    protos = F.normalize(acc, dim=-1)
    covered = [D.classes[i] for i in range(C) if cnt[i] > 0]
    torch.save({'protos': protos, 'cnt': cnt, 'covered': covered,
                'enc': a.enc, 'ckpt': a.ckpt if a.enc in ('ctft', 'b2lora') else None,
                'squash_tta': a.squash_tta, 'model': clip_model, 'classes': D.classes}, a.out)
    print(f'wrote {a.out}: covered {len(covered)} classes, {int(cnt.sum())} images', flush=True)


if __name__ == '__main__':
    main()
