"""Train-split image embeddings for an encoder (pseudo-unseen proxy queries).

  python research/extract_encoder_train.py --backend open_clip --model 'hf-hub:timm/ViT-SO400M-16-SigLIP2-384' \\
      --out outputs/emb_train_siglip2.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, OUT  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, 'data', 'dl', 'images')


def load_dino(name: str):
    model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
    model = model.to(DEV).eval()
    from torchvision import transforms as T

    pre = T.Compose([
        T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(518),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['open_clip', 'dinov2'], default='open_clip')
    ap.add_argument('--model', default='hf-hub:timm/ViT-SO400M-16-SigLIP2-384')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=64)
    a = ap.parse_args()

    D = FishData()
    # same train pool as other emb_train_* (intersection used in builders)
    import pickle
    from collections import defaultdict

    lab = json.load(open(os.path.join(ROOT, 'data', 'dl', 'label_train.json')))
    members = ['ctftshift', 'ftshift', 'fullft336shift', 'L', 'fullft336_v2']
    train_tags = ['emb_train_ctftshift', 'emb_train_ftshift', 'emb_train_fullft336shift', 'emb_train', 'emb_train_fullft336_v2']
    common = None
    for tag in train_tags:
        d = torch.load(os.path.join(OUT, f'{tag}.pt'), weights_only=False)
        s = set(d['files'])
        common = s if common is None else (common & s)
    common = sorted(common)
    print(f'train files {len(common)}', flush=True)

    if a.backend == 'dinov2':
        model, pre = load_dino(a.model)
        enc = lambda x: model(x).float()
    else:
        model, _, pre = open_clip.create_model_and_transforms(a.model)
        model = model.to(DEV).eval()
        enc = lambda x: model.encode_image(x).float()

    t0 = time.time()
    feats = []
    with torch.no_grad():
        for i in range(0, len(common), a.bs):
            batch = common[i:i + a.bs]
            imgs = []
            for f in batch:
                imgs.append(pre(Image.open(os.path.join(IMG, f)).convert('RGB')))
            x = torch.stack(imgs).to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = F.normalize(enc(x), dim=-1).cpu()
            feats.append(f)
            if i % (a.bs * 40) == 0:
                print(f'  {i}/{len(common)} [{time.time()-t0:.0f}s]', flush=True)
    feats = torch.cat(feats)
    torch.save({'files': common, 'feats': feats, 'backend': a.backend, 'model': a.model}, a.out)
    print(f'wrote {a.out} {feats.shape} in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
