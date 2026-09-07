"""Embed iNat photos -> class-mean prototypes for arbitrary image encoders (disclosed external data).

OpenCLIP:  --backend open_clip --model 'hf-hub:timm/ViT-SO400M-16-SigLIP2-384'
DINOv2:    --backend dinov2 --model dinov2_vitl14

  python research/embed_inat_encoder.py --backend open_clip --model 'hf-hub:timm/ViT-SO400M-16-SigLIP2-384' \\
      --out outputs/inat_protos_siglip2.pt
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
    dim = model.embed_dim
    return model, pre, dim, 'dinov2'


def encode_batch(model, backend, pre, paths):
    imgs = [pre(Image.open(p).convert('RGB')) for p in paths]
    x = torch.stack(imgs).to(DEV)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        if backend == 'dinov2':
            f = model(x).float()
        else:
            f = model.encode_image(x).float()
    return F.normalize(f, dim=-1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['open_clip', 'dinov2'], default='open_clip')
    ap.add_argument('--model', default='hf-hub:timm/ViT-SO400M-16-SigLIP2-384')
    ap.add_argument('--files', default=os.path.join(OUT, 'inat_image_files.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=64)
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
    print(f'{len(items)} images', flush=True)

    if a.backend == 'dinov2':
        model, pre, dim, tag = load_dino(a.model)
        model_id = a.model
    else:
        model, _, pre = open_clip.create_model_and_transforms(a.model)
        model = model.to(DEV).eval()
        tag = 'open_clip'
        model_id = a.model
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            dim = int(model.encode_image(pre(Image.new('RGB', (384, 384))).unsqueeze(0).to(DEV)).shape[-1])

    acc = torch.zeros(len(D.classes), dim)
    cnt = torch.zeros(len(D.classes))
    t0 = time.time()
    for i in range(0, len(items), a.bs):
        batch = items[i:i + a.bs]
        paths = [p for _, p in batch]
        try:
            f = encode_batch(model, a.backend, pre, paths)
        except Exception as e:
            print(f'skip batch {i}: {e}', flush=True)
            continue
        for j, (ci, _) in enumerate(batch):
            acc[ci] += f[j]
            cnt[ci] += 1
        if i // a.bs % 40 == 0:
            print(f'  {i}/{len(items)} [{time.time()-t0:.0f}s]', flush=True)

    protos = torch.zeros_like(acc)
    for i in range(len(D.classes)):
        if cnt[i] > 0:
            protos[i] = F.normalize(acc[i] / cnt[i], dim=-1)
    covered = [D.classes[i] for i in range(len(D.classes)) if cnt[i] > 0]
    torch.save({
        'protos': protos, 'cnt': cnt, 'covered': covered,
        'backend': tag, 'model': model_id, 'classes': D.classes,
    }, a.out)
    print(f'wrote {a.out}: {len(covered)} classes dim={dim}', flush=True)


if __name__ == '__main__':
    main()
