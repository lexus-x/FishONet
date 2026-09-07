"""TaxaBind embeddings of iNat research photos -> per-class prototypes (512-d)."""
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

MODEL = 'hf-hub:MVRL/taxabind-vit-b-16'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', default=os.path.join(OUT, 'inat_image_files.json'))
    ap.add_argument('--out', default=os.path.join(OUT, 'inat_protos_taxabind.pt'))
    ap.add_argument('--bs', type=int, default=128)
    a = ap.parse_args()

    D = FishData()
    meta = json.load(open(a.files))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, pre = open_clip.create_model_and_transforms(MODEL)
    model = model.to(dev).eval()

    acc = torch.zeros(len(D.classes), 512)
    cnt = torch.zeros(len(D.classes))
    items = []
    for c, paths in meta.items():
        if c not in D.ci:
            continue
        for p in paths:
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                items.append((D.ci[c], p))
    print(f'{len(items)} images', flush=True)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(items), a.bs):
            batch_c, batch_x = [], []
            for ci, p in items[i:i + a.bs]:
                try:
                    batch_x.append(pre(Image.open(p).convert('RGB')))
                    batch_c.append(ci)
                except Exception:
                    continue
            if not batch_x:
                continue
            x = torch.stack(batch_x).to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = F.normalize(model.encode_image(x).float(), dim=-1).cpu()
            for j, ci in enumerate(batch_c):
                acc[ci] += f[j]
                cnt[ci] += 1
            if i // a.bs % 20 == 0:
                print(f'  {i}/{len(items)} [{time.time()-t0:.0f}s]', flush=True)

    protos = torch.zeros_like(acc)
    for i in range(len(D.classes)):
        if cnt[i] > 0:
            protos[i] = F.normalize(acc[i] / cnt[i], dim=-1)
    covered = [D.classes[i] for i in range(len(D.classes)) if cnt[i] > 0]
    torch.save({'protos': protos, 'cnt': cnt, 'covered': covered, 'model': MODEL, 'classes': D.classes}, a.out)
    print(f'wrote {a.out}: {len(covered)} classes, {int(cnt.sum())} imgs', flush=True)


if __name__ == '__main__':
    main()
