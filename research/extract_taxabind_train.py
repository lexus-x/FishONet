"""Extract TaxaBind frozen embeddings for TRAIN images (needed for proxy b sweeps).

Enables including TaxaBind in the pseudo-unseen fusion weight search
(research/common.py proxy), which was previously blocked by having only
test/unseen TaxaBind embeddings.

  python research/extract_taxabind_train.py
Writes:
  outputs/emb_train_taxabind.pt
Disclose: MVRL/taxabind-vit-b-16 (general biology dual-encoder; not fish-specific).
"""
import json
import os
import time

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
DATA = os.path.join(ROOT, 'data', 'dl')
IMG = os.path.join(DATA, 'images')
MODEL = 'hf-hub:MVRL/taxabind-vit-b-16'


def main():
    dev = 'cuda'
    model, _, pre = open_clip.create_model_and_transforms(MODEL)
    model = model.to(dev).eval()
    lab = json.load(open(os.path.join(DATA, 'label_train.json')))
    files = [fn for fn in lab if os.path.exists(os.path.join(IMG, fn))]
    print(f'train files: {len(files)}', flush=True)
    t0 = time.time()

    @torch.no_grad()
    def enc_split(files, bs=128):
        out = []
        for i in range(0, len(files), bs):
            batch = []
            for f in files[i:i + bs]:
                batch.append(pre(Image.open(os.path.join(IMG, f)).convert('RGB')))
            x = torch.stack(batch).to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = model.encode_image(x).float()
            out.append(F.normalize(f, dim=-1).cpu())
            if i % (bs * 40) == 0:
                print(f'  img {i}/{len(files)} [{time.time()-t0:.0f}s]', flush=True)
        return torch.cat(out)

    feats = enc_split(files)
    p = os.path.join(OUT, 'emb_train_taxabind.pt')
    torch.save({'files': files, 'feats': feats, 'model': MODEL}, p)
    print('wrote', p, feats.shape, f'[{time.time()-t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
