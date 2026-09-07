"""Extract TaxaBind frozen embeddings for unseen-route fusion (proxy +1.86pt).

  python research/extract_taxabind.py
Writes:
  outputs/emb_{test,unseen}_taxabind.pt
  outputs/text_emb_taxabind_taxctx.pt  (key emb_taxctx, class order = all_classes.pkl)
"""
import json
import os
import pickle
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


def prompts_taxctx(classes):
    tax = json.load(open(os.path.join(OUT, 'taxonomy_full.json')))
    cn = json.load(open(os.path.join(OUT, 'common_names.json')))
    out = []
    for c in classes:
        t = tax.get(c) or {}
        s = f'a photo of {c}'
        if cn.get(c):
            s += f", commonly known as {cn[c]}"
        if t.get('family'):
            s += f", a fish of the family {t['family']}"
        out.append(s + '.')
    return out


def main():
    dev = 'cuda'
    model, _, pre = open_clip.create_model_and_transforms(MODEL)
    tok = open_clip.get_tokenizer(MODEL)
    model = model.to(dev).eval()
    classes = list(pickle.load(open(os.path.join(DATA, 'all_classes.pkl'), 'rb')))
    texts = prompts_taxctx(classes)
    t0 = time.time()

    @torch.no_grad()
    def enc_text(bs=256):
        out = []
        for i in range(0, len(texts), bs):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = model.encode_text(tok(texts[i:i + bs]).to(dev)).float()
            out.append(F.normalize(f, dim=-1).cpu())
            if i % (bs * 20) == 0:
                print(f'  text {i}/{len(texts)} [{time.time()-t0:.0f}s]', flush=True)
        return torch.cat(out)

    @torch.no_grad()
    def enc_split(files, bs=64):
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

    print('encoding text...', flush=True)
    T = enc_text()
    tp = os.path.join(OUT, 'text_emb_taxabind_taxctx.pt')
    torch.save({'emb_taxctx': T, 'classes': classes, 'texts': texts, 'model': MODEL}, tp)
    print('wrote', tp, T.shape, flush=True)

    for split in ['test', 'unseen']:
        files = list(pickle.load(open(os.path.join(DATA, 'splits', f'{split}.pkl'), 'rb')))
        print(f'encoding {split} n={len(files)}...', flush=True)
        feats = enc_split(files)
        p = os.path.join(OUT, f'emb_{split}_taxabind.pt')
        torch.save({'files': files, 'feats': feats, 'model': MODEL}, p)
        print('wrote', p, feats.shape, flush=True)

    print(f'done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
