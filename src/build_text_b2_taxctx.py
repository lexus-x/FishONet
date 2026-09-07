"""Taxonomy-context text embeddings for BioCLIP-2 (768-d), unseen-route alignment.

Disclosed: competition class names + outputs/taxonomy_full.json + outputs/common_names.json.

  conda activate onet
  python src/build_text_b2_taxctx.py
"""
import argparse
import json
import pickle

import open_clip
import torch
import torch.nn.functional as F

MODEL = 'hf-hub:imageomics/bioclip-2'
DEV = 'cuda'
D = 'data/dl'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs/text_emb_b2_taxctx.pt')
    a = ap.parse_args()

    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    tax = json.load(open('outputs/taxonomy_full.json'))
    cn = json.load(open('outputs/common_names.json'))

    model, _, _ = open_clip.create_model_and_transforms(MODEL)
    tok = open_clip.get_tokenizer(MODEL)
    model = model.to(DEV).eval()

    txts = []
    for c in classes:
        t = tax.get(c) or {}
        fam = t.get('family')
        com = cn.get(c)
        s = f'a photo of {c}'
        if com:
            s += f', commonly known as {com}'
        if fam:
            s += f', a fish of the family {fam}'
        s += '.'
        txts.append(s)

    print(f'encoding {len(txts)} taxctx prompts with {MODEL}', flush=True)

    @torch.no_grad()
    def enc(texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            t = tok(texts[i:i + bs]).to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = model.encode_text(t).float()
            out.append(F.normalize(f, dim=-1).cpu())
            if i % 5120 == 0:
                print(f'  {i}/{len(texts)}', flush=True)
        return torch.cat(out)

    emb = enc(txts)
    torch.save({'classes': classes, 'emb_taxctx': emb, 'emb_taxon': emb}, a.out)
    print('saved', a.out, emb.shape, flush=True)


if __name__ == '__main__':
    main()
