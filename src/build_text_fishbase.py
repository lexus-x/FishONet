"""Embed reports/descriptions_fishbase.json (multi-bullet FishBase morphological descriptions,
external data per COMPETITION_RULES.md 3.2, disclosed) with the SAME text tower as the taxon
embeddings. Per-class: L2-norm each bullet's embedding, average, L2-norm again (classes with
multiple descriptions), matching outputs/text_emb_h_taxon.pt's model exactly for comparability.

  conda activate onet && python src/build_text_fishbase.py
"""
import json
import pickle

import open_clip
import torch
import torch.nn.functional as F

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'
DEV = 'cuda'
D = 'data/dl'


def main():
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    fb = json.load(open('reports/descriptions_fishbase.json'))
    cov = sum(1 for c in classes if c in fb)
    print(f'coverage: {cov}/{len(classes)} classes have >=1 FishBase description', flush=True)

    model, _, _ = open_clip.create_model_and_transforms(MODEL)
    tok = open_clip.get_tokenizer(MODEL)
    model = model.to(DEV).eval()

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

    all_bullets, owner = [], []
    for c in classes:
        for b in fb.get(c, []):
            all_bullets.append(b)
            owner.append(c)
    print(f'encoding {len(all_bullets)} individual description bullets ...', flush=True)
    E = enc(all_bullets)  # (Nbullets, 1024), already L2-normalized per-bullet

    D_out = 1024
    acc = torch.zeros(len(classes), D_out)
    cnt = torch.zeros(len(classes))
    ci = {c: i for i, c in enumerate(classes)}
    for e, c in zip(E, owner):
        acc[ci[c]] += e
        cnt[ci[c]] += 1
    emb = F.normalize(acc / cnt.clamp(min=1).unsqueeze(1), dim=-1)  # L2, avg, L2

    out = {'classes': classes, 'emb_fishbase': emb, 'coverage_mask': (cnt > 0), 'n_bullets': cnt}
    torch.save(out, 'outputs/text_emb_h_fishbase.pt')
    print(f'saved outputs/text_emb_h_fishbase.pt, coverage {int((cnt > 0).sum())}/{len(classes)}', flush=True)


if __name__ == '__main__':
    main()
