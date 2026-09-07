"""Quick frozen backbone proxy for Arboretum/BT-CLIP-O (general biology — ALLOWED).

Same protocol as research/unseen_backbone.py: taxctx prompts, DBNorm, fuse vs BioCLIP-H.
Kill if alone << 21.53 or fuse Δ < 0.5 on pseudo-unseen.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from common import DEV, OUT, DATA, dbnorm, load_emb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(DATA, 'dl')
MODEL = 'hf-hub:Arboretum/BT-CLIP-O'
TAG = 'arbor'


def taxctx(sp, common, fam):
    if common and fam:
        return f'a photo of {sp}, commonly known as {common}, a fish of the family {fam}.'
    if fam:
        return f'a photo of {sp}, a fish of the family {fam}.'
    return f'a photo of {sp}.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_imgs', type=int, default=0, help='0=all pseudo-unseen imgs')
    args = ap.parse_args()
    device = DEV
    torch.set_num_threads(8)

    classes = list(pickle.load(open(os.path.join(D, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(os.path.join(D, 'label_train.json')))
    # taxonomy helpers if present
    tax = {}
    for p in ['taxonomy_resolved.json', 'taxonomy.json']:
        fp = os.path.join(D, p)
        if os.path.exists(fp):
            tax = json.load(open(fp))
            break
    # descriptions may have common names
    desc = json.load(open(os.path.join(D, 'descriptions.json')))

    # image index
    imgidx = {}
    for dp, _, fs in os.walk(os.path.join(D, 'images')):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                imgidx[f] = os.path.join(dp, f)

    # hard-sim split from train emb
    tr = load_emb(os.path.join(OUT, 'emb_train_h.pt'))
    by = defaultdict(list)
    for fn in tr[2]:
        if fn in lab and lab[fn] in ci and fn in imgidx:
            by[lab[fn]].append(fn)
    seen = sorted(by.keys())
    order = sorted(seen, key=lambda c: len(by[c]))
    pseudo = order[: int(len(seen) * 0.2)]
    known = [c for c in seen if c not in set(pseudo)]
    cand = torch.tensor([i for i, c in enumerate(classes) if c not in set(known)])
    cand_pos = {int(x): j for j, x in enumerate(cand.tolist())}

    qfiles = []
    for c in pseudo:
        for fn in by[c]:
            qfiles.append((fn, cand_pos[ci[c]]))
    if args.max_imgs:
        qfiles = qfiles[: args.max_imgs]
    print(f'pseudo={len(pseudo)} queries={len(qfiles)} cand={len(cand)}', flush=True)

    print('loading', MODEL, flush=True)
    model, _, pp = open_clip.create_model_and_transforms(MODEL)
    tok = open_clip.get_tokenizer(MODEL)
    model = model.to(device).eval()

    # text embeddings taxctx for all classes (or cand only — need all for fuse baseline compare)
    # Build prompts
    prompts = []
    for sp in classes:
        info = desc.get(sp, {}) if isinstance(desc, dict) else {}
        if isinstance(info, str):
            common, fam = '', ''
        else:
            common = (info.get('common_name') or info.get('common') or '') if isinstance(info, dict) else ''
            fam = (info.get('family') or '') if isinstance(info, dict) else ''
        if sp in tax and isinstance(tax[sp], dict):
            fam = tax[sp].get('family') or fam
            common = tax[sp].get('common_name') or common
        prompts.append(taxctx(sp, common, fam))

    print('encoding text...', flush=True)
    embs = []
    with torch.no_grad():
        for i in range(0, len(prompts), 256):
            t = tok(prompts[i:i + 256]).to(device)
            e = model.encode_text(t)
            e = F.normalize(e.float(), dim=-1)
            embs.append(e.cpu())
    T = torch.cat(embs, 0)
    torch.save({'emb_taxctx': T, 'classes': classes}, os.path.join(OUT, f'text_emb_{TAG}_taxctx.pt'))

    class DS(Dataset):
        def __len__(self):
            return len(qfiles)

        def __getitem__(self, i):
            fn, y = qfiles[i]
            try:
                im = pp(Image.open(imgidx[fn]).convert('RGB'))
            except Exception:
                im = torch.zeros(3, 224, 224)
            return im, y

    dl = DataLoader(DS(), batch_size=64, num_workers=8, pin_memory=True)
    feats, ys = [], []
    print('encoding images...', flush=True)
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            f = model.encode_image(x)
            f = F.normalize(f.float(), dim=-1)
            feats.append(f.cpu())
            ys.append(y)
    Q = torch.cat(feats, 0)
    gold = torch.cat(ys, 0)
    torch.save({'feats': Q, 'files': [f for f, _ in qfiles], 'gold': gold},
               os.path.join(OUT, f'unseen_backbone_{TAG}_img.pt'))

    Tc = T[cand]
    S = dbnorm(Q.to(device) @ Tc.to(device).t())
    pred = S.argmax(1).cpu()
    acc = (pred == gold).float().mean().item() * 100
    print(f'{TAG} alone taxctx: {acc:.2f}', flush=True)

    # BioCLIP-H baseline on same queries
    Th = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1)
    # need H image feats for same files — use emb_train_h
    hi, hf, _ = load_emb(os.path.join(OUT, 'emb_train_h.pt'))
    Qh = torch.stack([hf[hi[fn]] for fn, _ in qfiles])
    Sh = dbnorm(Qh.to(device) @ Th[cand].to(device).t())
    acc_h = (Sh.argmax(1).cpu() == gold).float().mean().item() * 100
    fuse = dbnorm(Qh.to(device) @ Th[cand].to(device).t()) + dbnorm(Q.to(device) @ Tc.to(device).t())
    acc_f = (fuse.argmax(1).cpu() == gold).float().mean().item() * 100
    print(f'BioCLIP-H taxctx: {acc_h:.2f}', flush=True)
    print(f'fused H+{TAG}: {acc_f:.2f} (delta {acc_f-acc_h:+.2f})', flush=True)

    res = {TAG: acc, 'bioclip_h_taxctx': acc_h, 'fused': acc_f, 'delta': acc_f - acc_h, 'n': len(qfiles)}
    json.dump(res, open(os.path.join(OUT, f'unseen_backbone_{TAG}_results.json'), 'w'), indent=1)
    print('wrote', f'outputs/unseen_backbone_{TAG}_results.json', flush=True)


if __name__ == '__main__':
    main()
