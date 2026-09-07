"""Definitive test: does the shift-fine-tuned TaxaBind (tb_ctftshift) beat FROZEN
TaxaBind when fused into the v36 unseen stack? Self-contained on the pseudo-unseen
proxy (rarest-20% classes). Encodes the ~2.3k pseudo query images with BOTH frozen
and LoRA-FT TaxaBind, then compares non-TB-stack + w*TB for each.
"""
import json
import math
import os
import pickle
import time
from collections import defaultdict

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

OUT = 'outputs'
D = 'data/dl'
IMG = f'{D}/images'
DEV = 'cuda'
MODEL = 'hf-hub:MVRL/taxabind-vit-b-16'
t0 = time.time()


class LoRALinear(nn.Module):
    def __init__(s, base, r=16, alpha=32, dropout=0.05):
        super().__init__()
        s.base = base
        s.r = r
        s.scaling = alpha / max(r, 1)
        s.A = nn.Parameter(torch.zeros(r, base.in_features))
        s.B = nn.Parameter(torch.zeros(base.out_features, r))
        s.drop = nn.Dropout(dropout)

    def forward(s, x):
        return s.base(x) + (s.drop(x) @ s.A.t() @ s.B.t()) * s.scaling


def inject(model, r, alpha, top_k):
    blocks = model.visual.transformer.resblocks
    sel = range(len(blocks)) if top_k <= 0 else range(len(blocks) - top_k, len(blocks))
    for i in sel:
        blocks[i].mlp.c_fc = LoRALinear(blocks[i].mlp.c_fc, r, alpha)
        blocks[i].mlp.c_proj = LoRALinear(blocks[i].mlp.c_proj, r, alpha)


def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx[f] = os.path.join(dp, f)
    return idx


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def main():
    torch.set_num_threads(4)
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(f'{D}/label_train.json'))
    imgidx = index_images(IMG)
    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1)
    TnL = F.normalize(torch.load(f'{OUT}/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1)
    TTX = F.normalize(torch.load(f'{OUT}/text_emb_h_promptens.pt', weights_only=False)['emb_taxctx'].float(), dim=-1)
    Ttb = F.normalize(torch.load(f'{OUT}/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(), dim=-1)

    legs = {t: load(f'{OUT}/emb_train_{t}.pt') for t in
            ['ctftshift', 'ftshift', 'fullft336_v2', 'fullft336shift']}
    idxL, featL, _ = load(f'{OUT}/emb_train.pt')  # L leg
    legs['L'] = (idxL, featL, None)

    core = ['ctftshift', 'ftshift', 'fullft336shift', 'L', 'fullft336_v2']
    common = None
    for t in core:
        s = set(legs[t][0].keys())
        common = s if common is None else (common & s)
    common = [fn for fn in common if fn in lab and lab[fn] in ci and fn in imgidx]
    by = defaultdict(list)
    for fn in common:
        by[lab[fn]].append(fn)
    seen = sorted(by.keys())
    order = sorted(seen, key=lambda c: len(by[c]))
    pseudo = set(order[:int(len(seen) * 0.2)])
    kept_set = set(order[int(len(seen) * 0.2):])
    cand = torch.tensor([i for i, c in enumerate(classes) if c not in kept_set])
    cand_pos = {int(x): j for j, x in enumerate(cand.tolist())}

    qf, gold = [], []
    for c in pseudo:
        for fn in by[c]:
            qf.append(fn)
            gold.append(cand_pos[ci[c]])
    gold = torch.tensor(gold).to(DEV)
    print(f'[{time.time()-t0:.0f}s] pseudo queries={len(qf)} cand={len(cand)}', flush=True)

    # non-TB v36 stack score over cand for the query images
    def stk(t):
        idx, feats, _ = legs[t]
        return torch.stack([feats[idx[fn]] for fn in qf]).to(DEV)

    TtH_c, TTX_c, TnL_c, Ttb_c = TtH[cand].to(DEV), TTX[cand].to(DEV), TnL[cand].to(DEV), Ttb[cand].to(DEV)
    base = (dbnorm(stk('ctftshift') @ TtH_c.t()) + 0.5 * dbnorm(stk('L') @ TnL_c.t())
            + 0.75 * dbnorm(stk('fullft336_v2') @ TtH_c.t()) + 1.0 * dbnorm(stk('ftshift') @ TtH_c.t())
            + 1.0 * dbnorm(stk('ctftshift') @ TTX_c.t()))
    a_base = (base.argmax(1) == gold).float().mean().item() * 100
    print(f'[{time.time()-t0:.0f}s] non-TB stack proxy = {a_base:.2f}', flush=True)

    # encode queries with frozen + FT taxabind
    _, _, pp = open_clip.create_model_and_transforms(MODEL)

    @torch.no_grad()
    def encode(model):
        model = model.to(DEV).eval()
        out = []
        for i in range(0, len(qf), 128):
            batch = torch.stack([pp(Image.open(imgidx[fn]).convert('RGB')) for fn in qf[i:i + 128]]).to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = model.encode_image(batch).float()
            out.append(F.normalize(f, dim=-1).cpu())
        return torch.cat(out).to(DEV)

    frozen, _, _ = open_clip.create_model_and_transforms(MODEL)
    Qtb_frozen = encode(frozen)
    print(f'[{time.time()-t0:.0f}s] encoded frozen TB', flush=True)

    ck = torch.load(f'{OUT}/tb_ctftshift.pt', weights_only=False)
    ft, _, _ = open_clip.create_model_and_transforms(MODEL)
    inject(ft, ck['rank'], ck['alpha'], ck['top_k'])
    missing = ft.load_state_dict(ck['state'], strict=False)
    Qtb_ft = encode(ft)
    print(f'[{time.time()-t0:.0f}s] encoded FT TB (best_db={ck.get("db")})', flush=True)

    print('\n=== fusion: non-TB stack + w * dbnorm(TB @ Ttb) ===', flush=True)
    res = {'non_tb': a_base, 'frozen': {}, 'ft': {}}
    for name, Q in [('frozen', Qtb_frozen), ('ft', Qtb_ft)]:
        tbterm = dbnorm(Q @ Ttb_c.t())
        for w in [0.5, 1.0, 1.5, 2.0]:
            a = ((base + w * tbterm).argmax(1) == gold).float().mean().item() * 100
            res[name][str(w)] = a
            print(f'  {name:6s} w={w}: {a:.2f}  (d_vs_nonTB {a-a_base:+.2f})', flush=True)
    best_frozen = max(res['frozen'].values())
    best_ft = max(res['ft'].values())
    print(f'\nbest frozen-TB fuse: {best_frozen:.2f} | best FT-TB fuse: {best_ft:.2f} | '
          f'FT vs frozen: {best_ft-best_frozen:+.2f}', flush=True)
    json.dump(res, open(f'{OUT}/tb_ft_fuse_test_results.json', 'w'), indent=1)
    print('wrote outputs/tb_ft_fuse_test_results.json', flush=True)


if __name__ == '__main__':
    main()
