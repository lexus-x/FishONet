"""Final validated b-stack search on pseudo-unseen proxy with class-disjoint CV.
Includes all non-TB legs (TtH/TTX/TnL) + frozen TaxaBind + FT TaxaBind (Ttb),
encoding the ~2.3k pseudo query images once. Greedy coordinate ascent tuned on
half the classes, reported on the held-out half. Starts from the v36 stack.
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
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1)


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

    legs = {t: load(f'{OUT}/emb_train_{t}.pt') for t in ['ctftshift', 'ctftbig', 'ftshift', 'fullft336_v2', 'fullft336shift']}
    legs['L'] = load(f'{OUT}/emb_train.pt')

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
    pseudo_sorted = order[:int(len(seen) * 0.2)]
    pseudo = set(pseudo_sorted)
    kept_set = set(order[int(len(seen) * 0.2):])
    cand = torch.tensor([i for i, c in enumerate(classes) if c not in kept_set])
    cand_pos = {int(x): j for j, x in enumerate(cand.tolist())}
    tune_cls = set(pseudo_sorted[::2])

    qf, gold, is_tune = [], [], []
    for c in pseudo:
        for fn in by[c]:
            qf.append(fn)
            gold.append(cand_pos[ci[c]])
            is_tune.append(c in tune_cls)
    gold = torch.tensor(gold).to(DEV)
    tune_mask = torch.tensor(is_tune, device=DEV)
    val_mask = ~tune_mask
    print(f'[{time.time()-t0:.0f}s] pseudo q={len(qf)} tune={int(tune_mask.sum())} val={int(val_mask.sum())} cand={len(cand)}', flush=True)

    TtH_c, TTX_c, TnL_c, Ttb_c = TtH[cand].to(DEV), TTX[cand].to(DEV), TnL[cand].to(DEV), Ttb[cand].to(DEV)

    def stk(t):
        idx, feats = legs[t]
        return torch.stack([feats[idx[fn]] for fn in qf]).to(DEV)

    terms = {}
    for t in ['ctftshift', 'ctftbig', 'ftshift', 'fullft336_v2', 'fullft336shift']:
        terms[(t, 'TtH')] = dbnorm(stk(t) @ TtH_c.t())
        terms[(t, 'TTX')] = dbnorm(stk(t) @ TTX_c.t())
    terms[('L', 'TnL')] = dbnorm(stk('L') @ TnL_c.t())

    # encode TB (frozen + FT)
    _, _, pp = open_clip.create_model_and_transforms(MODEL)

    @torch.no_grad()
    def encode(model):
        model = model.to(DEV).eval()
        out = []
        for i in range(0, len(qf), 128):
            b = torch.stack([pp(Image.open(imgidx[fn]).convert('RGB')) for fn in qf[i:i + 128]]).to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = model.encode_image(b).float()
            out.append(F.normalize(f, dim=-1).cpu())
        return torch.cat(out).to(DEV)

    fr, _, _ = open_clip.create_model_and_transforms(MODEL)
    terms[('tb_frozen', 'Ttb')] = dbnorm(encode(fr) @ Ttb_c.t())
    ck = torch.load(f'{OUT}/tb_ctftshift.pt', weights_only=False)
    ftm, _, _ = open_clip.create_model_and_transforms(MODEL)
    inject(ftm, ck['rank'], ck['alpha'], ck['top_k'])
    ftm.load_state_dict(ck['state'], strict=False)
    terms[('tb_ft', 'Ttb')] = dbnorm(encode(ftm) @ Ttb_c.t())
    print(f'[{time.time()-t0:.0f}s] terms={len(terms)}', flush=True)

    def acc(w, mask=None):
        S = None
        for k, v in w.items():
            if v == 0 or k not in terms:
                continue
            S = terms[k] * v if S is None else S + terms[k] * v
        p = S.argmax(1)
        m = torch.ones_like(gold, dtype=torch.bool) if mask is None else mask
        return (p[m] == gold[m]).float().mean().item() * 100

    base = {('ctftshift', 'TtH'): 1.0, ('L', 'TnL'): 0.5, ('fullft336_v2', 'TtH'): 0.75,
            ('ftshift', 'TtH'): 1.0, ('ctftshift', 'TTX'): 1.0, ('tb_frozen', 'Ttb'): 1.0}
    print(f'v36 base: full={acc(base):.2f} tune={acc(base,tune_mask):.2f} val={acc(base,val_mask):.2f}', flush=True)

    cur = dict(base)
    all_terms = list(terms.keys())
    grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    best = acc(cur, tune_mask)
    for it in range(4):
        improved = False
        for k in all_terms:
            lb, lw = best, cur.get(k, 0.0)
            for w in grid:
                cur[k] = w
                a = acc(cur, tune_mask)
                if a > lb + 1e-6:
                    lb, lw = a, w
            cur[k] = lw
            if lb > best + 1e-6:
                best, improved = lb, True
        print(f'[{time.time()-t0:.0f}s] iter{it}: tune={best:.2f}', flush=True)
        if not improved:
            break
    cur = {k: v for k, v in cur.items() if v != 0}
    print('\n=== tuned stack ===', flush=True)
    for k, v in sorted(cur.items(), key=lambda x: -x[1]):
        print(f'  {k[0]:16s} @ {k[1]:4s} : {v}', flush=True)
    print(f'v36 base  val={acc(base,val_mask):.2f} full={acc(base):.2f}', flush=True)
    print(f'tuned     val={acc(cur,val_mask):.2f} full={acc(cur):.2f}  (val d={acc(cur,val_mask)-acc(base,val_mask):+.2f})', flush=True)
    json.dump({'base': {f'{k[0]}@{k[1]}': v for k, v in base.items()},
               'tuned': {f'{k[0]}@{k[1]}': v for k, v in cur.items()},
               'base_val': acc(base, val_mask), 'tuned_val': acc(cur, val_mask),
               'base_full': acc(base), 'tuned_full': acc(cur)},
              open(f'{OUT}/fusion_sweep_final_results.json', 'w'), indent=1)
    print('wrote outputs/fusion_sweep_final_results.json', flush=True)


if __name__ == '__main__':
    main()
