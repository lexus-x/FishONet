"""Probe heavier ctftshift query TTA on pseudo-unseen (v36 text stack baseline).

Compares cached ctftshift train queries vs on-the-fly re-encode with:
  light: center-crop + hflip (2 views)
  squash4: pp + hflip + squash + squash_hflip (4 views, current deploy extract)
  heavy8: light4 + extra wide resize (256 short) center-crop + hflip (8 views)

Does not re-extract test/unseen — proxy only. Kill bar: +0.5 on full pseudo-unseen stack.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT  # noqa: E402

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'
DEV = 'cuda'
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class LoRALinear(nn.Module):
    def __init__(s, base, r=16, alpha=32):
        super().__init__()
        s.base = base
        for p in s.base.parameters():
            p.requires_grad_(False)
        s.r = r
        s.scaling = alpha / max(r, 1)
        s.A = nn.Parameter(torch.zeros(r, base.in_features))
        s.B = nn.Parameter(torch.zeros(base.out_features, r))
        s.drop = nn.Dropout(0.0)
        s.lora_scale = 1.0

    def forward(s, x):
        return s.base(x) + (s.drop(x) @ s.A.t() @ s.B.t()) * s.scaling * s.lora_scale


def inject_lora(model, r, alpha, top_k=0):
    blocks = model.visual.transformer.resblocks
    sel = range(len(blocks)) if top_k <= 0 else range(len(blocks) - top_k, len(blocks))
    for i in sel:
        blk = blocks[i]
        blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, r, alpha)
        blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, r, alpha)


def set_scale(model, s):
    for blk in model.visual.transformer.resblocks:
        for m in (blk.mlp.c_fc, blk.mlp.c_proj):
            if isinstance(m, LoRALinear):
                m.lora_scale = s


def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx[f] = os.path.join(dp, f)
    return idx


def top1(S, gold):
    return (S.argmax(1) == gold).float().mean().item() * 100


def stack_score(legs, q_ctft, TtH_c, TnL_c, TTX_c, gold):
    S = (
        dbnorm(q_ctft @ TtH_c.t())
        + 0.5 * dbnorm(legs['L'] @ TnL_c.t())
        + 0.75 * dbnorm(legs['fullft336_v2'] @ TtH_c.t())
        + 1.0 * dbnorm(legs['ftshift'] @ TtH_c.t())
        + 1.0 * dbnorm(q_ctft @ TTX_c.t())
    )
    return top1(S, gold)


@torch.no_grad()
def encode_views(model, pp, img, mode):
    squash = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    wide = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    views = [pp(img)]
    if mode in ('squash4', 'heavy8'):
        views.append(squash(img))
    if mode == 'heavy8':
        views.append(wide(img))
    feats = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for v in views:
            x = v.unsqueeze(0).to(DEV)
            f = F.normalize(model.encode_image(x).float(), dim=-1)
            f2 = F.normalize(model.encode_image(torch.flip(x, dims=[-1])).float(), dim=-1)
            feats.append(f)
            feats.append(f2)
    f = torch.cat(feats, dim=0).sum(0, keepdim=True)
    return F.normalize(f, dim=-1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(OUT, 'ctft_shift.pt'))
    ap.add_argument('--scale', type=float, default=0.4)
    ap.add_argument('--mode', default='all', choices=['all', 'light', 'squash4', 'heavy8'])
    ap.add_argument('--max_q', type=int, default=0, help='0 = all pseudo queries')
    a = ap.parse_args()

    t0 = time.time()
    D = FishData()
    cand = D.cand
    TtH_c = D.TtH[cand]
    TnL_c = D.TnL[cand]
    TTX = F.normalize(
        torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(),
        dim=-1,
    )
    TTX_c = TTX[cand]

    legs = {}
    for tag in ['ftshift', 'fullft336_v2']:
        idx, feats, _ = load_emb(os.path.join(OUT, f'emb_train_{tag}.pt'))
        q, gold = [], []
        for c in D.pseudo:
            for fn in D.by[c]:
                if fn in idx:
                    q.append(feats[idx[fn]])
                    gold.append(D.cand_pos[D.ci[c]])
        legs[tag] = torch.stack(q)
    qL, gold = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in D.LtI:
                qL.append(D.LtF[D.LtI[fn]])
                gold.append(D.cand_pos[D.ci[c]])
    legs['L'] = torch.stack(qL)
    gold_t = torch.tensor(gold)

    ct_idx, ct_feat, _ = load_emb(os.path.join(OUT, 'emb_train_ctftshift.pt'))
    q_cached, fn_list = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in ct_idx:
                q_cached.append(ct_feat[ct_idx[fn]])
                fn_list.append(fn)
    q_cached = torch.stack(q_cached)
    if a.max_q:
        q_cached = q_cached[: a.max_q]
        fn_list = fn_list[: a.max_q]
        gold_t = gold_t[: a.max_q]
        for k in legs:
            legs[k] = legs[k][: a.max_q]

    base = stack_score(legs, q_cached, TtH_c, TnL_c, TTX_c, gold_t)
    print(f'cached ctftshift stack = {base:.2f}  n={len(fn_list)}', flush=True)

    ck = torch.load(a.ckpt, weights_only=False)
    model, _, pp = open_clip.create_model_and_transforms(MODEL)
    inject_lora(model, ck['rank'], ck['alpha'], ck['top_k'])
    own = dict(model.named_parameters())
    for n, v in ck['state'].items():
        own[n].data.copy_(v)
    model = model.to(DEV).eval()
    set_scale(model, a.scale)

    imgidx = index_images(os.path.join(os.path.dirname(OUT), 'data', 'dl', 'images'))
    modes = ['light', 'squash4', 'heavy8'] if a.mode == 'all' else [a.mode]
    results = {'cached_baseline': base, 'n': len(fn_list), 'ckpt': a.ckpt}

    for mode in modes:
        qs = []
        t1 = time.time()
        for i, fn in enumerate(fn_list):
            img = Image.open(imgidx[fn]).convert('RGB')
            qs.append(encode_views(model, pp, img, mode).squeeze(0))
            if i and i % 400 == 0:
                print(f'  {mode} {i}/{len(fn_list)}', flush=True)
        q = torch.stack(qs)
        acc = stack_score(legs, q, TtH_c, TnL_c, TTX_c, gold_t)
        results[mode] = {'acc': acc, 'delta': acc - base}
        print(f'{mode}: {acc:.2f}  delta={acc-base:+.2f}  ({time.time()-t1:.0f}s)', flush=True)

    out_path = os.path.join(OUT, 'ctft_heavy_tta_proxy_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'wrote {out_path} total {(time.time()-t0):.0f}s', flush=True)


if __name__ == '__main__':
    main()
