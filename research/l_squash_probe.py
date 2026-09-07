"""Cheap decisive probe of the ONE flagged un-run lever:
shift/squash-TTA on the frozen BioCLIP-2 L (emb_train.pt) leg, stacked into v36.

Re-encodes only the 2,318 pseudo-unseen proxy queries with the frozen L model using
squash-TTA (Resize(224,224) squashing the elongated fish + hflip), then swaps the L(name)
leg in the v36 unseen stack and measures the proxy delta. If <0.5 -> dead (kill bar), and a
full shift-LoRA on L (weaker base than the H already in the stack) cannot do better.
"""
import json
import os
import sys
import time

import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT

DEV = 'cuda'
t0 = time.time()
D = FishData()
cand = D.cand

# collect pseudo-unseen query files (same order as common.queries)
imgroot = 'data/dl/images'
idxpaths = {}
for dp, _, fs in os.walk(imgroot):
    for f in fs:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            idxpaths[f] = os.path.join(dp, f)

qfiles, gold = [], []
for c in D.pseudo:
    for fn in D.by[c]:
        if fn in D.LtI:  # same filter as queries(D.LtI,...)
            qfiles.append(fn)
            gold.append(D.cand_pos[D.ci[c]])
gold = torch.tensor(gold)
print(f'[{time.time()-t0:.0f}s] pseudo queries={len(qfiles)}', flush=True)

model, _, pp = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
model = model.to(DEV).eval()
mean = (0.48145466, 0.4578275, 0.40821073); std = (0.26862954, 0.26130258, 0.27577711)
squash_tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                       T.ToTensor(), T.Normalize(mean, std)])


def encode(files, squash):
    outs = []
    bs = 128
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(files), bs):
            imgs = [Image.open(idxpaths[fn]).convert('RGB') for fn in files[i:i + bs]]
            x = torch.stack([pp(im) for im in imgs]).to(DEV)
            f = F.normalize(model.encode_image(x).float(), dim=-1)
            f = f + F.normalize(model.encode_image(torch.flip(x, [-1])).float(), dim=-1)
            if squash:
                xs = torch.stack([squash_tf(im) for im in imgs]).to(DEV)
                f = f + F.normalize(model.encode_image(xs).float(), dim=-1)
                f = f + F.normalize(model.encode_image(torch.flip(xs, [-1])).float(), dim=-1)
            outs.append(F.normalize(f, dim=-1).cpu())
    return torch.cat(outs)


qL_base = encode(qfiles, squash=False)
qL_sq = encode(qfiles, squash=True)
print(f'[{time.time()-t0:.0f}s] encoded base+squash', flush=True)

TtH_c = D.TtH[cand]
TnL_c = D.TnL[cand]
TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1)
TTX_c = TTX[cand]


def qq(idx, feats):
    return torch.stack([feats[idx[fn]] for fn in qfiles])


legs = {}
for tag in ['ctftshift', 'ftshift', 'fullft336_v2']:
    i2, ff, _ = load_emb(os.path.join(OUT, f'emb_train_{tag}.pt'))
    legs[tag] = qq(i2, ff)


def t1(S):
    return (S.argmax(1) == gold).float().mean().item() * 100


def stack(qL, extra=None, w=0.0):
    S = (dbnorm(legs['ctftshift'] @ TtH_c.t()) + 0.5 * dbnorm(qL @ TnL_c.t())
         + 0.75 * dbnorm(legs['fullft336_v2'] @ TtH_c.t())
         + 1.0 * dbnorm(legs['ftshift'] @ TtH_c.t())
         + 1.0 * dbnorm(legs['ctftshift'] @ TTX_c.t()))
    if extra is not None:
        S = S + w * dbnorm(extra @ TnL_c.t())
    return S


base = t1(stack(qL_base))
swap = t1(stack(qL_sq))
print(f'\nv36 stack, L(name) baseline  = {base:.2f}', flush=True)
print(f'v36 stack, L(name) squash-TTA = {swap:.2f}  delta {swap-base:+.2f}', flush=True)
rows = {'baseline': base, 'squash_swap': swap, 'delta_swap': swap - base}
for w in [0.5, 1.0]:
    a = t1(stack(qL_base, extra=qL_sq, w=w))
    print(f'v36 stack + {w}*L_squash(name): {a:.2f}  delta {a-base:+.2f}', flush=True)
    rows[f'add_squash_w{w}'] = a
json.dump(rows, open(os.path.join(OUT, 'l_squash_probe_results.json'), 'w'), indent=1)
print('wrote outputs/l_squash_probe_results.json', flush=True)
