"""Extended sweep 2: extended weights + greedy combos of winning members (box).

Members: fullft336 (FF1), fullft336_v2 (FF2), h336 text (H3), v2b, v2c (when
extracted). Caches member score matrices under outputs/extcache/ so reruns are
instant. Split-half stability on finalists.

Run: python research/ext_ensemble2.py
"""
import sys, os, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from common import FishData, zc

t0 = time.time()
D = FishData()
S, s2i = D.S, D.s2i
trby, valrows = D.v20_holdout_split()
VY = torch.tensor([y for _, y in valrows])
N = len(valrows)
CACHE = 'outputs/extcache'
os.makedirs(CACHE, exist_ok=True)


def member_score(tag, use_text, chunk=1000):
    cp = os.path.join(CACHE, f'{tag}_text{int(use_text)}.pt')
    if os.path.exists(cp):
        return torch.load(cp, weights_only=False)
    p = f'outputs/emb_train_{tag}.pt'
    if tag == 'L':
        p = 'outputs/emb_train.pt'
    if not os.path.exists(p):
        return None
    d = torch.load(p, weights_only=False)
    TrI = {fn: i for i, fn in enumerate(d['files'])}
    TrF = F.normalize(d['feats'].float(), dim=-1)
    qF = torch.stack([TrF[TrI[f]] for f, _ in valrows if f in TrI])
    P = torch.zeros(S, TrF.shape[1]); cnt = torch.zeros(S)
    TF, TL = [], []
    for c in D.seen:
        for fn in trby[c]:
            if fn not in TrI:
                continue
            f = TrF[TrI[fn]]
            P[s2i[c]] += f; cnt[s2i[c]] += 1; TF.append(f); TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    TF = torch.stack(TF); TL = torch.tensor(TL)
    out = torch.empty(qF.shape[0], S)
    for i in range(0, qF.shape[0], chunk):
        e = qF[i:i + chunk]
        ps = e @ P.t()
        sim = e @ TF.t()
        cm = torch.full((e.shape[0], S), -1e9)
        cm.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + 2.0 * cm
        if use_text:
            sc = sc + 4.0 * (e @ D.TseenTax.t())
        out[i:i + chunk] = sc
    torch.save(out, cp)
    return out


def base_v20():
    sF = member_score('ft', True)
    sC = member_score('cap', True)
    sL = member_score('L', False)
    return zc(sF) + 1.0 * zc(sC) + 0.5 * zc(sL)


def acc(M, idx=None):
    idx = torch.arange(M.shape[0]) if idx is None else idx
    return (M[idx].argmax(1) == VY[idx]).float().mean().item() * 100


base = base_v20()
b = acc(base)
g = torch.Generator().manual_seed(0)
perm = torch.randperm(N, generator=g)
h1, h2 = perm[:N // 2], perm[N // 2:]
print(f'[{time.time()-t0:.0f}s] GATE v20 = {b:.2f}', flush=True)
assert abs(b - 87.94) < 0.3

Z = {}
for tag, txt in [('fullft336', True), ('fullft336_v2', True), ('h336', True),
                 ('v2b', True), ('v2c', True), ('ctftbig', True)]:
    s = member_score(tag, txt)
    if s is None:
        print(f'{tag}: missing, skip', flush=True)
    else:
        Z[tag] = zc(s)
        print(f'[{time.time()-t0:.0f}s] {tag} ready', flush=True)

print('--- extended single-member weights ---', flush=True)
for tag in Z:
    for w in [0.75, 1.0, 1.5, 2.0, 3.0]:
        a = acc(base + w * Z[tag])
        print(f'v20 + {w}*{tag}: {a:.2f}', flush=True)

print('--- pairwise FF combos ---', flush=True)
best = (b, 'v20', {})
if 'fullft336' in Z and 'fullft336_v2' in Z:
    for a_ in [0.5, 0.75, 1.0, 1.5]:
        for b_ in [0.0, 0.5, 0.75, 1.0]:
            M = base + a_ * Z['fullft336'] + b_ * Z['fullft336_v2']
            aa = acc(M)
            tag = f'FF1x{a_}+FF2x{b_}'
            if aa > best[0]:
                best = (aa, tag, {'ff1': a_, 'ff2': b_})
            print(f'{tag}: {aa:.2f}', flush=True)

print('--- add h336 / v2b / v2c on top of best ---', flush=True)
cur = best[2]
Mb = base + cur.get('ff1', 0) * Z.get('fullft336', 0) + cur.get('ff2', 0) * Z.get('fullft336_v2', 0)
for tag in ['h336', 'v2b', 'v2c', 'ctftbig']:
    if tag not in Z:
        continue
    for w in [0.25, 0.5, 0.75, 1.0]:
        aa = acc(Mb + w * Z[tag])
        a1 = acc(Mb + w * Z[tag], h1)
        a2 = acc(Mb + w * Z[tag], h2)
        print(f'best + {w}*{tag}: {aa:.2f} ({a1:.2f}/{a2:.2f})', flush=True)

print(f'BEST SO FAR: {best}', flush=True)
