"""Extended-member ensemble bake-off (runs on the GPU box, cwd=~/onet).

Gate: reproduce v20 holdout = 87.94. Then test each cached embedding set as an
ADDITIVE ensemble member (scored proto + 2*cmax [+ 4*taxon for H-space members]),
sweep weight, then greedy forward selection with split-half stability.

Members tested (whichever exist in outputs/): v2b, dino, h, h_tta, h336,
ftrobust, fullft336, fullft336_v2, ctftbig.

Run: python research/ext_ensemble.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from common import FishData, protos_and_train, zc

t0 = time.time()
D = FishData()
S, s2i = D.S, D.s2i
trby, valrows = D.v20_holdout_split()
VY = torch.tensor([y for _, y in valrows])
N = len(valrows)
print(f'[{time.time()-t0:.0f}s] val={N}', flush=True)

OUT = 'outputs'
MEMBERS = ['v2b', 'dino', 'h', 'h_tta', 'h336', 'ftrobust', 'fullft336', 'fullft336_v2', 'ctftbig']


def load_member(tag):
    p = os.path.join(OUT, f'emb_train_{tag}.pt')
    if not os.path.exists(p):
        return None
    d = torch.load(p, weights_only=False)
    idx = {fn: i for i, fn in enumerate(d['files'])}
    feats = F.normalize(d['feats'].float(), dim=-1)
    q = []
    keep = []
    for j, (f, _) in enumerate(valrows):
        if f in idx:
            q.append(feats[idx[f]])
            keep.append(j)
    return idx, feats, torch.stack(q), torch.tensor(keep)


# --- proper per-member scoring (protos from trby) ---
def member_score(TrI, TrF, qF, use_text, chunk=1000):
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
    return out


# base v20 components (recompute here for exactness)
sF = member_score(D.FtI, D.FtF, torch.stack([D.FtF[D.FtI[f]] for f, _ in valrows]), True)
sC = member_score(D.CtI, D.CtF, torch.stack([D.CtF[D.CtI[f]] for f, _ in valrows]), True)
sL = member_score(D.LtI, D.LtF, torch.stack([D.LtF[D.LtI[f]] for f, _ in valrows]), False)
base = zc(sF) + 1.0 * zc(sC) + 0.5 * zc(sL)


def acc(M, idx=None):
    idx = torch.arange(M.shape[0]) if idx is None else idx
    return (M[idx].argmax(1) == VY[idx]).float().mean().item() * 100


b = acc(base)
g = torch.Generator().manual_seed(0)
perm = torch.randperm(N, generator=g)
h1, h2 = perm[:N // 2], perm[N // 2:]
print(f'[{time.time()-t0:.0f}s] GATE v20 = {b:.2f} (halves {acc(base,h1):.2f}/{acc(base,h2):.2f})', flush=True)
assert abs(b - 87.94) < 0.3

results = []
for tag in MEMBERS:
    m = load_member(tag)
    if m is None:
        print(f'--- {tag}: MISSING, skip', flush=True)
        continue
    TrI, TrF, q, keep = m
    if len(keep) != N:
        print(f'--- {tag}: covers {len(keep)}/{N} val, skip (incomplete)', flush=True)
        continue
    is_h_space = TrF.shape[1] == 1024 and tag != 'dino'
    for use_text in ([False, True] if is_h_space else [False]):
        sM = member_score(TrI, TrF, q, use_text)
        alone = acc(zc(sM))
        row = {'member': tag, 'text': use_text, 'alone': alone}
        for w in [0.1, 0.25, 0.5, 0.75, 1.0]:
            a = acc(base + w * zc(sM))
            row[f'w{w}'] = a
        print(f'{tag:>12} text={use_text} alone={alone:.2f} | ' +
              ' '.join(f'w{w}={row[f"w{w}"]:.2f}' for w in [0.1, 0.25, 0.5, 0.75, 1.0]), flush=True)
        results.append(row)

json.dump(results, open('outputs/ext_ensemble_results.json', 'w'), indent=1)
print('wrote outputs/ext_ensemble_results.json', flush=True)
