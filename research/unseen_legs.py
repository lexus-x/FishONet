"""EXACT deployed-unseen hard-sim on the box (ctft train features exist there).

Deployed unseen route (v15..v20): dis(ctft @ taxonH) + 0.5 * dis(L @ nameL)
over non-kept candidate classes, DBNorm = dis(., 0.05, 0.5). Hard-sim split:
rarest-20% seen classes = pseudo-unseen. Project reference: ~27.7-27.8.

Tests whether extra cached encoder legs help:
  + w * dis(frozenH @ taxonH)          (emb_train_h)
  + w * dis(frozenH_TTA @ taxonH)      (emb_train_h_tta)
  + w * dis(fullft336 @ taxonH)        (emb_train_fullft336[_v2])
  + w * dis(ctftbig @ nameH / descH)   (text variants on the strong leg)

Run on box: python research/unseen_legs.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from common import FishData, dbnorm

t0 = time.time()
D = FishData()
cand = D.cand
print(f'[{time.time()-t0:.0f}s] cand={len(cand)}', flush=True)

OUT = 'outputs'


def load(tag):
    p = os.path.join(OUT, f'emb_train_{tag}.pt')
    if not os.path.exists(p):
        return None
    d = torch.load(p, weights_only=False)
    idx = {fn: i for i, fn in enumerate(d['files'])}
    feats = F.normalize(d['feats'].float(), dim=-1)
    return idx, feats


def queries(idx_feats):
    idx, feats = idx_feats
    q, y = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx:
                q.append(feats[idx[fn]])
                y.append(D.cand_pos[D.ci[c]])
    return torch.stack(q), torch.tensor(y)


def t1(S, gold):
    return (S.argmax(1) == gold).float().mean().item() * 100


LEGS = {}
ref = {}
for tag in ['ctftbig', 'h', 'h_tta', 'fullft336', 'fullft336_v2']:
    li = load(tag)
    if li is None:
        print(f'{tag}: MISSING', flush=True)
        continue
    q, gold = queries(li)
    LEGS[tag] = (q, gold)
    print(f'[{time.time()-t0:.0f}s] {tag}: queries={len(q)}', flush=True)

# L leg (deployed second term)
qL, goldL = queries((D.LtI, D.LtF))
TtH_c = D.TtH[cand]
TnL_c = D.TnL[cand]
TnH_c = D.TnH[cand]
TdH_c = D.TdH[cand]

results = {}
base_gold = None
M = {}
for tag, (q, gold) in LEGS.items():
    M[tag] = q @ TtH_c.t()
    base_gold = gold
M['L'] = qL @ TnL_c.t()
LEGS['L'] = (qL, goldL)

deployed = dbnorm(M['ctftbig']) + 0.5 * dbnorm(M['L'])
a_dep = t1(deployed, base_gold)
print(f'[{time.time()-t0:.0f}s] DEPLOYED hard-sim: {a_dep:.2f} (expect ~27.7)', flush=True)
results['deployed'] = a_dep

print('--- single legs (DBNorm) ---', flush=True)
for tag in M:
    a = t1(dbnorm(M[tag]), LEGS[tag][1])
    results[f'alone_{tag}'] = a
    print(f'{tag:>14}: {a:.2f}', flush=True)

print('--- additive 3rd legs on deployed ---', flush=True)
for tag in ['h', 'h_tta', 'fullft336', 'fullft336_v2']:
    if tag not in M:
        continue
    for w in [0.25, 0.5, 1.0]:
        a = t1(deployed + w * dbnorm(M[tag]), base_gold)
        results[f'deployed+{w}*{tag}'] = a
        print(f'deployed + {w}*dis({tag}): {a:.2f} (delta {a-a_dep:+.2f})', flush=True)

print('--- ctftbig text variants (replace taxon leg) ---', flush=True)
qc = LEGS['ctftbig'][0]
for name, T in [('nameH', TnH_c), ('descH', TdH_c),
                ('taxon+name', F.normalize(TtH_c + TnH_c, dim=-1))]:
    Mv = qc @ T.t()
    a = t1(dbnorm(Mv) + 0.5 * dbnorm(M['L']), base_gold)
    results[f'ctft_{name}+0.5L'] = a
    print(f'ctft@{name} + 0.5*L: {a:.2f} (delta {a-a_dep:+.2f})', flush=True)

print('--- DBNorm temperature re-sweep on deployed ---', flush=True)
for tc in [0.02, 0.05, 0.1]:
    for tr in [0.1, 0.5, 1.0]:
        a = t1(dbnorm(M['ctftbig'], tc, tr) + 0.5 * dbnorm(M['L'], tc, tr), base_gold)
        results[f'tc{tc}_tr{tr}'] = a
        print(f'tc={tc} tr={tr}: {a:.2f} (delta {a-a_dep:+.2f})', flush=True)

json.dump(results, open('outputs/unseen_legs_results.json', 'w'), indent=1)
print('wrote outputs/unseen_legs_results.json', flush=True)
