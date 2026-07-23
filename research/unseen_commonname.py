"""Target the catastrophic weak link: real unseen zero-shot accuracy b=13.93% (vs seen a=88.83%).
Decomposition proved overall = 0.50*s + 0.061*u; even a perfect gate caps at 56% with current b,
but lifting b to ~25% makes 53% reachable with a merely-decent gate. Project playbook (verified,
Parashar EMNLP'23) claims scientific->common-name gives 2-5x zero-shot gain. The common-name
H-space embeddings already exist (text_emb_h_common.pt, all 17393 classes) but have NEVER been
wired into the unseen route (which uses ctftbig@taxon + 0.5*L@scientific-name).

This compares unseen-route text sources on the pseudo-unseen class holdout (rarest-20% of seen).
Holdout absolute numbers are ~2x inflated vs real (holdout unseen ~28% -> real 14%), but RELATIVE
ranking between text sources should transfer. If common-names clearly wins here, it's the next real
submission lever.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from common import FishData, dbnorm

t0 = time.time()
D = FishData()
cand = D.cand                      # non-kept class indices (pseudo-unseen + true-unseen candidate pool)
print(f'[{time.time()-t0:.0f}s] cand pool={len(cand)}  pseudo_unseen_classes={len(D.pseudo)}', flush=True)

OUT = 'outputs'


def load(tag):
    p = os.path.join(OUT, f'emb_train_{tag}.pt')
    if not os.path.exists(p):
        return None
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1)


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


# image encoders for the unseen route
ctftbig = load('ctftbig')
qc, gold = queries(ctftbig)
qL, goldL = queries((D.LtI, D.LtF))
assert torch.equal(gold, goldL)
print(f'[{time.time()-t0:.0f}s] unseen queries={len(qc)}', flush=True)

# text anchors, restricted to candidate pool
TtH = D.TtH[cand]     # taxon (H)
TnL = D.TnL[cand]     # scientific name (L)
TnH = D.TnH[cand]     # scientific name (H)

# common-name H-space anchors
cn = torch.load(os.path.join(OUT, 'text_emb_h_common.pt'), weights_only=False)
assert cn['classes'] == D.classes, 'common-name class order mismatch'
TcH = F.normalize(cn['emb_common'].float(), dim=-1)[cand]   # common name (H)

MtH = qc @ TtH.t()   # ctftbig @ taxon(H)
MnL = qL @ TnL.t()   # L @ scientific-name(L)
McH = qc @ TcH.t()   # ctftbig @ common-name(H)
MnH = qc @ TnH.t()   # ctftbig @ scientific-name(H)

res = {}


def score(name, M):
    a = t1(M, gold)
    res[name] = a
    print(f'  {name:<44} {a:.2f}', flush=True)


print('--- single text anchors (dbnorm, ctftbig or L image feats) ---')
score('ctftbig@taxonH', dbnorm(MtH))
score('ctftbig@sciname_H', dbnorm(MnH))
score('ctftbig@commonname_H', dbnorm(McH))
score('L@sciname_L', dbnorm(MnL))

print('--- DEPLOYED baseline (what the real 45.19 submission used) ---')
dep = dbnorm(MtH) + 0.5 * dbnorm(MnL)
DEP_KEY = 'DEPLOYED ctftbig@taxonH+0.5*L@sciname'
dep_acc = t1(dep, gold)
res[DEP_KEY] = dep_acc
print(f'  DEPLOYED ctftbig@taxonH + 0.5*L@sciname       {dep_acc:.2f}')

print('--- common-name blends (does common name lift it?) ---')
for w in [0.25, 0.5, 1.0, 1.5, 2.0]:
    a = t1(dep + w * dbnorm(McH), gold)
    res[f'DEPLOYED + {w}*common_H'] = a
    print(f'  DEPLOYED + {w}*common_H                        {a:.2f}  (delta {a-dep_acc:+.2f})')

print('--- common name REPLACING taxon as the primary anchor ---')
for w in [0.25, 0.5, 1.0]:
    a = t1(dbnorm(McH) + w * dbnorm(MnL), gold)
    res[f'common_H + {w}*L@sciname'] = a
    print(f'  common_H + {w}*L@sciname                       {a:.2f}')
for w in [0.5, 1.0]:
    a = t1(dbnorm(McH) + w * dbnorm(MtH), gold)
    res[f'common_H + {w}*taxonH'] = a
    print(f'  common_H + {w}*taxonH                          {a:.2f}')

print('--- triple blend: taxon + sciname + common ---')
for wc in [0.5, 1.0]:
    a = t1(dbnorm(MtH) + 0.5 * dbnorm(MnL) + wc * dbnorm(McH), gold)
    res[f'taxonH + 0.5*L@sci + {wc}*common_H'] = a
    print(f'  taxonH + 0.5*L@sci + {wc}*common_H            {a:.2f}')

best = max(res.items(), key=lambda kv: kv[1])
print(f'\nBEST: {best[0]} = {best[1]:.2f}  (deployed baseline = {dep_acc:.2f})')
json.dump(res, open('outputs/unseen_commonname_results.json', 'w'), indent=1)
print('wrote outputs/unseen_commonname_results.json')
