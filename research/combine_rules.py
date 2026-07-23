"""Alternative combination rules for the seen ensemble (box or local).

Gate: v20 (sum of zc) = 87.94. Tests:
  A. Product-of-Experts: sum of per-encoder log_softmax(score / T)
  B. max-rule / rank-average
  C. Linear probe (multinomial logistic) on concatenated [ft;cap;L] features
     trained on trby (52,393), evaluated on the 11,866 holdout.
Run: python research/combine_rules.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from common import FishData, protos_and_train, zc

t0 = time.time()
D = FishData()
S, s2i = D.S, D.s2i
trby, valrows = D.v20_holdout_split()
VY = torch.tensor([y for _, y in valrows])
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

encs = {
    'ft': (D.FtI, D.FtF, True),
    'cap': (D.CtI, D.CtF, True),
    'L': (D.LtI, D.LtF, False),
}
scores, tr_feats, val_feats = {}, {}, {}
for name, (TrI, TrF, use_text) in encs.items():
    P, TF, TL = protos_and_train(TrI, TrF, trby, D.seen, S, s2i)
    vF = torch.stack([TrF[TrI[f]] for f, _ in valrows])
    out = torch.empty(vF.shape[0], S)
    for i in range(0, vF.shape[0], 1000):
        e = vF[i:i + 1000]
        ps = e @ P.t()
        sim = e @ TF.t()
        cm = torch.full((e.shape[0], S), -1e9)
        cm.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + 2.0 * cm
        if use_text:
            sc = sc + 4.0 * (e @ D.TseenTax.t())
        out[i:i + 1000] = sc
    scores[name] = out
    tr_feats[name] = torch.stack([TrF[TrI[f]] for c in D.seen for f in trby[c]])
    val_feats[name] = vF
    print(f'[{time.time()-t0:.0f}s] {name} scored', flush=True)

trY = torch.tensor([s2i[c] for c in D.seen for f in trby[c]])
base = zc(scores['ft']) + 1.0 * zc(scores['cap']) + 0.5 * zc(scores['L'])


def acc(M):
    return (M.argmax(1) == VY).float().mean().item() * 100


print(f'GATE v20: {acc(base):.2f}', flush=True)

# A. PoE: sum of log_softmax(score/T)
for T in [0.02, 0.05, 0.1, 0.2]:
    poe = (F.log_softmax(scores['ft'] / T, dim=1) +
           F.log_softmax(scores['cap'] / T, dim=1) +
           0.5 * F.log_softmax(scores['L'] / T, dim=1))
    print(f'PoE T={T}: {acc(poe):.2f}', flush=True)

# A2. hybrid: zc-sum + PoE residual weight
for T in [0.05, 0.1]:
    for w in [0.25, 0.5, 1.0]:
        hy = base + w * (F.log_softmax(scores['ft'] / T, dim=1) +
                         F.log_softmax(scores['cap'] / T, dim=1) +
                         0.5 * F.log_softmax(scores['L'] / T, dim=1))
        print(f'v20 + {w}*PoE(T={T}): {acc(hy):.2f}', flush=True)

# C. linear probe on concat features
d = sum(v.shape[1] for v in tr_feats.values())
Xtr = torch.cat([tr_feats['ft'], tr_feats['cap'], tr_feats['L']], dim=1).to(DEV)
Xva = torch.cat([val_feats['ft'], val_feats['cap'], val_feats['L']], dim=1).to(DEV)
Ytr = trY.to(DEV)
W = torch.zeros(d, S, device=DEV, requires_grad=True)
bs = torch.zeros(S, device=DEV, requires_grad=True)
opt = torch.optim.Adam([W, bs], lr=1e-2, weight_decay=1e-4)
for step in range(3000):
    idx = torch.randint(0, Xtr.shape[0], (4096,), device=DEV)
    loss = F.cross_entropy(Xtr[idx] @ W + bs, Ytr[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f'  probe step {step} loss {loss.item():.3f}', flush=True)
with torch.no_grad():
    lp = (Xva @ W + bs).cpu()
print(f'LINEAR PROBE (concat ft+cap+L): {acc(lp):.2f}', flush=True)
for w in [0.5, 1.0, 2.0]:
    print(f'v20 + {w}*zc(probe): {acc(base + w * zc(lp)):.2f}', flush=True)
print('DONE', flush=True)
