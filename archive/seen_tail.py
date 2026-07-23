"""SEEN long-tail rescue: does adding a TEXT term to the prototype blend help low-count classes?
Same ft.py holdout (classes>=3, last 20%). Stratify val accuracy by train-image-count.
Deployed blend = proto_sim + 2*class_max_sim. Test blend + lam*text_sim, esp. for tail classes.
Uses FT features (deployed seen route)."""
import json, torch
import torch.nn.functional as F
from collections import defaultdict

D = 'data/dl'
TnH = F.normalize(torch.load('outputs/text_emb_h.pt', weights_only=False)['emb_name'].float(), dim=-1)
classesAll = torch.load('outputs/text_emb_h.pt', weights_only=False)['classes']
ciAll = {c: i for i, c in enumerate(classesAll)}
d = torch.load('outputs/emb_train_ft.pt', weights_only=False)
files, feats = d['files'], F.normalize(d['feats'].float(), dim=1)
fmap = {fn: i for i, fn in enumerate(files)}
lab = json.load(open(f'{D}/label_train.json'))

by = defaultdict(list)
for fn, sp in lab.items():
    if fn in fmap: by[sp].append(fn)
species = sorted(by.keys()); s2i = {c: i for i, c in enumerate(species)}
C = len(species)
# text emb per seen species (aligned to s2i)
Tseen = torch.stack([TnH[ciAll[c]] for c in species])      # [C, d]

tr_items, va_items, cnt_of = [], [], {}
for sp, fns in by.items():
    fns = sorted(fns); si = s2i[sp]; cnt_of[si] = len(fns)
    if len(fns) >= 3:
        k = max(1, round(0.2 * len(fns))); tr, va = fns[:-k], fns[-k:]
    else:
        tr, va = fns, []
    tr_items += [(fn, si) for fn in tr]; va_items += [(fn, si) for fn in va]

tr_idx = torch.tensor([fmap[fn] for fn, si in tr_items]); tr_lab = torch.tensor([si for fn, si in tr_items])
trF = feats[tr_idx]
protos = torch.zeros(C, feats.shape[1]); c2 = torch.zeros(C)
protos.index_add_(0, tr_lab, trF); c2.index_add_(0, tr_lab, torch.ones(len(tr_lab)))
tr_count = c2.clone()                                       # #train imgs per class (after holdout)
protos = F.normalize(protos / c2.clamp(min=1).unsqueeze(1), dim=1)
va_idx = torch.tensor([fmap[fn] for fn, si in va_items]); vaF = feats[va_idx]
vaY = torch.tensor([si for fn, si in va_items])

def scores(vf):
    psim = vf @ protos.t()
    S = vf @ trF.t()
    cmax = torch.full((vf.shape[0], C), -1.0)
    cmax.scatter_reduce_(1, tr_lab.unsqueeze(0).expand(vf.shape[0], -1), S, reduce='amax', include_self=True)
    tsim = vf @ Tseen.t()
    return psim, cmax, tsim

ps, cm, ts = [], [], []
for s in range(0, vaF.shape[0], 512):
    a, b, c = scores(vaF[s:s+512]); ps.append(a); cm.append(b); ts.append(c)
ps = torch.cat(ps); cm = torch.cat(cm); ts = torch.cat(ts)
blend = ps + 2.0 * cm
N = len(vaY)
cnt_val = tr_count[vaY]    # train-count of each val sample's class

def strat(pred):
    ok = (pred == vaY)
    print(f'    overall {100*ok.float().mean():.2f}   '
          + '  '.join(f'cnt{lo}-{hi}:{100*ok[(cnt_val>=lo)&(cnt_val<=hi)].float().mean():.1f}(n={int(((cnt_val>=lo)&(cnt_val<=hi)).sum())})'
                      for lo, hi in [(1,2),(3,4),(5,9),(10,9999)]))

print('BLEND (deployed):'); strat(blend.argmax(1))
print('\nBLEND + lam*text_sim:')
for lam in (0.5, 1.0, 1.5, 2.0, 3.0):
    strat((blend + lam * ts).argmax(1))
    print(f'      ^ lam={lam}')
print('\ntext-only argmax:'); strat(ts.argmax(1))
print('\nproto-only (NCM):'); strat(ps.argmax(1))
# count-gated text: only add text where train count is low
print('\nCOUNT-GATED text (add lam*text only if class train-count<=T):')
for T in (2, 4):
    for lam in (1.0, 2.0):
        gate = (tr_count <= T).float().unsqueeze(0)        # [1,C]
        strat((blend + lam * gate * ts).argmax(1))
        print(f'      ^ gate<= {T}  lam={lam}')
