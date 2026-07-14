"""Deployed-config unseen A/B: db(H_*)+0.5*db(L_name) on the hard-sim split.
Confirms whether swapping H name->taxon helps the ACTUAL ensemble used in predict_v07."""
import json, pickle, torch
from collections import defaultdict
import torch.nn.functional as F

D = 'data/dl'
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
ci = {c: i for i, c in enumerate(classes)}
TnH = F.normalize(torch.load('outputs/text_emb_h.pt', weights_only=False)['emb_name'].float(), dim=-1)
TtH = F.normalize(torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1)
TnL = F.normalize(torch.load('outputs/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1)

def loadf(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1)
HI, HF = loadf('outputs/emb_train_h.pt')
LI, LF = loadf('outputs/emb_train.pt')
lab = json.load(open(f'{D}/label_train.json'))

by = defaultdict(list)
common = [fn for fn in HI if fn in LI and fn in lab and lab[fn] in ci]
for fn in common: by[lab[fn]].append(fn)
seen = sorted(by.keys())
order = sorted(seen, key=lambda c: len(by[c]))
n_un = int(len(seen) * 0.2)
pseudo = set(order[:n_un]); known = set(c for c in seen if c not in pseudo)
nonk = torch.tensor([i for i, c in enumerate(classes) if c not in known])
cand_pos = {nonk[j].item(): j for j in range(len(nonk))}
qfn, qy = [], []
for c in pseudo:
    for fn in by[c]: qfn.append(fn); qy.append(ci[c])
qH = torch.stack([HF[HI[fn]] for fn in qfn]); qL = torch.stack([LF[LI[fn]] for fn in qfn])
gold = torch.tensor([cand_pos[y] for y in qy])
print(f'queries={len(qy)} candidates={len(nonk)}')

def db(M): return (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-6)
def t1(S): return (S.argmax(1) == gold).float().mean().item() * 100

sHn = db(qH @ TnH[nonk].t()); sHt = db(qH @ TtH[nonk].t()); sL = db(qL @ TnL[nonk].t())
print('\n=== DEPLOYED ensemble (hard-sim) ===')
print(f'  H_name  alone        : {t1(sHn):.2f}')
print(f'  H_taxon alone        : {t1(sHt):.2f}')
print(f'  H_name  + 0.5 L_name : {t1(sHn + 0.5*sL):.2f}   <-- current v07')
print(f'  H_taxon + 0.5 L_name : {t1(sHt + 0.5*sL):.2f}   <-- v09 candidate')
for w in (0.3, 0.5, 0.7):
    print(f'  H_taxon + {w} L_name   : {t1(sHt + w*sL):.2f}')
# blend H name+taxon
for a in (0.3, 0.5):
    sHb = db(qH @ ((1-a)*TnH + a*TtH)[nonk].t())
    print(f'  H(name+{a}taxon)+0.5L : {t1(sHb + 0.5*sL):.2f}')
print('\nReal calibration: real_unseen ~ 0.5 * sim')
