"""Does adding a TTA-H channel to v20's seen ensemble (ft+1.0cap+0.5L) raise the holdout?
Locally provable check only — CPU. Prints v20 baseline then sweeps the TTA-H weight.
If holdout doesn't move, TTA only helps distribution shift (invisible here), which is a
submission-only bet; if it rises, it's a real win worth one submission.
"""
import json, torch
from collections import defaultdict
import torch.nn.functional as F
torch.set_num_threads(8); LAM = 4.0

def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])
def zc(M): return (M - M.mean()) / (M.std() + 1e-6)

txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
txtL = torch.load('outputs/text_emb.pt', weights_only=False)
classes = txtHt['classes']; ci = {c: i for i, c in enumerate(classes)}
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1)
lab = json.load(open('data/dl/label_train.json'))

FtI, FtF, Ftf = load('outputs/emb_train_ft.pt')
CtI, CtF, _ = load('outputs/emb_train_cap.pt')
LtI, LtF, _ = load('outputs/emb_train.pt')
HtI, HtF, _ = load('outputs/emb_train_h_tta.pt')
trainfiles = [fn for fn in Ftf if fn in LtI and fn in CtI and fn in HtI and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in trainfiles: by[lab[fn]].append(fn)
seen = sorted(by.keys()); s2i = {c: i for i, c in enumerate(seen)}; S = len(seen)
TseenTax = torch.stack([TtH[ci[c]] for c in seen])

def protos_and_train(TrI, TrF, fns_by_cls, order):
    P = torch.zeros(S, TrF.shape[1]); cnt = torch.zeros(S); TF = []; TL = []
    for c in order:
        for fn in fns_by_cls[c]:
            f = TrF[TrI[fn]]; P[s2i[c]] += f; cnt[s2i[c]] += 1; TF.append(f); TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1); return P, torch.stack(TF), torch.tensor(TL)

def seen_score(qF, P, TF, TL, Tx=None, lam=0.0):
    out = torch.empty(qF.shape[0], S)
    for i in range(0, qF.shape[0], 1000):
        e = qF[i:i + 1000]; ps = e @ P.t(); sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9); cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + 2.0 * cmax
        if Tx is not None and lam > 0: sc = sc + lam * (e @ Tx.t())
        out[i:i + 1000] = sc
    return out

# holdout split (same as v20)
trby = defaultdict(list); valrows = []
for c in seen:
    fns = sorted(by[c])
    if len(fns) >= 3:
        k = max(1, round(0.2 * len(fns)))
        for f in fns[:-k]: trby[c].append(f)
        for f in fns[-k:]: valrows.append((f, s2i[c]))
    else:
        for f in fns: trby[c].append(f)
VY = torch.tensor([y for _, y in valrows])

PFh, TFFh, TLFh = protos_and_train(FtI, FtF, trby, seen)
PCh, TFCh, TLCh = protos_and_train(CtI, CtF, trby, seen)
PLh, TFLh, TLLh = protos_and_train(LtI, LtF, trby, seen)
PHh, TFHh, TLHh = protos_and_train(HtI, HtF, trby, seen)
vF = torch.stack([FtF[FtI[f]] for f, _ in valrows]); vC = torch.stack([CtF[CtI[f]] for f, _ in valrows])
vL = torch.stack([LtF[LtI[f]] for f, _ in valrows]); vH = torch.stack([HtF[HtI[f]] for f, _ in valrows])
sF = zc(seen_score(vF, PFh, TFFh, TLFh, TseenTax, LAM))
sC = zc(seen_score(vC, PCh, TFCh, TLCh, TseenTax, LAM))
sL = zc(seen_score(vL, PLh, TFLh, TLLh))
sH = zc(seen_score(vH, PHh, TFHh, TLHh, TseenTax, LAM))

base = (sF + 1.0*sC + 0.5*sL).argmax(1).eq(VY).float().mean().item()*100
print(f'v20 baseline (ft+1.0cap+0.5L): {base:.2f}%  (expect ~87.94)')
print('adding TTA-H channel at weight w:')
best = (base, 0.0)
for w in [0.25, 0.5, 0.75, 1.0, 1.5]:
    acc = (sF + 1.0*sC + 0.5*sL + w*sH).argmax(1).eq(VY).float().mean().item()*100
    flag = '  <-- beats base' if acc > base else ''
    print(f'  w={w:<4}: {acc:.2f}%{flag}')
    if acc > best[0]: best = (acc, w)
print(f'\nBEST: {best[0]:.2f}% at TTA-H w={best[1]}  (delta {best[0]-base:+.2f})')
