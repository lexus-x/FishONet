"""SEEN-route holdout eval for the heavy-aug shift-robust checkpoint (ft_robust_lora.pt),
using the EXACT same per-class holdout + blend formula (proto + 2*class_max + LAM*taxon_text)
as predict_v12.py's own holdout print, so the number is directly comparable to the deployed
light-aug model's holdout (87.22% per project notes). Only the feature source file differs
(emb_train_cap.pt instead of emb_train_ft.pt).

  python src/eval_ft_robust_holdout.py
"""
import json, torch
from collections import defaultdict
import torch.nn.functional as F
torch.set_num_threads(8); LAM = 4.0


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M): return (M - M.mean()) / (M.std() + 1e-6)


txtH = torch.load('outputs/text_emb_h.pt', weights_only=False)
txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
classes = txtH['classes']; ci = {c: i for i, c in enumerate(classes)}
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1)
lab = json.load(open('data/dl/label_train.json'))

FtrI, FtrF, Ftrf = load('outputs/emb_train_cap.pt')
LtrI, LtrF, _ = load('outputs/emb_train.pt')
trainfiles = [fn for fn in Ftrf if fn in LtrI and fn in lab and lab[fn] in ci]
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


trby = defaultdict(list); valrows = []
for c in seen:
    fns = sorted(by[c])
    if len(fns) >= 3:
        k = max(1, round(0.2 * len(fns)))
        for f in fns[:-k]: trby[c].append(f)
        for f in fns[-k:]: valrows.append((f, s2i[c]))
    else:
        for f in fns: trby[c].append(f)

PH, TFH, TLH = protos_and_train(FtrI, FtrF, trby, seen)
PL, TFL, TLL = protos_and_train(LtrI, LtrF, trby, seen)
VY = torch.tensor([y for _, y in valrows])
vqH = torch.stack([FtrF[FtrI[f]] for f, _ in valrows])
vqL = torch.stack([LtrF[LtrI[f]] for f, _ in valrows])
sHt = seen_score(vqH, PH, TFH, TLH, TseenTax, LAM)
sL = seen_score(vqL, PL, TFL, TLL)


def a(M): return (M.argmax(1) == VY).float().mean().item() * 100


a_fth = a(zc(sHt)); a_ens = a(zc(sHt) + 0.5 * zc(sL)); best = max(a_fth, a_ens)
print(f'ft_robust holdout: FT-H alone {a_fth:.2f}%  FT-H+0.5L ensemble {a_ens:.2f}%  best {best:.2f}%')
print('compare to deployed light-aug (v09/v12) holdout: 87.06% (FT-H alone) / 87.22% (ensemble)')

