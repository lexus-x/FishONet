"""Same as unified_holdout_ctftbig_4ch.py but using the fullft336_v2 checkpoint
(8-epoch retrain, standalone val NCM=84.67% vs v1's 83.60%) instead of v1. Compares
against v1's known result (+0.33pt holdout at w=0.75, baseline oracle_seen=88.29,
handoff/unified_frontier_4ch_fullft336.json).
"""
import json, pickle, torch
from collections import defaultdict
import torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'

def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])

def zc(M): return (M - M.mean()) / (M.std() + 1e-6)

txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
ci = {c: i for i, c in enumerate(classes)}
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
lab = json.load(open(f'{D}/label_train.json'))

FtI, FtF, Ftf = load('outputs/emb_train_ft.pt')
CtI, CtF, Ctf = load('outputs/emb_train_cap.pt')
LtI, LtF, Ltf = load('outputs/emb_train.pt')
HtI, HtF, Htf = load('outputs/emb_train_h.pt')
QtI, QtF, Qtf = load('outputs/emb_train_fullft336_v2.pt')

common = [fn for fn in Htf if fn in FtI and fn in CtI and fn in LtI and fn in QtI and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in common: by[lab[fn]].append(fn)
seen = sorted(by.keys())
print(f'seen classes total: {len(seen)}  train imgs usable: {len(common)}')

order = sorted(seen, key=lambda c: len(by[c]))
n_pu = int(len(seen) * 0.2)
pseudo_unseen = order[:n_pu]
kept_seen = [c for c in seen if c not in set(pseudo_unseen)]
s2i = {c: i for i, c in enumerate(kept_seen)}
S = len(kept_seen)
kept_idx = torch.tensor([ci[c] for c in kept_seen])

trby = defaultdict(list)
seen_query_fns, seen_query_y = [], []
for c in kept_seen:
    fns = sorted(by[c])
    if len(fns) >= 3:
        k = max(1, round(0.2 * len(fns)))
        for f in fns[:-k]: trby[c].append(f)
        for f in fns[-k:]:
            seen_query_fns.append(f); seen_query_y.append(s2i[c])
    else:
        for f in fns: trby[c].append(f)

unseen_query_fns, unseen_query_y = [], []
for c in pseudo_unseen:
    for f in by[c]:
        unseen_query_fns.append(f); unseen_query_y.append(ci[c])
print(f'seen queries: {len(seen_query_fns)}  unseen(pseudo) queries: {len(unseen_query_fns)}')

def protos_and_train(FI, FF, order):
    P = torch.zeros(S, FF.shape[1]); cnt = torch.zeros(S); TF, TL = [], []
    for c in order:
        for fn in trby[c]:
            f = FF[FI[fn]]; P[s2i[c]] += f; cnt[s2i[c]] += 1; TF.append(f); TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev)

PF, TFF, TLF = protos_and_train(FtI, FtF, kept_seen)
PC, TFC, TLC = protos_and_train(CtI, CtF, kept_seen)
PL, TFL, TLL = protos_and_train(LtI, LtF, kept_seen)
PQ, TFQ, TLQ = protos_and_train(QtI, QtF, kept_seen)
TseenTax = TtH[kept_idx.to(dev)]
LAM = 4.0

def seen_score(qF, P, TF, TL, Tx=None, lam=0.0):
    qF = qF.to(dev); n = qF.shape[0]; out = torch.empty(n, S, device=dev)
    for i in range(0, n, 2000):
        e = qF[i:i + 2000]; ps = e @ P.t(); sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + 2.0 * cmax
        if Tx is not None and lam > 0: sc = sc + lam * (e @ Tx.t())
        out[i:i + 2000] = sc
    return out

def gather(FI, FF, fns): return torch.stack([FF[FI[fn]] for fn in fns])

all_qfn = seen_query_fns + unseen_query_fns
n_seen_q = len(seen_query_fns); n_q = len(all_qfn)
qF = gather(FtI, FtF, all_qfn); qC = gather(CtI, CtF, all_qfn)
qL = gather(LtI, LtF, all_qfn); qQ = gather(QtI, QtF, all_qfn)

sc_ft = seen_score(qF, PF, TFF, TLF, TseenTax, LAM)
sc_cap = seen_score(qC, PC, TFC, TLC, TseenTax, LAM)
sc_l = seen_score(qL, PL, TFL, TLL)
sc_q = seen_score(qQ, PQ, TFQ, TLQ)

gold_full = torch.tensor([kept_idx[y].item() for y in seen_query_y] + unseen_query_y, device=dev)
kidx = kept_idx.to(dev)
def oracle_seen(block):
    pred = kidx[block[:n_seen_q].argmax(1)]
    return (pred == gold_full[:n_seen_q]).float().mean().item() * 100
def se_of(p, n):
    p/=100.0; return ((p*(1-p)/max(n,1))**0.5)*100

base = zc(sc_ft) + 1.0*zc(sc_cap) + 0.5*zc(sc_l)
base_acc = oracle_seen(base)
se = se_of(base_acc, n_seen_q)
print(f'BASELINE (ft+1.0cap+0.5L) [v2 checkpoint] oracle_seen={base_acc:.2f}  SE={se:.2f}  n={n_seen_q}')

results = [dict(w_q=0.0, oracle_seen=base_acc, delta=0.0, se=se)]
for w_q in [0.25, 0.5, 0.75, 1.0, 1.5]:
    blk = base + w_q * zc(sc_q)
    acc = oracle_seen(blk)
    results.append(dict(w_q=w_q, oracle_seen=acc, delta=acc-base_acc, se=se))
    print(f'[+fullft336_v2 w={w_q}] oracle_seen={acc:.2f}  delta={acc-base_acc:+.2f}  SE={se:.2f}')

best = max(results, key=lambda r: r['oracle_seen'])
print('BEST:', best)
json.dump(dict(baseline=base_acc, se=se, n_seen_q=n_seen_q, results=results, best=best),
          open('handoff/unified_frontier_4ch_fullft336_v2.json', 'w'), indent=2)
print('wrote handoff/unified_frontier_4ch_fullft336_v2.json')
