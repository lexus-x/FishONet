"""Re-validate handoff/unified_holdout_ctftbig_hardroute.py's finding (hard hard-threshold
routing beats the smooth gamma-gate by +4.3pt holdout: seenmax(ensemble) 56.30 vs smooth-gate
51.96, both at a 4-channel seen_block) -- but with THIS session's improvements folded in:
  - seen_block: ft/cap/L + ctftbig (W_B=1.0, capped -- see src/unified_holdout_ctftbig_ext.py)
  - unseen text-block: ctftbig+0.5*L + 0.75*fullft336_v2 (W_U=0.75)
Same pseudo-unseen class-holdout protocol as unified_holdout_ctftbig_ext.py, so numbers are
directly comparable to that file's best_compliant overall_5644=56.76 (smooth gate).

Why hard routing might transfer differently (better OR worse) to real data than the smooth
gate did (which came in at real overall=45.19 vs holdout 56.76, a -11.6pt miss): it removes the
shared-argmax cross-competition, but its threshold is STILL a per-batch relative min/max
calibration (same potential real-vs-holdout instability as the smooth gate's novelty
normalization) -- so this is a real hypothesis to test, not a guaranteed fix.
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


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
txtL = torch.load('outputs/text_emb.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
ci = {c: i for i, c in enumerate(classes)}
NCLS = len(classes)
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)
lab = json.load(open(f'{D}/label_train.json'))

FtI, FtF, Ftf = load('outputs/emb_train_ft.pt')
CtI, CtF, Ctf = load('outputs/emb_train_cap.pt')
LtI, LtF, Ltf = load('outputs/emb_train.pt')
HtI, HtF, Htf = load('outputs/emb_train_h.pt')
BtI, BtF, Btf = load('outputs/emb_train_ctftbig.pt')
FF2I, FF2F, FF2f = load('outputs/emb_train_fullft336_v2.pt')

common = [fn for fn in Htf if fn in FtI and fn in CtI and fn in LtI and fn in BtI and fn in FF2I
          and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in common:
    by[lab[fn]].append(fn)
seen = sorted(by.keys())
order = sorted(seen, key=lambda c: len(by[c]))
n_pu = int(len(seen) * 0.2)
pseudo_unseen = order[:n_pu]
kept_seen = [c for c in seen if c not in set(pseudo_unseen)]
s2i = {c: i for i, c in enumerate(kept_seen)}
S = len(kept_seen)
kept_idx = torch.tensor([ci[c] for c in kept_seen])
other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(kept_seen)])
print(f'kept_seen={S}  pseudo_unseen_classes={len(pseudo_unseen)}  other_idx={len(other_idx)}')

trby = defaultdict(list)
seen_query_fns, seen_query_y = [], []
for c in kept_seen:
    fns = sorted(by[c])
    if len(fns) >= 3:
        k = max(1, round(0.2 * len(fns)))
        for f in fns[:-k]:
            trby[c].append(f)
        for f in fns[-k:]:
            seen_query_fns.append(f)
            seen_query_y.append(s2i[c])
    else:
        for f in fns:
            trby[c].append(f)
unseen_query_fns, unseen_query_y = [], []
for c in pseudo_unseen:
    for f in by[c]:
        unseen_query_fns.append(f)
        unseen_query_y.append(ci[c])
print(f'seen queries: {len(seen_query_fns)}  unseen(pseudo) queries: {len(unseen_query_fns)}')


def protos_and_train(FI, FF, order):
    P = torch.zeros(S, FF.shape[1])
    cnt = torch.zeros(S)
    TF, TL = [], []
    for c in order:
        for fn in trby[c]:
            f = FF[FI[fn]]
            P[s2i[c]] += f
            cnt[s2i[c]] += 1
            TF.append(f)
            TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev)


PF, TFF, TLF = protos_and_train(FtI, FtF, kept_seen)
PC, TFC, TLC = protos_and_train(CtI, CtF, kept_seen)
PL, TFL, TLL = protos_and_train(LtI, LtF, kept_seen)
PB, TFB, TLB = protos_and_train(BtI, BtF, kept_seen)
TseenTax = TtH[kept_idx.to(dev)]
LAM = 4.0


def seen_score(qF, P, TF, TL, Tx=None, lam=0.0):
    qF = qF.to(dev)
    n = qF.shape[0]
    out = torch.empty(n, S, device=dev)
    for i in range(0, n, 2000):
        e = qF[i:i + 2000]
        ps = e @ P.t()
        sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + 2.0 * cmax
        if Tx is not None and lam > 0:
            sc = sc + lam * (e @ Tx.t())
        out[i:i + 2000] = sc
    return out


def gather(FI, FF, fns):
    return torch.stack([FF[FI[fn]] for fn in fns])


all_qfn = seen_query_fns + unseen_query_fns
n_seen_q = len(seen_query_fns)
n_q = len(all_qfn)
qF = gather(FtI, FtF, all_qfn)
qC = gather(CtI, CtF, all_qfn)
qL = gather(LtI, LtF, all_qfn)
qB = gather(BtI, BtF, all_qfn)
qFF2 = gather(FF2I, FF2F, all_qfn)

sc_ft = seen_score(qF, PF, TFF, TLF, TseenTax, LAM)
sc_cap = seen_score(qC, PC, TFC, TLC, TseenTax, LAM)
sc_l = seen_score(qL, PL, TFL, TLL)
sc_b = seen_score(qB, PB, TFB, TLB, TseenTax, LAM)
W_B = 1.0
seen_block = zc(sc_ft) + 1.0 * zc(sc_cap) + 0.5 * zc(sc_l) + W_B * zc(sc_b)

qB_d = qB.to(dev)
qL_d = qL.to(dev)
qFF2_d = qFF2.to(dev)
M_taxon = qB_d @ TtH.t()
M_name = qL_d @ TnL.t()
M_taxon_FF2 = qFF2_d @ TtH.t()
W_U = 0.75
text_dbnorm_full = (dbnorm(M_taxon, 0.05, 0.5) + 0.5 * dbnorm(M_name, 0.05, 0.5)
                     + W_U * dbnorm(M_taxon_FF2, 0.05, 0.5))
oth = other_idx.to(dev)
text_unseen_only = text_dbnorm_full[:, oth]

gold_full = torch.tensor([kept_idx[y].item() for y in seen_query_y] + unseen_query_y, device=dev)
is_seen_row = torch.zeros(n_q, dtype=torch.bool, device=dev)
is_seen_row[:n_seen_q] = True
kidx = kept_idx.to(dev)

energy = torch.logsumexp(seen_block, dim=1)
seenmax = seen_block.max(1).values


def route_eval(signal, name):
    best = None
    rows = []
    lo, hi = signal.min().item(), signal.max().item()
    for frac in [i / 40 for i in range(1, 40)]:
        thr = lo + frac * (hi - lo)
        route_seen = signal >= thr
        route_acc = (route_seen == is_seen_row).float().mean().item() * 100
        pred = torch.full((n_q,), -1, dtype=torch.long, device=dev)
        if route_seen.any():
            pred[route_seen] = kidx[seen_block[route_seen].argmax(1)]
        if (~route_seen).any():
            pred[~route_seen] = oth[text_unseen_only[~route_seen].argmax(1)]
        seen_acc = (pred[is_seen_row] == gold_full[is_seen_row]).float().mean().item() * 100
        unseen_acc = (pred[~is_seen_row] == gold_full[~is_seen_row]).float().mean().item() * 100
        overall = 0.5635 * seen_acc + 0.4365 * unseen_acc
        row = dict(signal=name, thr_frac=frac, route_acc=route_acc, seen=seen_acc, unseen=unseen_acc, overall=overall)
        rows.append(row)
        if best is None or row['overall'] > best['overall']:
            best = row
    return best, rows


results = {}
for sig, name in [(seenmax, 'seenmax(ensemble)'), (energy, 'energy(ensemble)')]:
    best, rows = route_eval(sig, name)
    results[name] = dict(best=best, rows=rows)
    print(f'[{name}] BEST: thr_frac={best["thr_frac"]:.3f} route_acc={best["route_acc"]:.2f} '
          f'seen={best["seen"]:.2f} unseen={best["unseen"]:.2f} overall={best["overall"]:.2f}')

print(f'\nreference (this session smooth-gate best_compliant): overall=56.76 (seen=85.10 unseen=20.10, w_text=0.5 gamma=30)')
print(f'reference (Jul16 hardroute, older 4ch seen_block, no ctftbig/fullft336_v2 ext): overall=56.30')

json.dump(results, open('outputs/unified_frontier_hardroute_ext.json', 'w'), indent=2)
print('wrote outputs/unified_frontier_hardroute_ext.json')
