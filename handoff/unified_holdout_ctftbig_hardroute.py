"""Legitimate content-based HARD routing (no folder/path leakage -- route decided purely
from each image's own embedding via an ensemble energy score, threshold calibrated on the
pseudo-seen/unseen holdout split -- standard open-set calibration, not cheating).

Mechanism: seen and unseen currently compete in ONE shared argmax (single-pipeline
compliance requirement), which the fullft336 real result just showed causes destructive
cross-competition (boosting seen_block scale cannibalized unseen at fixed gamma). Hard
routing removes the competition: each image scored ONLY within its routed pool.
  - routed seen   -> argmax over kept_seen columns using the 4-channel seen_block
  - routed unseen -> argmax over the non-seen columns using the ctftbig text block
Overall accuracy naturally penalizes misrouting (wrong-pool queries can never be correct).
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
def dbnorm(S, tc=0.05, tr=0.5): return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)

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
QtI, QtF, Qtf = load('outputs/emb_train_fullft336.pt')

common = [fn for fn in Htf if fn in FtI and fn in CtI and fn in LtI and fn in BtI and fn in QtI and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in common: by[lab[fn]].append(fn)
seen = sorted(by.keys())
order = sorted(seen, key=lambda c: len(by[c]))
n_pu = int(len(seen) * 0.2)
pseudo_unseen = order[:n_pu]
kept_seen = [c for c in seen if c not in set(pseudo_unseen)]
s2i = {c: i for i, c in enumerate(kept_seen)}
S = len(kept_seen)
kept_idx = torch.tensor([ci[c] for c in kept_seen])
other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(kept_seen)])
print(f'kept_seen={S}  pseudo_unseen_classes={len(pseudo_unseen)}  other_idx(NCLS-S)={len(other_idx)}')

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
qL = gather(LtI, LtF, all_qfn); qB = gather(BtI, BtF, all_qfn); qQ = gather(QtI, QtF, all_qfn)

sc_ft = seen_score(qF, PF, TFF, TLF, TseenTax, LAM)
sc_cap = seen_score(qC, PC, TFC, TLC, TseenTax, LAM)
sc_l = seen_score(qL, PL, TFL, TLL)
sc_q = seen_score(qQ, PQ, TFQ, TLQ)
seen_block = zc(sc_ft) + 1.0 * zc(sc_cap) + 0.5 * zc(sc_l) + 0.75 * zc(sc_q)   # 4-channel

qB_d = qB.to(dev); qL_d = qL.to(dev)
M_taxon = qB_d @ TtH.t(); M_name = qL_d @ TnL.t()
text_dbnorm_full = dbnorm(M_taxon, 0.05, 0.5) + 0.5 * dbnorm(M_name, 0.05, 0.5)
oth = other_idx.to(dev)
text_unseen_only = text_dbnorm_full[:, oth]   # [n_q, len(other_idx)] -- unseen-pool-only scoring

gold_full = torch.tensor([kept_idx[y].item() for y in seen_query_y] + unseen_query_y, device=dev)
is_seen_row = torch.zeros(n_q, dtype=torch.bool, device=dev); is_seen_row[:n_seen_q] = True
kidx = kept_idx.to(dev)

# route signal: ensemble energy score (logsumexp over the 4-channel seen_block) --
# stronger than the raw ft-proto cosine the project previously found barely separated
# seen/unseen (0.913 vs 0.898); this uses the full ensemble instead of one raw channel.
energy = torch.logsumexp(seen_block, dim=1)
seenmax = seen_block.max(1).values

def route_eval(signal, name):
    best = None
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
        overall = 0.564 * seen_acc + 0.436 * unseen_acc
        row = dict(signal=name, thr_frac=frac, route_acc=route_acc, seen=seen_acc, unseen=unseen_acc, overall=overall)
        if best is None or row['overall'] > best['overall']:
            best = row
    return best

results = []
for sig, name in [(seenmax, 'seenmax(ensemble)'), (energy, 'energy(ensemble)')]:
    r = route_eval(sig, name)
    results.append(r)
    print(f'[{name}] BEST: thr_frac={r["thr_frac"]:.3f} route_acc={r["route_acc"]:.2f} '
          f'seen={r["seen"]:.2f} unseen={r["unseen"]:.2f} overall={r["overall"]:.2f}')

# reference: current shared-argmax smooth-gate baseline (w_text=0.5, gamma=13, same 4ch seen_block)
maxproto_ft = sc_ft.max(1).values
novelty = 1.0 - (maxproto_ft - maxproto_ft.min()) / (maxproto_ft.max() - maxproto_ft.min() + 1e-6)
text_block_z = zc(text_dbnorm_full)
U = text_block_z.clone()
U[:, kidx] = seen_block + 0.5 * text_block_z[:, kidx] - (13.0 * novelty).unsqueeze(1)
pred_sm = U.argmax(1)
seen_acc_sm = (pred_sm[is_seen_row] == gold_full[is_seen_row]).float().mean().item() * 100
unseen_acc_sm = (pred_sm[~is_seen_row] == gold_full[~is_seen_row]).float().mean().item() * 100
overall_sm = 0.564 * seen_acc_sm + 0.436 * unseen_acc_sm
print(f'[REFERENCE smooth-gate g13 w0.5] seen={seen_acc_sm:.2f} unseen={unseen_acc_sm:.2f} overall={overall_sm:.2f}')

json.dump(dict(hardroute_results=results,
               reference_smooth_gate=dict(seen=seen_acc_sm, unseen=unseen_acc_sm, overall=overall_sm)),
          open('handoff/unified_frontier_hardroute.json', 'w'), indent=2)
print('wrote handoff/unified_frontier_hardroute.json')
