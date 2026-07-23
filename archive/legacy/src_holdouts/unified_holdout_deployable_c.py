"""Deployable-now fallback holdout: H+L-only seen-block (no ft/cap -- those aren't
cached for the unseen-eval folder) WITH the variant-c smooth novelty gate.

Same split/gate/oracle structure as unified_holdout.py / unified_holdout_symmetric.py.
seen_block = zc(H-based proto+cmax+taxon) + 0.5*zc(L)   (identical to the symmetric script)
variant c: gamma_row = gamma * novelty, novelty = 1 - minmax(maxproto), maxproto = query's
own max seen-prototype cosine (H-backbone). Reads only the query's own scores, never split
membership -- compliant, and degrades gracefully vs the flat-gamma knife-edge.

This measures the no-extraction ceiling, for comparison against the full ft/cap/L/H c-variant
numbers already in outputs/unified_frontier.json.
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
txtH = torch.load('outputs/text_emb_h.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
assert txtHt['classes'] == classes and txtH['classes'] == classes
ci = {c: i for i, c in enumerate(classes)}
NCLS = len(classes)
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
TnH = F.normalize(txtH['emb_name'].float(), dim=-1).to(dev)

lab = json.load(open(f'{D}/label_train.json'))

HtI, HtF, Htf = load('outputs/emb_train_h.pt')
LtI, LtF, Ltf = load('outputs/emb_train.pt')

common = [fn for fn in Htf if fn in LtI and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in common:
    by[lab[fn]].append(fn)
seen = sorted(by.keys())
print(f'seen classes total: {len(seen)}  train imgs usable: {len(common)}')

order = sorted(seen, key=lambda c: len(by[c]))
n_pu = int(len(seen) * 0.2)
pseudo_unseen = order[:n_pu]
kept_seen = [c for c in seen if c not in set(pseudo_unseen)]
print(f'pseudo_unseen classes: {len(pseudo_unseen)}  kept_seen classes: {len(kept_seen)}')

s2i = {c: i for i, c in enumerate(kept_seen)}
S = len(kept_seen)
kept_idx = torch.tensor([ci[c] for c in kept_seen])
other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(kept_seen)])
TseenTax = TtH[kept_idx.to(dev)]

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


PH, TFH, TLH = protos_and_train(HtI, HtF, kept_seen)
PL, TFL, TLL = protos_and_train(LtI, LtF, kept_seen)
LAM = 4.0


def seen_score(qF, P, TF, TL, Tx=None, lam=0.0, ret_maxproto=False):
    qF = qF.to(dev)
    n = qF.shape[0]
    out = torch.empty(n, S, device=dev)
    ps_all = torch.empty(n, S, device=dev)
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
        ps_all[i:i + 2000] = ps
    if ret_maxproto:
        return out, ps_all.max(1).values
    return out


def gather(FI, FF, fns):
    return torch.stack([FF[FI[fn]] for fn in fns])


all_qfn = seen_query_fns + unseen_query_fns
n_seen_q = len(seen_query_fns)
n_unseen_q = len(unseen_query_fns)
n_q = n_seen_q + n_unseen_q

qH = gather(HtI, HtF, all_qfn)
qL = gather(LtI, LtF, all_qfn)

sc_h, maxproto = seen_score(qH, PH, TFH, TLH, TseenTax, LAM, ret_maxproto=True)
sc_l = seen_score(qL, PL, TFL, TLL)
seen_block = zc(sc_h) + 0.5 * zc(sc_l)   # [n_q, S] symmetric H(+taxon) + 0.5*L, NO ft/cap

qH_d = qH.to(dev)
M_taxon_full = qH_d @ TtH.t()
M_name_full = qH_d @ TnH.t()
text_dbnorm_full = dbnorm(M_taxon_full, 0.05, 0.5) + 0.5 * dbnorm(M_name_full, 0.05, 0.5)

gold_full = torch.tensor([kept_idx[y].item() for y in seen_query_y] + unseen_query_y, device=dev)
is_seen_row = torch.zeros(n_q, dtype=torch.bool, device=dev)
is_seen_row[:n_seen_q] = True


def acc(pred_classidx, mask):
    return (pred_classidx[mask] == gold_full[mask]).float().mean().item() * 100


# ================= SANITY GATE =================
pred_seen_only = kept_idx.to(dev)[seen_block[:n_seen_q].argmax(1)]
oracle_seen_acc = (pred_seen_only == gold_full[:n_seen_q]).float().mean().item() * 100

M_taxon_un = M_taxon_full[n_seen_q:][:, other_idx.to(dev)]
M_name_un = M_name_full[n_seen_q:][:, other_idx.to(dev)]
text_unseen_only = dbnorm(M_taxon_un, 0.05, 0.5) + 0.5 * dbnorm(M_name_un, 0.05, 0.5)
pred_unseen_only = other_idx.to(dev)[text_unseen_only.argmax(1)]
oracle_unseen_acc = (pred_unseen_only == gold_full[n_seen_q:]).float().mean().item() * 100

seen_only_acc = oracle_seen_acc  # identical by construction
pred_full_text_masked_nonkept = other_idx.to(dev)[text_dbnorm_full[n_seen_q:][:, other_idx.to(dev)].argmax(1)]
unseen_only_acc = (pred_full_text_masked_nonkept == gold_full[n_seen_q:]).float().mean().item() * 100

gate_ok = abs(seen_only_acc - oracle_seen_acc) < 1.0 and abs(unseen_only_acc - oracle_unseen_acc) < 1.0
print(f'SANITY(deployable-c): oracle_seen={oracle_seen_acc:.2f} | oracle_unseen={oracle_unseen_acc:.2f} '
      f'unseen_only_masked={unseen_only_acc:.2f} | gate_ok={gate_ok}')

oracle_overall = 0.564 * oracle_seen_acc + 0.436 * oracle_unseen_acc
real_unseen_est = oracle_unseen_acc * 0.55
print(f'oracle_overall_5644={oracle_overall:.2f}  real_unseen_est_x055={real_unseen_est:.2f}')

seen_block_z = seen_block
text_block_z = zc(text_dbnorm_full)
kidx = kept_idx.to(dev)

novelty = 1.0 - (maxproto - maxproto.min()) / (maxproto.max() - maxproto.min() + 1e-6)


def eval_config(w_text, gamma):
    """variant c: smooth novelty gate, gamma_row = gamma * novelty (per-query, own maxproto only)."""
    gamma_row = (gamma * novelty).unsqueeze(1)
    U = text_block_z.clone()
    U[:, kidx] = seen_block_z + w_text * text_block_z[:, kidx] - gamma_row
    pred = U.argmax(1)
    seen_acc = acc(pred, is_seen_row)
    unseen_acc = acc(pred, ~is_seen_row)
    overall = 0.564 * seen_acc + 0.436 * unseen_acc
    return seen_acc, unseen_acc, overall


frontier = []
gammas = [0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50, 70, 100]
w_texts = [0.25, 0.5, 1.0]
for w_text in w_texts:
    for gamma in gammas:
        seen_acc, unseen_acc, overall = eval_config(w_text, gamma)
        frontier.append(dict(variant=f'deployable_c_wtext{w_text}', gamma=gamma, seen=seen_acc,
                              unseen=unseen_acc, overall_5644=overall))
        print(f'[deployable_c w_text={w_text} gamma={gamma:>4}] seen={seen_acc:.2f} unseen={unseen_acc:.2f} '
              f'overall={overall:.2f}', flush=True)

u_seen0, u_unseen0, u_overall0 = eval_config(0.5, 0)

candidates = [f for f in frontier if f['unseen'] >= 0.5 * oracle_unseen_acc]
pool = candidates if candidates else frontier
best = max(pool, key=lambda f: f['overall_5644'])

result = dict(
    sanity=dict(seen_only_acc=seen_only_acc, oracle_seen_acc=oracle_seen_acc,
                unseen_only_acc=unseen_only_acc, oracle_unseen_acc=oracle_unseen_acc,
                passed=gate_ok, detail='deployable H+L-only seen-block (no ft/cap) + variant-c '
                                       'smooth novelty gate -- usable uniformly on all 35665 real eval images'),
    oracle_ref=dict(holdout_seen=oracle_seen_acc, holdout_unseen=oracle_unseen_acc,
                     holdout_overall_5644=oracle_overall, real_unseen_est_x055=real_unseen_est),
    unified_gamma0=dict(seen=u_seen0, unseen=u_unseen0, overall_5644=u_overall0),
    frontier=frontier,
    best_compliant=dict(config=f"variant=deployable_c_wtext{best['variant'].split('wtext')[1]} gamma={best['gamma']}",
                         seen=best['seen'], unseen=best['unseen'], overall_5644=best['overall_5644'],
                         delta_overall_vs_oracle=best['overall_5644'] - oracle_overall,
                         notes='H(+taxon)+0.5L seen-block (no ft/cap) + variant-c smooth novelty gate; '
                               'this is the no-extraction ceiling, for comparison against the full '
                               'ft/cap/L/H c-variant rows in outputs/unified_frontier.json'),
)
import os
os.makedirs('outputs', exist_ok=True)
json.dump(result, open('outputs/unified_frontier_deployable_c.json', 'w'), indent=2)
print('wrote outputs/unified_frontier_deployable_c.json')
print(json.dumps({k: v for k, v in result.items() if k != 'frontier'}, indent=2))
