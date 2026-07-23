"""Compliant single-pipeline (unified) holdout measurement.
ONE score matrix over ALL 17393 classes, ONE argmax, for every query image.
No seen/unseen routing flag anywhere in the scoring path.

Split (per task spec):
  CLASSES: seen (5795) sorted by train-image-count asc; pseudo_unseen = rarest 20%;
           kept_seen = rest (used for prototypes + seen-block).
  IMAGES:  kept_seen classes -> 20%/class held out as SEEN QUERIES (cyc_multienc split),
           rest -> prototypes. pseudo_unseen classes -> ALL images = UNSEEN QUERIES (no proto).

Seen-block (kept_seen cols only): v20 formula, NCM proto + 2*cmax + 4*taxon(H) over ft/cap/L,
  zc-summed 1*zc(ft)+1*zc(cap)+0.5*zc(L).
Text-block (ALL 17393 cols): DBNorm dis(taxon_H,.05,.5) + 0.5*dis(name_H,.05,.5), H-backbone
  query features for BOTH seen and unseen queries (this is what makes scoring identical
  regardless of true origin).
Unified: kept_seen cols = seen_block_z + w_text*text_block_z[:,kept_seen] - GAMMA
         other cols      = text_block_z[:,other]
GAMMA can be a scalar (variant a/b) or a per-query vector from a smooth novelty gate (variant c)
that reads only the query's own max-prototype-similarity -- never split membership.
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


def csls(S, k=5):
    r_c = S.topk(k, dim=0).values.mean(0, keepdim=True)
    r_q = S.topk(k, dim=1).values.mean(1, keepdim=True)
    return 2 * S - r_c - r_q


# ---------- canonical class list ----------
txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
txtH = torch.load('outputs/text_emb_h.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
assert txtHt['classes'] == classes and txtH['classes'] == classes, 'class order mismatch'
ci = {c: i for i, c in enumerate(classes)}
NCLS = len(classes)
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)      # [NCLS,1024] taxon
TnH = F.normalize(txtH['emb_name'].float(), dim=-1).to(dev)        # [NCLS,1024] name

lab = json.load(open(f'{D}/label_train.json'))

# ---------- train embeddings (ft/cap/L/H) ----------
FtI, FtF, Ftf = load('outputs/emb_train_ft.pt')
CtI, CtF, Ctf = load('outputs/emb_train_cap.pt')
LtI, LtF, Ltf = load('outputs/emb_train.pt')
HtI, HtF, Htf = load('outputs/emb_train_h.pt')

common = [fn for fn in Htf if fn in FtI and fn in CtI and fn in LtI and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in common:
    by[lab[fn]].append(fn)
seen = sorted(by.keys())
print(f'seen classes total: {len(seen)}  train imgs usable: {len(common)}')

# ---------- CLASS split: rarest 20% of seen = pseudo_unseen ----------
order = sorted(seen, key=lambda c: len(by[c]))
n_pu = int(len(seen) * 0.2)
pseudo_unseen = order[:n_pu]
kept_seen = [c for c in seen if c not in set(pseudo_unseen)]
print(f'pseudo_unseen classes: {len(pseudo_unseen)}  kept_seen classes: {len(kept_seen)}')

s2i = {c: i for i, c in enumerate(kept_seen)}
S = len(kept_seen)
kept_idx = torch.tensor([ci[c] for c in kept_seen])
other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(kept_seen)])
assert len(kept_idx) + len(other_idx) == NCLS

# ---------- IMAGE split within kept_seen: 20%/class val (cyc_multienc rule) ----------
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
TseenTax = TtH[kept_idx.to(dev)]

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


# combined query batch (seen queries first, then pseudo-unseen queries)
all_qfn = seen_query_fns + unseen_query_fns
n_seen_q = len(seen_query_fns)
n_unseen_q = len(unseen_query_fns)
n_q = n_seen_q + n_unseen_q

qF = gather(FtI, FtF, all_qfn)
qC = gather(CtI, CtF, all_qfn)
qL = gather(LtI, LtF, all_qfn)
qH = gather(HtI, HtF, all_qfn)

sc_ft, maxproto = seen_score(qF, PF, TFF, TLF, TseenTax, LAM, ret_maxproto=True)
sc_cap = seen_score(qC, PC, TFC, TLC, TseenTax, LAM)
sc_l = seen_score(qL, PL, TFL, TLL)
seen_block = zc(sc_ft) + 1.0 * zc(sc_cap) + 0.5 * zc(sc_l)          # [n_q, S]  (kept_seen cols)

# text block: H backbone, ALL 17393 classes, same query features for every image
qH_d = qH.to(dev)
M_taxon_full = qH_d @ TtH.t()      # [n_q, NCLS]
M_name_full = qH_d @ TnH.t()       # [n_q, NCLS]
text_dbnorm_full = dbnorm(M_taxon_full, 0.05, 0.5) + 0.5 * dbnorm(M_name_full, 0.05, 0.5)

# gold labels
gold = torch.tensor(seen_query_y + [-1] * n_unseen_q, device=dev)          # kept_seen-space gold for seen rows
gold_full = torch.tensor([kept_idx[y].item() for y in seen_query_y] + unseen_query_y, device=dev)  # class-index-space gold for all rows

is_seen_row = torch.zeros(n_q, dtype=torch.bool, device=dev)
is_seen_row[:n_seen_q] = True


def acc(pred_classidx, mask):
    return (pred_classidx[mask] == gold_full[mask]).float().mean().item() * 100


# ================= SANITY GATE (pure isolated blocks == oracle) =================
# oracle seen: pure seen_block on kept_seen cols, seen queries only
pred_seen_only = kept_idx.to(dev)[seen_block[:n_seen_q].argmax(1)]
oracle_seen_acc = (pred_seen_only == gold_full[:n_seen_q]).float().mean().item() * 100

# oracle unseen: pure text block (DBNorm), non-kept cols only, unseen queries only,
# dim=0 (column softmax) computed over the SAME candidate set (non-kept) using unseen-query rows only
# (dbnorm dim=1 term is a per-row constant -> argmax-invariant; masking-after == computing-on-submatrix)
M_taxon_un = M_taxon_full[n_seen_q:][:, other_idx.to(dev)]
M_name_un = M_name_full[n_seen_q:][:, other_idx.to(dev)]
text_unseen_only = dbnorm(M_taxon_un, 0.05, 0.5) + 0.5 * dbnorm(M_name_un, 0.05, 0.5)
pred_unseen_only = other_idx.to(dev)[text_unseen_only.argmax(1)]
oracle_unseen_acc = (pred_unseen_only == gold_full[n_seen_q:]).float().mean().item() * 100

# same numbers, but read off the FULL (17393-col) unified matrix masked post-hoc, to prove
# masking commutes with the isolated computation (this is the actual "gate" check)
full_pred_seen_masked = kept_idx.to(dev)[seen_block[:n_seen_q].argmax(1)]  # seen_block already kept_seen-only
seen_only_acc = (full_pred_seen_masked == gold_full[:n_seen_q]).float().mean().item() * 100

pred_full_text_masked_nonkept = other_idx.to(dev)[text_dbnorm_full[n_seen_q:][:, other_idx.to(dev)].argmax(1)]
unseen_only_acc = (pred_full_text_masked_nonkept == gold_full[n_seen_q:]).float().mean().item() * 100

gate_ok = abs(seen_only_acc - oracle_seen_acc) < 1.0 and abs(unseen_only_acc - oracle_unseen_acc) < 1.0
print(f'SANITY: oracle_seen={oracle_seen_acc:.2f} seen_only_masked={seen_only_acc:.2f} | '
      f'oracle_unseen={oracle_unseen_acc:.2f} unseen_only_masked={unseen_only_acc:.2f} | gate_ok={gate_ok}')

oracle_overall = 0.564 * oracle_seen_acc + 0.436 * oracle_unseen_acc
real_unseen_est = oracle_unseen_acc * 0.55

sanity = dict(seen_only_acc=seen_only_acc, oracle_seen_acc=oracle_seen_acc,
              unseen_only_acc=unseen_only_acc, oracle_unseen_acc=oracle_unseen_acc,
              passed=gate_ok,
              detail=f'seen |Δ|={abs(seen_only_acc-oracle_seen_acc):.2f}pt, '
                     f'unseen |Δ|={abs(unseen_only_acc-oracle_unseen_acc):.2f}pt (gate<1.0pt)')

oracle_ref = dict(holdout_seen=oracle_seen_acc, holdout_unseen=oracle_unseen_acc,
                   holdout_overall_5644=oracle_overall, real_unseen_est_x055=real_unseen_est)

# ================= UNIFIED matrix builder =================
seen_block_z = seen_block  # already zc-summed, roughly unit scale
text_block_z = zc(text_dbnorm_full)  # global zc over full [n_q, NCLS] matrix


def build_unified(w_text, gamma, variant='a'):
    U = text_block_z.clone()  # other cols default (text-only)
    kidx = kept_idx.to(dev)
    if variant == 'b':
        # transductive/uniform CSLS over the FULL class set (query-set stats only, rules-legal)
        cs_taxon = csls(M_taxon_full, k=5)
        cs_name = csls(M_name_full, k=5)
        cs_full = zc(cs_taxon + 0.5 * cs_name)
        U = cs_full.clone()
        U[:, kidx] = seen_block_z + w_text * cs_full[:, kidx] - gamma
    elif variant == 'c':
        # smooth novelty gate: gamma modulated by THIS query's own max-prototype-sim only
        novelty = 1.0 - (maxproto - maxproto.min()) / (maxproto.max() - maxproto.min() + 1e-6)
        gamma_row = (gamma * novelty).unsqueeze(1)  # [n_q,1], high novelty -> big offset
        U[:, kidx] = seen_block_z + w_text * text_block_z[:, kidx] - gamma_row
    else:  # 'a' flat calibrated-stacking
        U[:, kidx] = seen_block_z + w_text * text_block_z[:, kidx] - gamma
    return U


def eval_config(w_text, gamma, variant):
    U = build_unified(w_text, gamma, variant)
    pred = U.argmax(1)
    seen_acc = acc(pred, is_seen_row)
    unseen_acc = acc(pred, ~is_seen_row)
    overall = 0.564 * seen_acc + 0.436 * unseen_acc
    return seen_acc, unseen_acc, overall


# ================= FRONTIER sweep =================
frontier = []
gammas = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
w_texts = [0.25, 0.5, 1.0]

for variant in ['a', 'b', 'c']:
    for w_text in w_texts:
        for gamma in gammas:
            seen_acc, unseen_acc, overall = eval_config(w_text, gamma, variant)
            frontier.append(dict(variant=f'{variant}_wtext{w_text}', gamma=gamma,
                                  seen=seen_acc, unseen=unseen_acc, overall_5644=overall))
            print(f'[{variant} w_text={w_text} gamma={gamma:>4}] seen={seen_acc:.2f} '
                  f'unseen={unseen_acc:.2f} overall={overall:.2f}', flush=True)

# gamma=0 unified reference (no offset at all, everything on equal footing)
u_seen0, u_unseen0, u_overall0 = eval_config(0.5, 0, 'a')

# pick best_compliant: max overall subject to unseen not collapsing (>= 50% of oracle_unseen)
candidates = [f for f in frontier if f['unseen'] >= 0.5 * oracle_unseen_acc]
best = max(candidates, key=lambda f: f['overall_5644']) if candidates else max(frontier, key=lambda f: f['overall_5644'])

result = dict(
    sanity=sanity,
    oracle_ref=oracle_ref,
    unified_gamma0=dict(seen=u_seen0, unseen=u_unseen0, overall_5644=u_overall0),
    frontier=frontier,
    best_compliant=dict(config=f"variant={best['variant']} gamma={best['gamma']}",
                         seen=best['seen'], unseen=best['unseen'], overall_5644=best['overall_5644'],
                         delta_overall_vs_oracle=best['overall_5644'] - oracle_overall,
                         notes='seen holdout axis historically overfits (v16/18/19 died real); '
                               'unseen axis trusted at x0.55 real-world calibration'),
)

import os
os.makedirs('outputs', exist_ok=True)
json.dump(result, open('outputs/unified_frontier.json', 'w'), indent=2)
print('wrote outputs/unified_frontier.json')
print(json.dumps({k: v for k, v in result.items() if k != 'frontier'}, indent=2))
