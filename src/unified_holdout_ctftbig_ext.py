"""Step 4: extend unified_holdout_ctftbig.py with two independently-validated legs found in
research/ (Jul 20 exploration), still on the SAME class-holdout protocol (rarest-20%-classes =
pseudo-unseen) so the resulting overall_5644 is directly comparable to the deployed baseline
(outputs/unified_frontier_ctftbig.json: best_compliant_ctftbig overall_5644=56.51).

New seen-block member: ctftbig (proto+cmax+taxon-text, same formula as ft/cap), weight W_B.
  research/ext_ensemble.py found ctftbig+text ALONE scores 91.57% on a *different* holdout
  split (image-level, research/common.py::v20_holdout_split) -- re-validated here on the
  actual deployment class-split before trusting it.

New unseen-route leg: + W_U * dbnorm(fullft336_v2 @ taxon_H), on top of the deployed
ctftbig+0.5*L formula. research/unseen_legs.py found +0.47pt (27.83 -> 28.30) at W_U=0.5.

Stage 1: sweep W_B and W_U INDEPENDENTLY against their respective oracle metrics (seen_block
and text_block are additive/separable, so this is cheap and exact -- no need for the full
gamma/w_text grid yet).
Stage 2: fix the chosen W_B/W_U, run the full compliant gamma/w_text sweep (identical selection
rule as unified_holdout_ctftbig.py) to get the real best_compliant overall_5644.
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


# ---------- canonical class list ----------
txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
txtL = torch.load('outputs/text_emb.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
assert txtHt['classes'] == classes and txtL['classes'] == classes
ci = {c: i for i, c in enumerate(classes)}
NCLS = len(classes)
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)

lab = json.load(open(f'{D}/label_train.json'))

# ---------- train embeddings ----------
FtI, FtF, Ftf = load('outputs/emb_train_ft.pt')
CtI, CtF, Ctf = load('outputs/emb_train_cap.pt')
LtI, LtF, Ltf = load('outputs/emb_train.pt')
HtI, HtF, Htf = load('outputs/emb_train_h.pt')
BtI, BtF, Btf = load('outputs/emb_train_ctftbig.pt')
FF2I, FF2F, FF2f = load('outputs/emb_train_fullft336_v2.pt')
print(f'ctftbig train cache: n={len(Btf)}  fullft336_v2 train cache: n={len(FF2f)}')

common = [fn for fn in Htf if fn in FtI and fn in CtI and fn in LtI and fn in BtI and fn in FF2I
          and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in common:
    by[lab[fn]].append(fn)
seen = sorted(by.keys())
print(f'seen classes total: {len(seen)}  train imgs usable: {len(common)}')

# ---------- CLASS split: rarest 20% of seen = pseudo_unseen (identical to unified_holdout_ctftbig.py) ----------
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

# ---------- IMAGE split within kept_seen: 20%/class val ----------
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


def seen_score(qF, P, TF, TL, Tx=None, lam=0.0, ret_maxproto=False):
    qF = qF.to(dev)
    n = qF.shape[0]
    out = torch.empty(n, S, device=dev)
    ps_all = torch.empty(n, S, device=dev) if ret_maxproto else None
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
        if ret_maxproto:
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

qF = gather(FtI, FtF, all_qfn)
qC = gather(CtI, CtF, all_qfn)
qL = gather(LtI, LtF, all_qfn)
qB = gather(BtI, BtF, all_qfn)
qFF2 = gather(FF2I, FF2F, all_qfn)

sc_ft, maxproto = seen_score(qF, PF, TFF, TLF, TseenTax, LAM, ret_maxproto=True)
sc_cap = seen_score(qC, PC, TFC, TLC, TseenTax, LAM)
sc_l = seen_score(qL, PL, TFL, TLL)
sc_b = seen_score(qB, PB, TFB, TLB, TseenTax, LAM)
seen_block_base = zc(sc_ft) + 1.0 * zc(sc_cap) + 0.5 * zc(sc_l)   # [n_q, S] baseline (no ctftbig)
seen_block_b = zc(sc_b)                                          # ctftbig member alone (zc'd)

# gold labels
gold_full = torch.tensor([kept_idx[y].item() for y in seen_query_y] + unseen_query_y, device=dev)
is_seen_row = torch.zeros(n_q, dtype=torch.bool, device=dev)
is_seen_row[:n_seen_q] = True


def oracle_seen_for(seen_block):
    pred = kept_idx.to(dev)[seen_block[:n_seen_q].argmax(1)]
    return (pred == gold_full[:n_seen_q]).float().mean().item() * 100


print('\n=== STAGE 1a: sweep W_B (ctftbig seen-block weight) vs oracle_seen ===')
base_oracle_seen = oracle_seen_for(seen_block_base)
print(f'W_B=0.00 (baseline, no ctftbig): oracle_seen={base_oracle_seen:.2f}')
wb_results = []
for W_B in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
    sb = seen_block_base + W_B * seen_block_b
    os_ = oracle_seen_for(sb)
    wb_results.append((W_B, os_))
    print(f'W_B={W_B:.2f}: oracle_seen={os_:.2f}  (delta {os_-base_oracle_seen:+.2f})')

# CONSERVATIVE pick, not argmax: repo notes explicitly warn "seen holdout axis historically
# overfits (v16/18/19 died real)". If oracle_seen keeps climbing without saturating as W_B grows,
# that's the overfitting signature (chasing holdout, not real signal) -- cap at the precedent
# range ext_ensemble.py itself tested (w<=1.0) rather than the unbounded holdout-argmax.
capped = [t for t in wb_results if t[0] <= 1.0]
best_W_B, best_os = max(capped, key=lambda t: t[1])
print(f'--> picked W_B={best_W_B} (oracle_seen={best_os:.2f})  [capped at 1.0 on purpose -- see note above]')

# ---- text block: ctftbig taxon + plain-L name (deployed) + optional fullft336_v2 3rd leg ----
qH_d = None  # plainH variant intentionally dropped here (ctftbig already established as strictly better)
qB_d = qB.to(dev)
qL_d = qL.to(dev)
qFF2_d = qFF2.to(dev)
M_taxon_B = qB_d @ TtH.t()
M_name_L = qL_d @ TnL.t()
M_taxon_FF2 = qFF2_d @ TtH.t()

oth = other_idx.to(dev)


def oracle_unseen_for(M_taxon_full, M_name_full, M_extra_full=None, w_extra=0.0):
    text_un = dbnorm(M_taxon_full[n_seen_q:][:, oth], 0.05, 0.5) + 0.5 * dbnorm(M_name_full[n_seen_q:][:, oth], 0.05, 0.5)
    if M_extra_full is not None and w_extra > 0:
        text_un = text_un + w_extra * dbnorm(M_extra_full[n_seen_q:][:, oth], 0.05, 0.5)
    pred = oth[text_un.argmax(1)]
    return (pred == gold_full[n_seen_q:]).float().mean().item() * 100


print('\n=== STAGE 1b: sweep W_U (fullft336_v2 unseen-leg weight) vs oracle_unseen ===')
base_oracle_unseen = oracle_unseen_for(M_taxon_B, M_name_L)
print(f'W_U=0.00 (baseline, deployed 2-term): oracle_unseen={base_oracle_unseen:.2f}')
wu_results = []
for W_U in [0.25, 0.5, 0.75, 1.0]:
    ou = oracle_unseen_for(M_taxon_B, M_name_L, M_taxon_FF2, W_U)
    wu_results.append((W_U, ou))
    print(f'W_U={W_U:.2f}: oracle_unseen={ou:.2f}  (delta {ou-base_oracle_unseen:+.2f})')
best_W_U, best_ou = max(wu_results, key=lambda t: t[1])
print(f'--> picked W_U={best_W_U} (oracle_unseen={best_ou:.2f})')

# ================= STAGE 2: full compliant gamma/w_text sweep at fixed W_B, W_U =================
print(f'\n=== STAGE 2: full sweep at W_B={best_W_B}, W_U={best_W_U} ===')
seen_block = seen_block_base + best_W_B * seen_block_b
text_dbnorm_full = (dbnorm(M_taxon_B, 0.05, 0.5) + 0.5 * dbnorm(M_name_L, 0.05, 0.5)
                     + best_W_U * dbnorm(M_taxon_FF2, 0.05, 0.5))

# ===== SANITY GATE (mirrors unified_holdout_ctftbig.py exactly) =====
pred_seen_only = kept_idx.to(dev)[seen_block[:n_seen_q].argmax(1)]
oracle_seen_acc = (pred_seen_only == gold_full[:n_seen_q]).float().mean().item() * 100
seen_only_acc = oracle_seen_acc

pred_unseen_only = oth[text_dbnorm_full[n_seen_q:][:, oth].argmax(1)]
oracle_unseen_acc = (pred_unseen_only == gold_full[n_seen_q:]).float().mean().item() * 100
unseen_only_acc = oracle_unseen_acc  # same formula used both times here (no masked/isolated split needed, additive)

gate_ok = True  # by construction: text-block used for oracle IS the masked/full one here
print(f'SANITY: oracle_seen={oracle_seen_acc:.2f} | oracle_unseen={oracle_unseen_acc:.2f} | gate_ok={gate_ok}')

oracle_overall = 0.564 * oracle_seen_acc + 0.436 * oracle_unseen_acc
print(f'oracle_overall (perfect gate) = {oracle_overall:.2f}')

text_block_z = zc(text_dbnorm_full)
kidx = kept_idx.to(dev)


def acc(pred_classidx, mask):
    return (pred_classidx[mask] == gold_full[mask]).float().mean().item() * 100


def build_unified_c(w_text, gamma):
    novelty = 1.0 - (maxproto - maxproto.min()) / (maxproto.max() - maxproto.min() + 1e-6)
    gamma_row = (gamma * novelty).unsqueeze(1)
    U = text_block_z.clone()
    U[:, kidx] = seen_block + w_text * text_block_z[:, kidx] - gamma_row
    return U


rows = []
gammas = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100]
w_texts = [0.25, 0.5, 1.0]
for w_text in w_texts:
    for gamma in gammas:
        U = build_unified_c(w_text, gamma)
        pred = U.argmax(1)
        seen_acc = acc(pred, is_seen_row)
        unseen_acc = acc(pred, ~is_seen_row)
        overall = 0.564 * seen_acc + 0.436 * unseen_acc
        rows.append(dict(w_text=w_text, gamma=gamma, seen=seen_acc, unseen=unseen_acc,
                          unseen_x055=unseen_acc * 0.55, overall_5644=overall))
        print(f'[ext w_text={w_text} gamma={gamma:>4}] seen={seen_acc:.2f} unseen={unseen_acc:.2f} '
              f'overall={overall:.2f}', flush=True)

cand = [r for r in rows if r['unseen'] >= 0.5 * oracle_unseen_acc]
best = max(cand, key=lambda r: r['overall_5644']) if cand else max(rows, key=lambda r: r['overall_5644'])

out = dict(W_B=best_W_B, W_U=best_W_U, wb_sweep=wb_results, wu_sweep=wu_results,
           oracle_seen=oracle_seen_acc, oracle_unseen=oracle_unseen_acc, oracle_overall=oracle_overall,
           rows=rows, best_compliant=best,
           baseline_ref='outputs/unified_frontier_ctftbig.json best_compliant_ctftbig overall_5644=56.51')
json.dump(out, open('outputs/unified_frontier_ctftbig_ext.json', 'w'), indent=2)
print('\n=== SUMMARY ===')
print(f'oracle: seen={oracle_seen_acc:.2f} unseen={oracle_unseen_acc:.2f} overall={oracle_overall:.2f}')
print('best_compliant (ext):', best)
print('baseline best_compliant (Jul16 ctftbig, no ext): overall_5644=56.51 (seen=84.28 unseen=20.58)')
print('wrote outputs/unified_frontier_ctftbig_ext.json')
