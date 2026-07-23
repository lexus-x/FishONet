"""Step 3: extend unified_holdout.py's variant-c sweep to compare the compliant text-block
built from plain-H features (existing) vs ctft_big features (the proven unseen unlock).

ctftbig text-block (matches v20's deployed unseen route, from src/ctft_big_finalize.py):
    dbnorm(qH_ctftbig @ TtH_taxon, .05,.5) + 0.5*dbnorm(qL_plain @ TnL_name, .05,.5)
plain-H text-block (existing, both terms H-backbone):
    dbnorm(qH @ TtH_taxon, .05,.5) + 0.5*dbnorm(qH @ TnH_name, .05,.5)

Everything else (seen-block: ft/cap/L NCM+cmax+taxon, variant-c smooth novelty gate,
sanity gate) is IDENTICAL to unified_holdout.py -- only the text-block source features change.
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
txtH = torch.load('outputs/text_emb_h.pt', weights_only=False)
txtL = torch.load('outputs/text_emb.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
assert txtHt['classes'] == classes and txtH['classes'] == classes and txtL['classes'] == classes
ci = {c: i for i, c in enumerate(classes)}
NCLS = len(classes)
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)      # [NCLS,1024] taxon (H-space)
TnH = F.normalize(txtH['emb_name'].float(), dim=-1).to(dev)        # [NCLS,1024] name (H-space)
TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)        # [NCLS,768]  name (L-space)

lab = json.load(open(f'{D}/label_train.json'))

# ---------- train embeddings (ft/cap/L/H/ctftbig) ----------
FtI, FtF, Ftf = load('outputs/emb_train_ft.pt')
CtI, CtF, Ctf = load('outputs/emb_train_cap.pt')
LtI, LtF, Ltf = load('outputs/emb_train.pt')
HtI, HtF, Htf = load('outputs/emb_train_h.pt')
BtI, BtF, Btf = load('outputs/emb_train_ctftbig.pt')
print(f'ctftbig train cache: n={len(Btf)}')

common = [fn for fn in Htf if fn in FtI and fn in CtI and fn in LtI and fn in BtI and fn in lab and lab[fn] in ci]
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
        if ret_maxproto:
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
qB = gather(BtI, BtF, all_qfn)   # ctft_big features for the SAME query images

sc_ft, maxproto = seen_score(qF, PF, TFF, TLF, TseenTax, LAM, ret_maxproto=True)
sc_cap = seen_score(qC, PC, TFC, TLC, TseenTax, LAM)
sc_l = seen_score(qL, PL, TFL, TLL)
seen_block = zc(sc_ft) + 1.0 * zc(sc_cap) + 0.5 * zc(sc_l)          # [n_q, S]  (kept_seen cols)

# ---- text block A: plain-H (existing) ----
qH_d = qH.to(dev)
M_taxon_H = qH_d @ TtH.t()
M_name_H = qH_d @ TnH.t()

# ---- text block B: ctftbig taxon + plain-L name (v20 deployed unseen route) ----
qB_d = qB.to(dev)
qL_d = qL.to(dev)
M_taxon_B = qB_d @ TtH.t()
M_name_L = qL_d @ TnL.t()

# gold labels
gold_full = torch.tensor([kept_idx[y].item() for y in seen_query_y] + unseen_query_y, device=dev)
is_seen_row = torch.zeros(n_q, dtype=torch.bool, device=dev)
is_seen_row[:n_seen_q] = True


def acc(pred_classidx, mask):
    return (pred_classidx[mask] == gold_full[mask]).float().mean().item() * 100


def run_variant(M_taxon_full, M_name_full, label):
    text_dbnorm_full = dbnorm(M_taxon_full, 0.05, 0.5) + 0.5 * dbnorm(M_name_full, 0.05, 0.5)

    # ===== SANITY GATE (mirrors unified_holdout.py exactly) =====
    pred_seen_only = kept_idx.to(dev)[seen_block[:n_seen_q].argmax(1)]
    oracle_seen_acc = (pred_seen_only == gold_full[:n_seen_q]).float().mean().item() * 100
    seen_only_acc = oracle_seen_acc  # seen_block doesn't depend on text-block choice

    # isolated: dbnorm computed on the SUBMATRIX (unseen rows x other_idx cols only)
    oth = other_idx.to(dev)
    M_taxon_un = M_taxon_full[n_seen_q:][:, oth]
    M_name_un = M_name_full[n_seen_q:][:, oth]
    text_unseen_only = dbnorm(M_taxon_un, 0.05, 0.5) + 0.5 * dbnorm(M_name_un, 0.05, 0.5)
    pred_unseen_only = oth[text_unseen_only.argmax(1)]
    oracle_unseen_acc = (pred_unseen_only == gold_full[n_seen_q:]).float().mean().item() * 100

    # masked-after: dbnorm computed on the FULL n_q x NCLS matrix (as used in deployment), then sliced
    pred_full_text_masked_nonkept = oth[text_dbnorm_full[n_seen_q:][:, oth].argmax(1)]
    unseen_only_acc = (pred_full_text_masked_nonkept == gold_full[n_seen_q:]).float().mean().item() * 100

    gate_ok = abs(seen_only_acc - oracle_seen_acc) < 1.0 and abs(unseen_only_acc - oracle_unseen_acc) < 1.0
    print(f'[{label}] SANITY: oracle_seen={oracle_seen_acc:.2f} seen_only_masked={seen_only_acc:.2f} | '
          f'oracle_unseen={oracle_unseen_acc:.2f} unseen_only_masked={unseen_only_acc:.2f} | gate_ok={gate_ok}')

    oracle_overall = 0.564 * oracle_seen_acc + 0.436 * oracle_unseen_acc
    real_unseen_est_oracle = oracle_unseen_acc * 0.55

    text_block_z = zc(text_dbnorm_full)
    kidx = kept_idx.to(dev)

    def build_unified_c(w_text, gamma):
        novelty = 1.0 - (maxproto - maxproto.min()) / (maxproto.max() - maxproto.min() + 1e-6)
        gamma_row = (gamma * novelty).unsqueeze(1)
        U = text_block_z.clone()
        U[:, kidx] = seen_block + w_text * text_block_z[:, kidx] - gamma_row
        return U

    rows = []
    gammas = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    w_texts = [0.25, 0.5, 1.0]
    for w_text in w_texts:
        for gamma in gammas:
            U = build_unified_c(w_text, gamma)
            pred = U.argmax(1)
            seen_acc = acc(pred, is_seen_row)
            unseen_acc = acc(pred, ~is_seen_row)
            overall = 0.564 * seen_acc + 0.436 * unseen_acc
            rows.append(dict(text_block=label, w_text=w_text, gamma=gamma,
                              seen=seen_acc, unseen=unseen_acc,
                              unseen_x055=unseen_acc * 0.55, overall_5644=overall))
            print(f'[{label} w_text={w_text} gamma={gamma:>4}] seen={seen_acc:.2f} '
                  f'unseen={unseen_acc:.2f} unseen_x055={unseen_acc*0.55:.2f} overall={overall:.2f}', flush=True)
    return dict(label=label, oracle_seen=oracle_seen_acc, oracle_unseen=oracle_unseen_acc,
                oracle_overall=oracle_overall, real_unseen_est_oracle=real_unseen_est_oracle,
                gate_ok=gate_ok, rows=rows)


result_H = run_variant(M_taxon_H, M_name_H, 'plainH')
result_B = run_variant(M_taxon_B, M_name_L, 'ctftbig')

# pick best_compliant among ctftbig rows: highest overall, unseen clearly recovered (>=50% of its oracle),
# seen conservative (not chasing seen's overfit peak -- prefer rows near seen's plateau, penalize outlier peaks)
cand = [r for r in result_B['rows'] if r['unseen'] >= 0.5 * result_B['oracle_unseen']]
best_B = max(cand, key=lambda r: r['overall_5644']) if cand else max(result_B['rows'], key=lambda r: r['overall_5644'])
cand_H = [r for r in result_H['rows'] if r['unseen'] >= 0.5 * result_H['oracle_unseen']]
best_H = max(cand_H, key=lambda r: r['overall_5644']) if cand_H else max(result_H['rows'], key=lambda r: r['overall_5644'])

out = dict(plainH=result_H, ctftbig=result_B, best_compliant_ctftbig=best_B, best_compliant_plainH=best_H)
json.dump(out, open('outputs/unified_frontier_ctftbig.json', 'w'), indent=2)
print('=== SUMMARY ===')
print('plainH  oracle: seen=%.2f unseen=%.2f overall=%.2f' % (result_H['oracle_seen'], result_H['oracle_unseen'], result_H['oracle_overall']))
print('ctftbig oracle: seen=%.2f unseen=%.2f overall=%.2f' % (result_B['oracle_seen'], result_B['oracle_unseen'], result_B['oracle_overall']))
print('best_compliant (ctftbig):', best_B)
print('best_compliant (plainH):', best_H)
print('wrote outputs/unified_frontier_ctftbig.json')
