"""Transductive prototype refinement on top of unified_holdout_ctftbig.py's seen block.
Copy-and-modify (unified_holdout_ctftbig.py left untouched) -- this codebase's convention:
each experiment script + its output JSON is the permanent provenance record.

Idea (from deep-research: PUTM/BD-CSPN family): the current NCM prototypes P are frozen,
built ONLY from train support embeddings. This refines P using the unlabeled query batch
itself (transductive, no labels used) -- single-shot BD-CSPN-lite blend of the original
prototype with a confidence+novelty-gated pseudo-labeled centroid from the query batch.

Gate is load-bearing: without it, true-unseen queries would get pulled into seen
prototypes. refine_prototypes() takes NO gold labels / is_seen_row -- that omission from
the signature IS the leakage guard, not an oversight.

Stage 1: cheap grid over (alpha, nov_q, conf_q) on the primary, non-conflated metric
(oracle_seen_acc: seen_block argmax only, no text, no gamma) + safety diagnostics
(churn, contamination -- diagnostic only, never fed back into the gate).
Stage 2: baseline vs single best stage-1 config plugged into the existing ctftbig
w_text/gamma sweep (secondary, conflated -- gamma's novelty penalty also shifts).
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


# ---------- canonical class list (identical to unified_holdout_ctftbig.py) ----------
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

common = [fn for fn in Htf if fn in FtI and fn in CtI and fn in LtI and fn in BtI and fn in lab and lab[fn] in ci]
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


all_qfn = seen_query_fns + unseen_query_fns
n_seen_q = len(seen_query_fns)
n_unseen_q = len(unseen_query_fns)
n_q = n_seen_q + n_unseen_q

qF = gather(FtI, FtF, all_qfn)
qC = gather(CtI, CtF, all_qfn)
qL = gather(LtI, LtF, all_qfn)
qH = gather(HtI, HtF, all_qfn)
qB = gather(BtI, BtF, all_qfn)

sc_ft, maxproto = seen_score(qF, PF, TFF, TLF, TseenTax, LAM, ret_maxproto=True)
sc_cap = seen_score(qC, PC, TFC, TLC, TseenTax, LAM)
sc_l = seen_score(qL, PL, TFL, TLL)
seen_block = zc(sc_ft) + 1.0 * zc(sc_cap) + 0.5 * zc(sc_l)   # baseline, [n_q, S]
novelty = 1.0 - (maxproto - maxproto.min()) / (maxproto.max() - maxproto.min() + 1e-6)

gold_full = torch.tensor([kept_idx[y].item() for y in seen_query_y] + unseen_query_y, device=dev)
is_seen_row = torch.zeros(n_q, dtype=torch.bool, device=dev)
is_seen_row[:n_seen_q] = True
kidx = kept_idx.to(dev)


def oracle_seen_acc_of(block):
    pred = kidx[block[:n_seen_q].argmax(1)]
    return (pred == gold_full[:n_seen_q]).float().mean().item() * 100


def se_of(p_pct, n):
    p = p_pct / 100.0
    return ((p * (1 - p) / max(n, 1)) ** 0.5) * 100


baseline_oracle_seen = oracle_seen_acc_of(seen_block)
baseline_se = se_of(baseline_oracle_seen, n_seen_q)
print(f'BASELINE oracle_seen_acc={baseline_oracle_seen:.2f}  (SE={baseline_se:.2f}, n={n_seen_q})')


def refine_prototypes(P, qF_this, seen_block0, novelty0, alpha, nov_q, conf_q, min_count=2):
    """Single-shot BD-CSPN-lite. No gold labels / is_seen_row in the signature -- that
    absence is the leakage guard against pulling true-unseen queries into seen prototypes."""
    conf, pred = F.softmax(seen_block0, dim=1).max(1)
    gate = (novelty0 <= torch.quantile(novelty0, nov_q)) & (conf >= torch.quantile(conf, conf_q))
    w = gate.float() * conf
    qFd = qF_this.to(P.device)
    sums = torch.zeros_like(P).index_add_(0, pred, qFd * w.unsqueeze(1))
    wsum = torch.zeros(P.shape[0], device=P.device).index_add_(0, pred, w)
    cnt = torch.zeros(P.shape[0], device=P.device).index_add_(0, pred, gate.float())
    centroid = F.normalize(sums / wsum.clamp(min=1e-8).unsqueeze(1), dim=-1)
    eligible = cnt >= min_count
    P2 = P.clone()
    P2[eligible] = F.normalize((1 - alpha) * P[eligible] + alpha * centroid[eligible], dim=-1)
    return P2, gate, pred


def run_config(alpha, nov_q, conf_q, min_count=2):
    if alpha == 0:
        seen_block2 = seen_block
        gated_n, contam_pct = 0, float('nan')
    else:
        P2F, gate, _ = refine_prototypes(PF, qF, seen_block, novelty, alpha, nov_q, conf_q, min_count)
        P2C, _, _ = refine_prototypes(PC, qC, seen_block, novelty, alpha, nov_q, conf_q, min_count)
        P2L, _, _ = refine_prototypes(PL, qL, seen_block, novelty, alpha, nov_q, conf_q, min_count)
        sc_ft2 = seen_score(qF, P2F, TFF, TLF, TseenTax, LAM)
        sc_cap2 = seen_score(qC, P2C, TFC, TLC, TseenTax, LAM)
        sc_l2 = seen_score(qL, P2L, TFL, TLL)
        seen_block2 = zc(sc_ft2) + 1.0 * zc(sc_cap2) + 0.5 * zc(sc_l2)
        gated_n = int(gate.float().sum().item())
        contam_pct = (~is_seen_row[gate]).float().mean().item() * 100 if gated_n > 0 else float('nan')
    oracle_seen2 = oracle_seen_acc_of(seen_block2)
    churn = (kidx[seen_block[:n_seen_q].argmax(1)] != kidx[seen_block2[:n_seen_q].argmax(1)]).float().mean().item() * 100
    return dict(alpha=alpha, nov_q=nov_q, conf_q=conf_q, min_count=min_count,
                oracle_seen=oracle_seen2, delta=oracle_seen2 - baseline_oracle_seen, se=baseline_se,
                churn_pct=churn, gated_n=gated_n, contam_pct=contam_pct), seen_block2


# ===== STAGE 1: cheap grid, primary metric = oracle_seen_acc (non-conflated) =====
configs = []
seen_block2_by_key = {}
alphas = [0.15, 0.3, 0.5]
nov_qs = [0.3, 0.5, 0.7]
conf_qs = [0.5, 0.7]

r0, _ = run_config(0.0, 0.5, 0.5)  # baseline row (alpha=0 -> gate params irrelevant)
configs.append(r0)
print(f'[baseline] oracle_seen={r0["oracle_seen"]:.2f} delta={r0["delta"]:+.2f} SE={r0["se"]:.2f} '
      f'churn={r0["churn_pct"]:.2f}% contam={r0["contam_pct"]}')

for alpha in alphas:
    for nov_q in nov_qs:
        for conf_q in conf_qs:
            r, sb2 = run_config(alpha, nov_q, conf_q)
            configs.append(r)
            seen_block2_by_key[(alpha, nov_q, conf_q)] = sb2
            print(f'[alpha={alpha} nov_q={nov_q} conf_q={conf_q}] oracle_seen={r["oracle_seen"]:.2f} '
                  f'delta={r["delta"]:+.2f} SE={r["se"]:.2f} churn={r["churn_pct"]:.2f}% '
                  f'gated_n={r["gated_n"]} contam={r["contam_pct"]:.2f}%', flush=True)

# safe = doesn't degrade beyond ~1 SE and contamination stays below the pseudo-unseen population share
pop_unseen_share = 100.0 * n_unseen_q / n_q
safe = [r for r in configs if r['alpha'] > 0 and r['delta'] >= -r['se']
        and (r['contam_pct'] != r['contam_pct'] or r['contam_pct'] < pop_unseen_share)]
best = max(safe, key=lambda r: r['delta']) if safe else None
print(f'pop_unseen_share={pop_unseen_share:.2f}%  safe_configs={len(safe)}/{len(configs)-1}')
print('BEST (safe, max delta):', best)

# ===== STAGE 2: secondary/conflated -- plug baseline + best config into existing gamma sweep =====
stage2 = None
if best is not None:
    key = (best['alpha'], best['nov_q'], best['conf_q'])
    seen_block_best = seen_block2_by_key[key]

    qB_d = qB.to(dev)
    qL_d = qL.to(dev)
    M_taxon_B = qB_d @ TtH.t()
    M_name_L = qL_d @ TnL.t()
    text_dbnorm_full = dbnorm(M_taxon_B, 0.05, 0.5) + 0.5 * dbnorm(M_name_L, 0.05, 0.5)
    text_block_z = zc(text_dbnorm_full)

    def acc(pred_classidx, mask):
        return (pred_classidx[mask] == gold_full[mask]).float().mean().item() * 100

    def gamma_sweep(sb, label):
        rows = []
        for w_text in [0.25, 0.5, 1.0]:
            for gamma in [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
                U = text_block_z.clone()
                U[:, kidx] = sb + w_text * text_block_z[:, kidx] - gamma
                pred = U.argmax(1)
                seen_acc = acc(pred, is_seen_row)
                unseen_acc = acc(pred, ~is_seen_row)
                overall = 0.564 * seen_acc + 0.436 * unseen_acc
                rows.append(dict(label=label, w_text=w_text, gamma=gamma, seen=seen_acc,
                                  unseen=unseen_acc, overall_5644=overall))
        return rows

    rows_baseline = gamma_sweep(seen_block, 'baseline')
    rows_best = gamma_sweep(seen_block_best, 'protorefine_best')
    b1 = max(rows_baseline, key=lambda r: r['overall_5644'])
    b2 = max(rows_best, key=lambda r: r['overall_5644'])
    print('[SECONDARY/CONFLATED] baseline best:', b1)
    print('[SECONDARY/CONFLATED] protorefine best:', b2)
    stage2 = dict(config=key, baseline_best=b1, protorefine_best=b2,
                   rows_baseline=rows_baseline, rows_protorefine=rows_best)

result = dict(
    baseline_oracle_seen=baseline_oracle_seen, baseline_se=baseline_se,
    n_seen_q=n_seen_q, n_unseen_q=n_unseen_q, pop_unseen_share=pop_unseen_share,
    stage1_configs=configs, stage1_best=best, stage2_secondary_conflated=stage2,
)
import os
os.makedirs('handoff', exist_ok=True)
json.dump(result, open('handoff/unified_frontier_ctftbig_protorefine.json', 'w'), indent=2)
print('wrote handoff/unified_frontier_ctftbig_protorefine.json')
