"""v107 agreement_recheck: does the leg-agreement bonus (v103, +0.40pt fused-argmax,
capped by 87.4% unanimity) change the FUSION enough to lift recall@10 over the deployed
98.854 baseline? Stage 1 only (probe2 pattern) -- proceed to leak-free re-ranker retrain
(rerank_seen_v83 pattern) ONLY if recall@10 genuinely improves.

  conda activate onet && python research/agreement_recheck_v107.py
"""
import json
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import LAM, D, MEMBERS, NFOLD, dev, holdout_split, load_train_embs, zc
from rerank_seen_v83 import components
from seen_inat_v95 import member_scores

DEPLOYED = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
LEGS = list(DEPLOYED.keys())
ALPHA_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
BASE_RECALL10 = 98.854
V83_RERANK = 91.674
K = 10
BEST_ALPHA = 4.0


def agree1_bonus(top1, N, S):
    b = torch.zeros(N, S, device=dev)
    idx = torch.arange(N, device=dev)
    for t in LEGS:
        b[idx, top1[t]] += 1.0
    return b


def unanimous_bonus(top1, N, S):
    b = torch.zeros(N, S, device=dev)
    idx = torch.arange(N, device=dev)
    t0, t1, t2 = LEGS
    all_agree = (top1[t0] == top1[t1]) & (top1[t1] == top1[t2])
    b[idx[all_agree], top1[t0][all_agree]] = 1.0
    return b


def stage2_retrain(leg_mats, Zbase, top1, bonuses, Tseen, S, yv, cls_arr):
    """Leak-free re-ranker retrain (rerank_seen_v83 methodology) over candidates
    drawn from the MODIFIED fusion (deployed block + BEST_ALPHA * agree1 bonus)."""
    n_train = {}
    sig = {}
    modblock = sum(DEPLOYED[t] * Zbase[t] for t in LEGS) + BEST_ALPHA * bonuses['agree1']
    for t, _, w in MEMBERS:
        if t not in DEPLOYED:
            continue
        qF, P, TF, TL = leg_mats[t]
        ps, cm, tx = components(qF, P, TF, TL, Tseen, S)
        sig[f'{t}_proto'], sig[f'{t}_cmax'], sig[f'{t}_taxon'] = ps, cm, tx
        counts = torch.zeros(S, device=dev)
        counts.index_add_(0, TL, torch.ones_like(TL, dtype=torch.float32))
        n_train[t] = counts
    sig['block'] = modblock
    sig['agree1'] = bonuses['agree1']
    n_train_all = n_train[LEGS[0]]  # class counts identical across legs (same seen split)
    signals = [f'{t}_{c}' for t in LEGS for c in ('proto', 'cmax', 'taxon')] + ['block', 'agree1']

    topk = modblock.topk(K, dim=1).indices
    nq = topk.shape[0]
    Y = (topk == yv.to(dev)[:, None]).float()
    ceiling = 100 * (Y.sum(1) > 0).float().mean().item()
    print(f'\n[stage2] modified-fusion candidate pool: gold in top-{K} = {ceiling:.3f}% '
          f'(deployed was {BASE_RECALL10:.3f}%)', flush=True)

    cols, names = [], []
    for name in signals:
        M = sig[name]
        v = torch.gather(M, 1, topk)
        cols += [v - M.max(1, keepdim=True).values,
                 (v - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-6)]
        names += [f'{name}_gapmax', f'{name}_z']
        r = torch.empty(nq, K, device=M.device)
        for s in range(0, nq, 256):
            e = min(s + 256, nq)
            r[s:e] = (M[s:e].unsqueeze(1) > v[s:e].unsqueeze(2)).sum(2).float()
        cols.append(r / S)
        names.append(f'{name}_pctrank')
    cols += [torch.arange(K, device=dev).float().expand(nq, K) / K,
             torch.log1p(n_train_all[topk])]
    names += ['block_rank', 'log_ntrain']
    X = torch.stack(cols, dim=2).cpu()
    Y = Y.cpu()

    base = 100 * (Y[:, 0] > 0).float().mean().item()
    nf = X.shape[2]
    print(f'[stage2] queries={nq} K={K} feats={nf} | base rank-1 on MODIFIED pool = {base:.3f}',
          flush=True)

    uniq = sorted(set(cls_arr.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cls_arr])
    qfold = np.repeat(folds, K)
    Xf = X.reshape(nq * K, nf).numpy()
    Yf = Y.reshape(nq * K).numpy()

    oof = np.zeros(nq * K)
    for f in range(NFOLD):
        tr, va = qfold != f, qfold == f
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
        m = LogisticRegression(max_iter=4000, C=1.0)
        m.fit((Xf[tr] - mu) / sd, Yf[tr])
        oof[va] = m.decision_function((Xf[va] - mu) / sd)
    sc = torch.tensor(oof).reshape(nq, K)
    acc = 100 * Y[torch.arange(nq), sc.argmax(1)].mean().item()
    print(f'\n[stage2] deployed rank-1 (v83, real-scored)   {V83_RERANK:6.3f}')
    print(f'[stage2] MODIFIED-fusion base rank-1 (proxy)  {base:6.3f}')
    print(f'[stage2] MODIFIED-fusion LEARNED re-ranker    {acc:6.3f}  '
          f'(vs v83: {acc - V83_RERANK:+.3f})')
    print(f'[stage2] ceiling at K={K}                       {ceiling:6.3f}', flush=True)
    return acc, ceiling, base


def main():
    torch.set_num_threads(8)
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby, val_seen = sp['trby'], [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    ci = {c: i for i, c in enumerate(classes)}
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    N = len(val_seen)
    print(f'val_seen rows {N} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zbase, leg_mats = {}, {}
    for t, _, _ in MEMBERS:
        if t not in DEPLOYED:
            continue
        idx, feats, _ = train[t]
        P = torch.zeros(S, feats.shape[1])
        cnt = torch.zeros(S)
        TFl, TLl = [], []
        for c in seen:
            for fn in trby[c]:
                if fn not in idx:
                    continue
                f = feats[idx[fn]]
                P[s2i[c]] += f
                cnt[s2i[c]] += 1
                TFl.append(f)
                TLl.append(s2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
        TF = torch.stack(TFl).to(dev)
        TL = torch.tensor(TLl).to(dev)
        qF = torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev)
        base = member_scores(qF, P, TF, TL, Tseen, S)
        Zbase[t] = zc(base)
        leg_mats[t] = (qF, P, TF, TL)

    fused_base = sum(DEPLOYED[t] * Zbase[t] for t in LEGS)
    top10_base = fused_base.topk(10, dim=1).indices.cpu()
    rec_base = {k: 100 * (top10_base[:, :k] == yv.unsqueeze(1)).any(1).float().mean().item()
                for k in (1, 5, 10)}
    print(f'\nDEPLOYED (alpha=0)  top1 {rec_base[1]:.3f}  recall@5 {rec_base[5]:.3f}  '
          f'recall@10 {rec_base[10]:.3f}  (expect ~98.854)', flush=True)

    top1 = {t: Zbase[t].argmax(1) for t in LEGS}
    bonuses = {'agree1': agree1_bonus(top1, N, S), 'unanimous': unanimous_bonus(top1, N, S)}
    n_unanimous = int((bonuses['unanimous'].sum(1) > 0).sum())
    print(f'rows where all 3 legs unanimously agree: {n_unanimous}/{N} '
          f'({100 * n_unanimous / N:.1f}%)', flush=True)

    print('\n=== recall@10 vs deployed 98.854, bonus x ALPHA grid ===', flush=True)
    rows = {}
    for bname, B in bonuses.items():
        for alpha in ALPHA_GRID:
            final = fused_base + alpha * B
            top10 = final.topk(10, dim=1).indices.cpu()
            r10 = 100 * (top10 == yv.unsqueeze(1)).any(1).float().mean().item()
            rows[f'{bname}_a{alpha}'] = r10
            print(f'  {bname:10s} alpha={alpha:<5} recall@10 {r10:.3f}  '
                  f'(delta {r10 - BASE_RECALL10:+.3f})', flush=True)

    best_key = max(rows, key=rows.get)
    best_r10 = rows[best_key]
    improved = best_r10 > BASE_RECALL10
    verdict = (f'RECALL@10 IMPROVES: {best_key} -> {best_r10:.3f} ({best_r10 - BASE_RECALL10:+.3f}) '
               '-- proceeding to leak-free stack retrain'
               if improved else
               f'DEAD at stage 1: best {best_key} -> {best_r10:.3f} '
               f'({best_r10 - BASE_RECALL10:+.3f}), does not exceed deployed {BASE_RECALL10}. '
               'No stack retrain (cannot re-rank a candidate pool into being better than it is).')
    print(f'\nVERDICT: {verdict}', flush=True)

    stack = None
    if improved:
        cls_arr = sp['cls'][:sp['n_seen']]
        acc, ceiling, mod_base = stage2_retrain(leg_mats, Zbase, top1, bonuses, Tseen, S, yv, cls_arr)
        clears = acc >= V83_RERANK + 1.0
        print(f"\n[stage2] {'CLEARS' if clears else 'does not clear'} the >=92.674 stack bar "
              f'(got {acc:.3f})', flush=True)
        stack = {'reranked_acc': acc, 'ceiling_at_k10': ceiling, 'modfusion_base_rank1': mod_base,
                 'v83_baseline': V83_RERANK, 'clears_bar': clears, 'best_alpha': BEST_ALPHA}

    json.dump({'rec_base': rec_base, 'rows': rows, 'best_key': best_key, 'best_r10': best_r10,
               'base_recall10': BASE_RECALL10, 'improved': improved, 'verdict': verdict,
               'n_unanimous_frac': n_unanimous / N, 'stage2': stack},
              open(f'{OUT}/agreement_recheck_v107.json', 'w'), indent=2)
    print(f'wrote {OUT}/agreement_recheck_v107.json', flush=True)


if __name__ == '__main__':
    main()
