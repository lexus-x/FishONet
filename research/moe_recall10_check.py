"""moe_recheck: does the per-image MoE fusion gate (seen_moe_fusion_v100.py, previously
measured 91.252% OOF fused-argmax accuracy vs deployed 89.87%, bucket-consistent
+3.82/+2.01/+1.16 by n_train) also lift RECALL@10 over the deployed fixed-weight fusion's
98.854 (research/probe2_retriever_recall.py's number)? Bare argmax roughly ties ctftshift
solo (91.20%), so the open question is whether the re-ranking headroom (recall@10 minus
top1) got bigger or smaller under the MoE weighting -- accuracy alone can't tell us that,
because the stack applies a re-ranker on top of these top-10 candidates.

Stage 1 (this script): reuse v100's exact leg-score + gate-training pipeline (same
holdout_split, same 5-fold class-disjoint CV, same Gate/fuse/train_fold), but instead of
only reporting argmax accuracy, also report recall@1/5/10 of the OOF-fused scores.
Kill gate: if recall@10 does not exceed 98.854, stop -- do not touch the re-ranker.

Stage 2 (only if stage1 passes): retrain the seen re-ranker (rerank_seen_v83.py pattern:
pool-size-invariant proto/cmax/taxon percentile features, LogisticRegression, K=10, 5-fold
class-disjoint CV) using top-10 candidates drawn from the MoE-fused block instead of the
deployed fixed-weight block, and report the true stack accuracy vs the 91.674 bar
(must clear >=92.674 to be worth anything).

  conda activate onet && python research/moe_recall10_check.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import LAM, MEMBERS, NFOLD, dev, holdout_split, load_train_embs, zc
from seen_moe_fusion_v100_d_per_image_moe_fusion_gate import (
    DEPLOYED, LEGS, Gate, build_leg_scores, fuse, leg_feats, train_fold,
)

STAGE1_BAR = 98.854   # deployed recall@10 (probe2 / task background)
STAGE2_BAR = 92.674   # +1.0pt over v83's 91.674 stack bar
K = 10


def recall_at(fused, yv, ks=(1, 5, 10)):
    top = fused.topk(max(ks), dim=1).indices.cpu()
    return {k: 100 * (top[:, :k] == yv.unsqueeze(1)).any(1).float().mean().item() for k in ks}


def main():
    torch.set_num_threads(8)
    Zbase, yv, S, n_train = build_leg_scores()
    N = yv.shape[0]

    fused_deploy = sum(DEPLOYED[t] * Zbase[t] for t in LEGS)
    rec_deploy = recall_at(fused_deploy, yv)
    print(f'\ndeployed fixed-weight fusion : top1 {rec_deploy[1]:.3f}  '
          f'recall@5 {rec_deploy[5]:.3f}  recall@10 {rec_deploy[10]:.3f}  '
          f'(sanity vs task background: top1 89.870, recall@5 97.767, recall@10 98.854)',
          flush=True)

    feats = []
    for t in LEGS:
        m, mx = leg_feats(Zbase[t])
        feats += [m, mx]
    G = torch.stack(feats, dim=1)
    G = (G - G.mean(0, keepdim=True)) / (G.std(0, keepdim=True) + 1e-6)
    Bstack = torch.stack([Zbase[t] for t in LEGS], dim=0)

    uniq = sorted(set(yv.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in yv.tolist()])

    oof_fused = torch.zeros(N, S, device=dev)
    for f in range(NFOLD):
        tr = torch.tensor(folds != f)
        va = torch.tensor(folds == f)
        fused_va, _ = train_fold(G[tr], Bstack[:, tr, :], yv[tr].to(dev),
                                  G[va], Bstack[:, va, :], seed=1000 + f)
        oof_fused[va] = fused_va
        print(f'  fold {f} trained', flush=True)

    rec_moe = recall_at(oof_fused, yv)
    acc_moe = 100 * (oof_fused.argmax(1).cpu() == yv).float().mean().item()
    print(f'\nMoE per-image gate (OOF)    : top1 {rec_moe[1]:.3f}  '
          f'recall@5 {rec_moe[5]:.3f}  recall@10 {rec_moe[10]:.3f}  '
          f'(argmax acc {acc_moe:.3f})', flush=True)

    d10 = rec_moe[10] - rec_deploy[10]
    print(f'\nrecall@10 delta vs deployed: {d10:+.3f}  (stage1 bar: must exceed '
          f'{STAGE1_BAR:.3f})', flush=True)

    passed = rec_moe[10] > STAGE1_BAR
    result = {'deployed': rec_deploy, 'moe_oof': rec_moe, 'moe_argmax_acc': acc_moe,
              'recall10_delta': d10, 'stage1_passed': passed}
    json.dump(result, open(f'{OUT}/moe_recall10_check.json', 'w'), indent=2)

    if not passed:
        print('\nVERDICT: recall@10 did NOT improve -- idea DEAD, stopping before re-ranker '
              '(cannot re-rank a candidate pool into being better than it is).', flush=True)
        return

    print('\nrecall@10 improved -- proceeding to stage 2 (leak-free re-ranker retrain on '
          'MoE-fusion candidates).', flush=True)
    stage2(oof_fused, yv, S, n_train, folds)


def stage2(oof_fused, yv, S, n_train, folds):
    """Retrain the seen re-ranker on top-K candidates drawn from the MoE-fused block,
    exact rerank_seen_v83.py feature/model recipe, but block = per-image MoE fusion."""
    lab_train, tb, b2f, b2l = load_train_embs()
    sp = holdout_split(lab_train, tb, b2f, b2l)
    kept, ci, trby = sp['kept'], sp['ci'], sp['trby']
    k2i = {c: i for i, c in enumerate(sorted(kept))}
    assert S == len(kept)
    val_seen = [f for f, y_ in zip(sp['val_files'], sp['y']) if y_ == 1]
    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt',
                                 weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[torch.tensor([ci[c] for c in sorted(kept)], device=dev)]

    ACTIVE = [t for t, _, w in MEMBERS if w > 0]
    SIGNALS = [f'{t}_{c}' for t in ACTIVE for c in ('proto', 'cmax', 'taxon')] + ['block']

    sig = {}
    for t, hs, w in MEMBERS:
        if w == 0:
            continue
        idx, feats, _ = lab_train[t]
        P = torch.zeros(S, feats.shape[1])
        cnt = torch.zeros(S)
        TF, TL = [], []
        for c in sorted(kept):
            for fn in trby[c]:
                if fn not in idx:
                    continue
                f = feats[idx[fn]]
                P[k2i[c]] += f
                cnt[k2i[c]] += 1
                TF.append(f)
                TL.append(k2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
        TFt = torch.stack(TF).to(dev)
        TLt = torch.tensor(TL, device=dev)
        Q = torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev)
        n = Q.shape[0]
        ps = torch.empty(n, S, device=dev)
        cm = torch.empty(n, S, device=dev)
        for i in range(0, n, 2000):
            e = Q[i:i + 2000]
            ps[i:i + 2000] = e @ P.t()
            sim = e @ TFt.t()
            c_ = torch.full((e.shape[0], S), -1e9, device=dev)
            c_.scatter_reduce_(1, TLt.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
            cm[i:i + 2000] = c_
        tx = Q @ Tseen.t()
        sig[f'{t}_proto'], sig[f'{t}_cmax'], sig[f'{t}_taxon'] = ps, cm, tx
        print(f'  signals {t} done', flush=True)

    block = oof_fused   # candidates come from the MoE-fused block, not the deployed fixed sum
    sig['block'] = block
    gold = yv.to(dev)

    topk = block.topk(K, dim=1).indices
    nq = topk.shape[0]

    def cand_features(sig, topk, n_train_, block_):
        nqq, k = topk.shape
        C = block_.shape[1]
        cols, names = [], []
        for name in SIGNALS:
            M = sig[name]
            v = torch.gather(M, 1, topk)
            cols += [v - M.max(1, keepdim=True).values,
                     (v - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-6)]
            names += [f'{name}_gapmax', f'{name}_z']
            r = torch.empty(nqq, k, device=M.device)
            for s in range(0, nqq, 256):
                e = min(s + 256, nqq)
                r[s:e] = (M[s:e].unsqueeze(1) > v[s:e].unsqueeze(2)).sum(2).float()
            cols.append(r / C)
            names.append(f'{name}_pctrank')
        cols += [torch.arange(k, device=block_.device).float().expand(nqq, k) / k,
                 torch.log1p(n_train_[topk])]
        names += ['block_rank', 'log_ntrain']
        return torch.stack(cols, dim=2), names

    n_train_dev = n_train.to(dev)
    X, names = cand_features(sig, topk, n_train_dev, block)
    Y = (topk == gold[:, None]).float()
    X, Y = X.cpu(), Y.cpu()

    nqf, k, nf = X.shape
    base = 100 * (Y[:, 0] > 0).float().mean().item()
    ceil = 100 * (Y.sum(1) > 0).float().mean().item()
    print(f'\nqueries={nqf} K={k} feats={nf} | gold in top-{k}: {ceil:.2f}%')
    print(f'MoE-fusion rank-1 (pre-rerank) = {base:.3f}\n', flush=True)

    cls = np.array([sp['cls'][i] for i, y_ in enumerate(sp['y']) if y_ == 1])
    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    qfold = np.array([fold_of[c] for c in cls])
    qfold = np.repeat(qfold, k)
    Xf = X.reshape(nqf * k, nf).numpy()
    Yf = Y.reshape(nqf * k).numpy()

    oof = np.zeros(nqf * k)
    for f in range(NFOLD):
        tr, va = qfold != f, qfold == f
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
        m = LogisticRegression(max_iter=4000, C=1.0)
        m.fit((Xf[tr] - mu) / sd, Yf[tr])
        oof[va] = m.decision_function((Xf[va] - mu) / sd)
        print(f'  fold {f}', flush=True)
    sc = torch.tensor(oof).reshape(nqf, k)
    acc = 100 * Y[torch.arange(nqf), sc.argmax(1)].mean().item()

    print(f'\nMoE-fusion rank-1 (pre-rerank) {base:6.3f}')
    print(f'LEARNED re-ranker on MoE cands {acc:6.3f}')
    print(f'v83 deployed-fusion stack bar  91.674')
    print(f'stage2 bar (>=)                {STAGE2_BAR:.3f}', flush=True)

    clears = acc >= STAGE2_BAR
    verdict = 'CLEARS' if clears else 'DOES NOT CLEAR'
    print(f'\nVERDICT: {verdict} the +1.0pt bar', flush=True)
    json.dump({'moe_stack_acc': acc, 'moe_prererank_rank1': base, 'v83_bar': 91.674,
               'stage2_bar': STAGE2_BAR, 'clears': clears},
              open(f'{OUT}/moe_recall10_check_stage2.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
