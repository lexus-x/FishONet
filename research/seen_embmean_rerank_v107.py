"""v107 follow-up: the embedding-mean fusion (research/seen_embmean_v107.py) cleared the fair
recall@10 bar (98.972 vs deployed 98.854, +0.118pt) -- per the audit protocol, that earns a real
stack-level test: retrain the seen re-ranker leak-free on candidates from THIS fusion instead of
the deployed weighted one, using the exact same methodology as research/rerank_seen_v83.py
(pool-size-invariant proto/cmax/taxon percentile-rank features, K=10, 5-fold class-disjoint CV,
LogisticRegression). Compare the resulting TRUE stack number to the deployed stack's 91.674%.

  conda activate onet && python research/seen_embmean_rerank_v107.py
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import LAM, NFOLD, D, dev, holdout_split, load_train_embs, zc

LEGS = ['ctftshift', 'ftshift', 'fullft336shift']
K = 10
SIGNALS = ['proto', 'cmax', 'taxon', 'block']
DEPLOYED_STACK = 91.674
KILL = 1.0


def components(Q, P, TF, TL, Tseen, S):
    n = Q.shape[0]
    ps = torch.empty(n, S, device=dev)
    cm = torch.empty(n, S, device=dev)
    for i in range(0, n, 2000):
        e = Q[i:i + 2000]
        ps[i:i + 2000] = e @ P.t()
        sim = e @ TF.t()
        c = torch.full((e.shape[0], S), -1e9, device=dev)
        c.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        cm[i:i + 2000] = c
    return ps, cm, Q @ Tseen.t()


def cand_features(sig, topk, n_train, block):
    nq, k = topk.shape
    C = block.shape[1]
    cols, names = [], []
    for name in SIGNALS:
        M = sig[name]
        v = torch.gather(M, 1, topk)
        cols += [v - M.max(1, keepdim=True).values,
                 (v - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-6)]
        names += [f'{name}_gapmax', f'{name}_z']
        r = torch.empty(nq, k, device=M.device)
        for s in range(0, nq, 256):
            e = min(s + 256, nq)
            r[s:e] = (M[s:e].unsqueeze(1) > v[s:e].unsqueeze(2)).sum(2).float()
        cols.append(r / C)
        names.append(f'{name}_pctrank')
    cols += [torch.arange(k, device=block.device).float().expand(nq, k) / k,
             torch.log1p(n_train[topk])]
    names += ['block_rank', 'log_ntrain']
    return torch.stack(cols, dim=2), names


def build_holdout():
    train, tb, b2f, b2l = load_train_embs()
    sp = holdout_split(train, tb, b2f, b2l)
    ns, kept, ci, trby = sp['n_seen'], sp['kept'], sp['ci'], sp['trby']
    k2i = {c: i for i, c in enumerate(kept)}
    S = len(kept)
    files = sp['val_files'][:ns]
    gold = torch.tensor([k2i[c] for c in sp['cls'][:ns]], device=dev)
    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[torch.tensor([ci[c] for c in kept], device=dev)]
    n_train = torch.tensor([len(trby[c]) for c in kept], dtype=torch.float32, device=dev)

    # build merged embedding-mean leg: L2norm each of the 3 legs, average, L2norm again
    TFl_per_leg, val_per_leg = {}, {}
    TL_l = []
    for t in LEGS:
        idx, feats, _ = train[t]
        TFl, TLl = [], []
        for c in kept:
            for fn in trby[c]:
                TFl.append(feats[idx[fn]])
                TLl.append(k2i[c])
        TFl_per_leg[t] = torch.stack(TFl).to(dev)
        val_per_leg[t] = torch.stack([feats[idx[fn]] for fn in files]).to(dev)
        TL_l = TLl  # identical ordering across legs
    TL = torch.tensor(TL_l, device=dev)
    TF_mean = F.normalize(sum(F.normalize(TFl_per_leg[t], dim=-1) for t in LEGS) / 3.0, dim=-1)
    Q_mean = F.normalize(sum(F.normalize(val_per_leg[t], dim=-1) for t in LEGS) / 3.0, dim=-1)

    P = torch.zeros(S, TF_mean.shape[1], device=dev)
    cnt = torch.zeros(S, device=dev)
    P.index_add_(0, TL, TF_mean)
    cnt.index_add_(0, TL, torch.ones_like(TL, dtype=torch.float32))
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)

    ps, cm, tx = components(Q_mean, P, TF_mean, TL, Tseen, S)
    block = zc(ps + 2.0 * cm + LAM * tx)
    sig = {'proto': ps, 'cmax': cm, 'taxon': tx, 'block': block}

    topk = block.topk(K, dim=1).indices
    X, names = cand_features(sig, topk, n_train, block)
    Y = (topk == gold[:, None]).float()
    base = 100 * (topk[:, 0] == gold).float().mean().item()
    print(f'embmean fusion rank-1 (bare argmax): {base:.3f}', flush=True)
    return X.cpu(), Y.cpu(), np.array(sp['cls'][:ns]), names, base


def main():
    torch.set_num_threads(8)
    X, Y, cls, names, base = build_holdout()
    nq, k, nf = X.shape
    print(f'queries={nq} K={k} feats={nf} | gold in top-{k}: {100*(Y.sum(1) > 0).float().mean():.3f}%', flush=True)

    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cls])
    qfold = np.repeat(folds, k)
    Xf = X.reshape(nq * k, nf).numpy()
    Yf = Y.reshape(nq * k).numpy()

    oof = np.zeros(nq * k)
    for f in range(NFOLD):
        tr, va = qfold != f, qfold == f
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
        m = LogisticRegression(max_iter=4000, C=1.0)
        m.fit((Xf[tr] - mu) / sd, Yf[tr])
        oof[va] = m.decision_function((Xf[va] - mu) / sd)
        print(f'  fold {f}', flush=True)
    sc = torch.tensor(oof).reshape(nq, k)
    acc = 100 * Y[torch.arange(nq), sc.argmax(1)].mean().item()
    gain_vs_own_base = acc - base
    lift_vs_deployed_stack = acc - DEPLOYED_STACK

    print(f'\nembmean bare rank-1        {base:6.3f}')
    print(f'embmean + LEARNED reranker {acc:6.3f}  (+{gain_vs_own_base:.3f} vs own base)')
    print(f'DEPLOYED stack (v83)       {DEPLOYED_STACK:6.3f}')
    print(f'lift vs deployed stack     {lift_vs_deployed_stack:+.3f}pt  (kill >= +{KILL})', flush=True)
    verdict = 'CLEARS' if lift_vs_deployed_stack >= KILL else 'DEAD'
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'embmean_base': base, 'embmean_reranked': acc, 'deployed_stack': DEPLOYED_STACK,
               'lift_vs_deployed_stack': lift_vs_deployed_stack, 'verdict': verdict},
              open(f'{OUT}/seen_embmean_rerank_v107.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/seen_embmean_rerank_v107.json', flush=True)


if __name__ == '__main__':
    main()
