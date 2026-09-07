"""v82: leak-free re-validation of the unseen re-ranker.

v81 scored +13.374 on the holdout and +0.006 real. Post-mortem (HANDOFF 2026-08-29): the holdout
candidate pool is 1,159 pseudo classes (gold-eligible) + 11,598 real-novel (never gold). That 9.1%
subpopulation is identifiable from the features, so the re-ranker learned "prefer gold-eligible"
instead of learning to rank evidence. Class-disjoint CV cannot see this.

Fix by construction: restrict the holdout candidate pool to the 1,159 pseudo classes ONLY. Every
candidate is then gold-eligible and the shortcut carries zero information -- any gain must come
from genuine evidence ranking.

Two changes to keep features pool-size invariant (holdout pool 1,159 vs eval 11,598):
  * rank -> PERCENTILE (rank / pool_size), not log-rank
  * z-scores and gap-to-max are already scale-free

If the re-ranker still beats the deployed fusion here, the idea is real and worth a slot.
If it collapses to ~0, the entire +13.374 was the leak and the unseen head is closed.

  conda activate onet && python research/rerank_leakfree_v82.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, OUT  # noqa: E402
from rerank_unseen_v81 import DEPLOYED_W, K, LEGS, NFOLD, build_legs  # noqa: E402


def pool_features(legs, topk, n_photos, fused):
    """Pool-size-invariant version of rerank_unseen_v81.pair_features."""
    nq, k = topk.shape
    C = fused.shape[1]
    cols, names = [], []
    for name in LEGS:
        M = legs[name]
        v = torch.gather(M, 1, topk)
        cols += [v - M.max(1, keepdim=True).values,
                 (v - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-6)]
        names += [f'{name}_gapmax', f'{name}_z']
        r = torch.empty(nq, k, device=M.device)
        for s in range(0, nq, 256):
            e = min(s + 256, nq)
            r[s:e] = (M[s:e].unsqueeze(1) > v[s:e].unsqueeze(2)).sum(2).float()
        cols.append(r / C)                      # PERCENTILE, pool-size invariant
        names.append(f'{name}_pctrank')
    fv = torch.gather(fused, 1, topk)
    cols += [fv - fused.max(1, keepdim=True).values,
             torch.arange(k, device=fv.device).float().expand(nq, k) / k,
             n_photos[topk], (n_photos[topk] > 0).float()]
    names += ['fused_gapmax', 'fused_rank', 'n_photos', 'has_bank']
    return torch.stack(cols, dim=2), names


def main():
    torch.set_num_threads(8)
    D = FishData()
    pool = torch.tensor(sorted(D.ci[c] for c in D.pseudo))       # 1,159 gold-eligible ONLY
    print(f'leak-free pool: {len(pool)} classes, all gold-eligible '
          f'(v81 pool was {len(D.cand)} with {len(D.pseudo)} eligible = '
          f'{100*len(D.pseudo)/len(D.cand):.1f}%)', flush=True)

    legs, _, n_photos = build_legs(D, pool)
    pos = {int(g): j for j, g in enumerate(pool.tolist())}
    gold = []
    for c in D.pseudo:
        for fn in D.by[c]:
            gold.append(pos[D.ci[c]])
    gold = torch.tensor(gold)
    assert gold.shape[0] == next(iter(legs.values())).shape[0]

    fused = sum(w * legs[n] for n, w in zip(LEGS, DEPLOYED_W))
    base = 100 * (fused.argmax(1) == gold).float().mean().item()
    print(f'deployed fusion top-1 on leak-free pool = {base:.3f}', flush=True)
    for kk in (1, 5, 10, 20):
        r = 100 * (fused.topk(kk, 1).indices == gold[:, None]).any(1).float().mean().item()
        print(f'  recall@{kk:<3d} {r:6.2f}', flush=True)

    topk = fused.topk(K, dim=1).indices
    X, names = pool_features(legs, topk, n_photos, fused)
    Y = (topk == gold[:, None]).float()
    nq, k, nf = X.shape
    print(f'\nqueries={nq} K={k} feats={nf} | gold in top-{k}: '
          f'{100*(Y.sum(1) > 0).float().mean():.2f}%', flush=True)
    base_top1 = 100 * (Y[:, 0] > 0).float().mean().item()

    gcls = np.array([c for c in D.pseudo for _ in D.by[c]])
    uniq = sorted(set(gcls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in gcls])
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
    sc = torch.tensor(oof).reshape(nq, k)
    acc = 100 * Y[torch.arange(nq), sc.argmax(1)].mean().item()
    gain = acc - base_top1
    print(f'\nfusion rank-1        {base_top1:6.3f}')
    print(f'LEAK-FREE re-ranker  {acc:6.3f}  ({gain:+.3f})')
    print(f'\nv81 (leaky pool) claimed +13.374 -> real +0.006', flush=True)
    print(f'verdict: {"REAL SIGNAL - worth a slot" if gain >= 1.0 else "NO SIGNAL - the +13.374 was the leak"}',
          flush=True)
    json.dump({'pool': len(pool), 'base_top1': base_top1, 'reranked': acc, 'gain': gain},
              open(os.path.join(OUT, 'rerank_leakfree_v82.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
