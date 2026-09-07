"""v84: model-class x K sweep for the unseen re-ranker, on the LEAK-FREE pool.

v81 compared logistic vs HistGradientBoosting -- but only on the LEAKY pool, where the winner was
whichever model best exploited "prefer gold-eligible". v82 fixed the leak by restricting the pool to
the 1,159 gold-eligible classes and, in doing so, dropped the comparison: it ships plain
LogisticRegression. The model-class question was therefore never asked on clean data.

K was never swept on the clean pool either. It is inherited as 20 from v81.

Both are free to test -- class-disjoint 5-fold CV, no submission. Convert with the confirmed
transfer anchor (HANDOFF 2026-08-29): ~0.0261 real overall-pt per leak-free proxy-pt on this head,
so the deployed +8.154 proxy = +0.213 real. A new arm must beat 8.154 by ~4 proxy-pt to be worth
+0.10 real, and by ~38 to be worth +1.00.

  conda activate onet && python research/rerank_leakfree_v84_sweep.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, OUT  # noqa: E402
from rerank_unseen_v81 import DEPLOYED_W, LEGS, NFOLD, build_legs  # noqa: E402
from rerank_leakfree_v82 import pool_features  # noqa: E402

KS = (20, 30, 50)
DEPLOYED_GAIN = 8.154   # v82 leak-free CV gain, = +0.213 real
ANCHOR = 0.0261         # real overall-pt per leak-free proxy-pt (v82 measured)

MODELS = {
    'logistic': lambda: LogisticRegression(max_iter=4000, C=1.0),
    'hgb': lambda: HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.06,
                                                  min_samples_leaf=40, l2_regularization=1.0,
                                                  random_state=0),
}


def cv_gain(X, Y, folds, mk):
    """Class-disjoint OOF re-rank accuracy. Same protocol as v82."""
    nq, k, nf = X.shape
    Xf, Yf = X.reshape(nq * k, nf).numpy(), Y.reshape(nq * k).numpy()
    qfold = np.repeat(folds, k)
    oof = np.zeros(nq * k)
    for f in range(NFOLD):
        tr, va = qfold != f, qfold == f
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
        m = mk()
        m.fit((Xf[tr] - mu) / sd, Yf[tr])
        # decision_function where available: predict_proba underflows to 0.0 in the deployed
        # builder (build_v83 line ~366) and argmax degenerates to index 0.
        Z = (Xf[va] - mu) / sd
        oof[va] = m.decision_function(Z) if hasattr(m, 'decision_function') else m.predict_proba(Z)[:, 1]
    sc = torch.tensor(oof).reshape(nq, k)
    return 100 * Y[torch.arange(nq), sc.argmax(1)].mean().item()


def main():
    torch.set_num_threads(8)
    D = FishData()
    pool = torch.tensor(sorted(D.ci[c] for c in D.pseudo))          # gold-eligible ONLY
    print(f'leak-free pool: {len(pool)} classes, all gold-eligible', flush=True)

    legs, _, n_photos = build_legs(D, pool)
    pos = {int(g): j for j, g in enumerate(pool.tolist())}
    gold = torch.tensor([pos[D.ci[c]] for c in D.pseudo for _ in D.by[c]])
    fused = sum(w * legs[n] for n, w in zip(LEGS, DEPLOYED_W))
    assert gold.shape[0] == fused.shape[0]

    base_top1 = 100 * (fused.argmax(1) == gold).float().mean().item()
    print(f'deployed fusion top-1 = {base_top1:.3f}  (v82 measured 59.28)', flush=True)
    for kk in (1, 5, 10, 20, 30, 50, 100):
        r = 100 * (fused.topk(kk, 1).indices == gold[:, None]).any(1).float().mean().item()
        print(f'  recall@{kk:<4d} {r:6.2f}', flush=True)

    gcls = np.array([c for c in D.pseudo for _ in D.by[c]])
    uniq = sorted(set(gcls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in gcls])

    print(f'\n{"K":>4} {"ceiling":>8} {"model":<10} {"top-1":>7} {"gain":>7} {"vs v82":>7} {"proj real":>10}')
    print('-' * 60)
    res = {}
    for K in KS:
        topk = fused.topk(K, dim=1).indices
        X, names = pool_features(legs, topk, n_photos, fused)
        Y = (topk == gold[:, None]).float()
        ceil = 100 * (Y.sum(1) > 0).float().mean().item()
        for mname, mk in MODELS.items():
            acc = cv_gain(X, Y, folds, mk)
            gain = acc - base_top1
            delta = gain - DEPLOYED_GAIN
            res[f'{mname}_K{K}'] = {'acc': acc, 'gain': gain, 'ceiling': ceil,
                                    'proj_real_delta': delta * ANCHOR}
            print(f'{K:>4} {ceil:>8.2f} {mname:<10} {acc:>7.3f} {gain:>+7.3f} {delta:>+7.3f} '
                  f'{delta * ANCHOR:>+10.3f}', flush=True)

    best = max(res, key=lambda k: res[k]['gain'])
    b = res[best]
    print(f'\nbest = {best}: {b["gain"]:+.3f} proxy-pt vs v82 deployed {DEPLOYED_GAIN:+.3f} '
          f'= {b["proj_real_delta"]:+.3f} projected real overall-pt')
    print('verdict: ' + ('WORTH A SLOT' if b['proj_real_delta'] >= 0.10 else
                         'NOT WORTH A SLOT (<0.10 real) - close it in HANDOFF'), flush=True)
    json.dump({'base_top1': base_top1, 'deployed_gain': DEPLOYED_GAIN, 'anchor': ANCHOR,
               'res': res, 'best': best},
              open(os.path.join(OUT, 'rerank_leakfree_v84_sweep.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
