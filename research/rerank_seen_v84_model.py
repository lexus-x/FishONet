"""v84: model-class comparison for the SEEN re-ranker, at the deployed K=10.

v83 ships LogisticRegression and never tried anything else on this head. v81 compared logistic vs
HistGradientBoosting, but only on the unseen head's LEAKY pool, so that comparison was discarded
along with the leak and never repeated on a clean pool.

The seen head cannot leak (every one of the kept classes is gold-eligible for a val_seen image), so
a fair comparison here is just a model swap on the cached holdout features. Ceiling at K=10 is
98.85 and the deployed re-ranker reaches 91.674, so ~7.2 proxy-pt sit inside the existing shortlist.

Transfer anchor (HANDOFF 2026-08-29, measured): 0.0295 real overall-pt per leak-free proxy-pt on
this head. Deployed gain +1.803 proxy = +0.0533 real.

  conda activate onet && python research/rerank_seen_v84_model.py
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
from learned_gate_v77 import NFOLD, OUT  # noqa: E402
from rerank_seen_v83 import CACHE  # noqa: E402

DEPLOYED_GAIN = 1.803   # v83 CV gain, = +0.0533 real
ANCHOR = 0.0295         # real overall-pt per proxy-pt, measured on this head

MODELS = {
    'logistic': lambda: LogisticRegression(max_iter=4000, C=1.0),
    'hgb': lambda: HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.06,
                                                  min_samples_leaf=40, l2_regularization=1.0,
                                                  random_state=0),
    'hgb_deep': lambda: HistGradientBoostingClassifier(max_iter=500, max_depth=None,
                                                       learning_rate=0.05, min_samples_leaf=20,
                                                       l2_regularization=1.0, random_state=0),
}


def main():
    torch.set_num_threads(8)
    d = torch.load(CACHE, weights_only=False)
    X, Y, cls = d['X'], d['Y'], d['cls']
    nq, k, nf = X.shape
    base = 100 * (Y[:, 0] > 0).float().mean().item()
    ceil = 100 * (Y.sum(1) > 0).float().mean().item()
    print(f'queries={nq} K={k} feats={nf} | rank-1 {base:.3f} | ceiling@{k} {ceil:.3f}', flush=True)

    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cls])
    qfold = np.repeat(folds, k)
    Xf, Yf = X.reshape(nq * k, nf).numpy(), Y.reshape(nq * k).numpy()

    print(f'\n{"model":<10} {"top-1":>7} {"gain":>7} {"vs v83":>7} {"proj real":>10}')
    print('-' * 45)
    res = {}
    for name, mk in MODELS.items():
        oof = np.zeros(nq * k)
        for f in range(NFOLD):
            tr, va = qfold != f, qfold == f
            mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
            m = mk()
            m.fit((Xf[tr] - mu) / sd, Yf[tr])
            Z = (Xf[va] - mu) / sd
            oof[va] = m.decision_function(Z) if hasattr(m, 'decision_function') else m.predict_proba(Z)[:, 1]
        acc = 100 * Y[torch.arange(nq), torch.tensor(oof).reshape(nq, k).argmax(1)].mean().item()
        gain, delta = acc - base, (acc - base) - DEPLOYED_GAIN
        res[name] = {'acc': acc, 'gain': gain, 'proj_real_delta': delta * ANCHOR}
        print(f'{name:<10} {acc:>7.3f} {gain:>+7.3f} {delta:>+7.3f} {delta * ANCHOR:>+10.3f}',
              flush=True)

    best = max(res, key=lambda n: res[n]['gain'])
    print(f'\nbest = {best}: {res[best]["proj_real_delta"]:+.3f} projected real overall-pt vs v83')
    print('verdict: ' + ('WORTH A SLOT' if res[best]['proj_real_delta'] >= 0.10 else
                         'NOT WORTH A SLOT (<0.10 real)'), flush=True)
    json.dump({'base': base, 'ceiling': ceil, 'deployed_gain': DEPLOYED_GAIN, 'anchor': ANCHOR,
               'res': res, 'best': best},
              open(os.path.join(OUT, 'rerank_seen_v84_model.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
