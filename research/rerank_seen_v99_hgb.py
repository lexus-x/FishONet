"""v99: swap v83's LogisticRegression re-ranker for HistGradientBoostingClassifier on the
IDENTICAL cached features (reuses outputs/rerank_seen_v83_holdout.pt -- no rebuild, no new
embeddings). Same K=10 candidates, same 32 pool-size-invariant features, same class-disjoint
5-fold CV split as rerank_seen_v83.py. Only the model class changes.

Why this specific lever: v83's own docstring says recall@10 is 98.85% but LR rank-1 only reaches
~91.67% (+1.803pt over the 89.87 base) -- ~7pt of headroom between rank-1 and recall@10 that a
linear-in-standardized-features model leaves on the table. The *_gapmax / *_z / *_pctrank
features per leg interact (e.g. "ctftshift_taxon_pctrank is only informative when
fullft336shift_cmax_gapmax is small") in ways a single linear decision boundary over 32 dims
cannot express but a shallow boosted-tree ensemble can, for free, on the exact same inputs.

Bar (pre-registered, from the assignment): this is a per-CANDIDATE re-ranker, so the leak-free
transfer anchor applies -- ~0.026-0.030 overall-pt per leak-free proxy-pt. The random baseline
here is the SAME leak-free construction as v82/v83 (seen pool = candidates always gold-eligible),
so a holdout gain is trustworthy in kind, but the assignment's bar is stated directly on holdout:
need a MEANINGFULLY bigger number than 91.21 (already real-tested negative), i.e.
>= 92.21% (>= +1.0pt over the 91.21 linear-reweight holdout figure). Since v83 LR already gets
to ~91.67% on this same task, the pass bar for HGB specifically is beating LR's 91.67% by a
further meaningful margin (not just replicating it) to clear the assignment's 92.21% floor.
Fail bar: HGB <= LR's rank-1 (91.67%ish) or the ceiling@10 (98.85%) isn't approached further ->
DEAD, model capacity was never the bottleneck (feature information was), matching the pattern
already established for gate C/f tuning being exhausted.

  conda activate onet && python research/rerank_seen_v99_hgb.py
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
from rerank_seen_v83 import CACHE, K, build_holdout  # noqa: E402


def main():
    torch.set_num_threads(8)
    if os.path.exists(CACHE):
        d = torch.load(CACHE, weights_only=False)
        X, Y, cls, names = d['X'], d['Y'], d['cls'], d['names']
        print(f'loaded cache X={tuple(X.shape)}', flush=True)
    else:
        X, Y, cls, names = build_holdout()
        torch.save({'X': X, 'Y': Y, 'cls': cls, 'names': names}, CACHE)

    nq, k, nf = X.shape
    base = 100 * (Y[:, 0] > 0).float().mean().item()
    ceiling = 100 * (Y.sum(1) > 0).float().mean().item()
    print(f'queries={nq} K={k} feats={nf} | deployed base {base:.2f}  ceiling@{k} {ceiling:.2f}\n',
          flush=True)

    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cls])
    qfold = np.repeat(folds, k)
    Xf = X.reshape(nq * k, nf).numpy()
    Yf = Y.reshape(nq * k).numpy()

    models = {
        'logistic': lambda: LogisticRegression(max_iter=4000, C=1.0),
        'hgb': lambda: HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.05,
            min_samples_leaf=30, l2_regularization=1.0, random_state=0),
    }
    oof = {name: np.zeros(nq * k) for name in models}
    for f in range(NFOLD):
        tr, va = qfold != f, qfold == f
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
        for name, mk in models.items():
            m = mk()
            if name == 'logistic':
                m.fit((Xf[tr] - mu) / sd, Yf[tr])
                oof[name][va] = m.decision_function((Xf[va] - mu) / sd)
            else:  # tree models are scale-invariant -- fit on raw features
                m.fit(Xf[tr], Yf[tr])
                oof[name][va] = m.predict_proba(Xf[va])[:, 1]
        print(f'  fold {f} done', flush=True)

    print(f'\n{"model":10s} {"rank-1":>8s} {"gain":>8s}', flush=True)
    results = {}
    for name in models:
        sc = torch.tensor(oof[name]).reshape(nq, k)
        acc = 100 * Y[torch.arange(nq), sc.argmax(1)].mean().item()
        gain = acc - base
        results[name] = {'acc': acc, 'gain': gain}
        print(f'{name:10s} {acc:8.3f} {gain:+8.3f}', flush=True)
    print(f'{"ceiling":10s} {ceiling:8.3f}', flush=True)

    hgb_vs_lr = results['hgb']['acc'] - results['logistic']['acc']
    print(f'\nHGB vs LR: {hgb_vs_lr:+.3f}pt', flush=True)
    verdict = ('CLEARS -- worth a submission slot' if results['hgb']['acc'] >= 92.21
               else 'DEAD -- model capacity is not the bottleneck, feature information is')
    print(f'VERDICT ({results["hgb"]["acc"]:.3f} vs 92.21 bar): {verdict}', flush=True)
    json.dump({'base': base, 'ceiling': ceiling, 'results': results, 'hgb_vs_lr': hgb_vs_lr},
              open(f'{OUT}/rerank_seen_v99_hgb.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/rerank_seen_v99_hgb.json', flush=True)


if __name__ == '__main__':
    main()
