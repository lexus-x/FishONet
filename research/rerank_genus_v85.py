"""v85: taxonomic-consensus features for the unseen re-ranker (leak-free pool).

Why this and not another feature: v84 showed HistGradientBoosting -- which can represent arbitrary
interactions of the existing 34 features -- gains NOTHING over logistic (-0.302 at K=20). So the
residual 14.67 proxy-pt is not an interaction the model failed to fit; it is information the 34
features do not contain. Any new feature must therefore be genuinely outside the per-candidate
score statistics.

Genus consensus is exactly that. All 34 current features are per-candidate score statistics; NONE
of them look at the RELATIONSHIP between candidates. If 6 of a query's top-20 are Sebastes, that is
evidence about the genus which no per-candidate score can express -- and it addresses the exact
observed failure (fusion retrieves the right species in the top-20 two thirds of the time and then
mis-orders it, HANDOFF:1149).

Distinct from the dead levers: HANDOFF kills genus as a *scoring leg* (genus-level prototypes,
"genus <=0") and a *hierarchy prior* (family top-1 23.8 < species 28.9; hard 2-stage -20pt). Both
predict FROM genus. This instead reads consensus AMONG an already-retrieved shortlist. Free: genus
is the first token of the scientific binomial.

Kill bar: the transfer anchor is 0.0261 real-pt per leak-free proxy-pt, so +0.10 real needs
+3.83 proxy-pt over the deployed +8.154.

  conda activate onet && python research/rerank_genus_v85.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, OUT  # noqa: E402
from rerank_unseen_v81 import DEPLOYED_W, LEGS, NFOLD, build_legs  # noqa: E402
from rerank_leakfree_v82 import pool_features  # noqa: E402

KS = (20, 50)
DEPLOYED_GAIN = 8.154
ANCHOR = 0.0261


def genus_features(topk, genus_of_pool, fused):
    """Consensus among the retrieved shortlist. Nothing here is a per-candidate score."""
    nq, k = topk.shape
    g = genus_of_pool[topk]                                  # [nq, k] genus id per candidate
    same = (g.unsqueeze(2) == g.unsqueeze(1)).float()        # [nq, k, k] same-genus mask
    rank = torch.arange(k, device=topk.device).float()

    count = same.sum(2) / k                                  # share of shortlist in this genus
    # rank-weighted mass of the genus: earlier shortlist positions weigh more
    w = ((k - rank) / k).view(1, 1, k)
    mass = (same * w).sum(2) / (w.sum() + 1e-6) * k
    # is this candidate the best-ranked member of its own genus?
    big = rank.view(1, 1, k) + (1 - same) * 1e6
    best_rank = big.min(2).values
    is_best = (best_rank == rank.view(1, k)).float()
    # gap between this candidate's fused score and the best of its genus
    fv = torch.gather(fused, 1, topk)
    gbest = (fv.unsqueeze(1) - (1 - same) * 1e6).max(2).values
    gap_to_genus_best = fv - gbest

    return ([count, mass, best_rank / k, is_best, gap_to_genus_best],
            ['genus_count', 'genus_mass', 'genus_bestrank', 'genus_is_best', 'genus_gapbest'])


def cv(X, Y, folds):
    nq, k, nf = X.shape
    Xf, Yf = X.reshape(nq * k, nf).numpy(), Y.reshape(nq * k).numpy()
    qfold = np.repeat(folds, k)
    oof = np.zeros(nq * k)
    for f in range(NFOLD):
        tr, va = qfold != f, qfold == f
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
        m = LogisticRegression(max_iter=4000, C=1.0)
        m.fit((Xf[tr] - mu) / sd, Yf[tr])
        oof[va] = m.decision_function((Xf[va] - mu) / sd)
    return 100 * Y[torch.arange(nq), torch.tensor(oof).reshape(nq, k).argmax(1)].mean().item()


def main():
    torch.set_num_threads(8)
    D = FishData()
    pool = torch.tensor(sorted(D.ci[c] for c in D.pseudo))
    legs, _, n_photos = build_legs(D, pool)
    pos = {int(g): j for j, g in enumerate(pool.tolist())}
    gold = torch.tensor([pos[D.ci[c]] for c in D.pseudo for _ in D.by[c]])
    fused = sum(w * legs[n] for n, w in zip(LEGS, DEPLOYED_W))

    names_pool = [D.classes[int(c)] for c in pool.tolist()]
    genus_names = [n.split()[0] for n in names_pool]
    gid = {g: i for i, g in enumerate(sorted(set(genus_names)))}
    genus_of_pool = torch.tensor([gid[g] for g in genus_names])
    multi = sum(1 for g in set(genus_names) if genus_names.count(g) > 1)
    print(f'pool {len(pool)} classes / {len(gid)} genera ({multi} genera with >1 species) '
          f'-- consensus can only fire where a genus has siblings', flush=True)

    base_top1 = 100 * (fused.argmax(1) == gold).float().mean().item()
    gcls = np.array([c for c in D.pseudo for _ in D.by[c]])
    uniq = sorted(set(gcls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in gcls])
    print(f'deployed fusion top-1 = {base_top1:.3f}\n', flush=True)

    print(f'{"K":>4} {"arm":<16} {"top-1":>7} {"gain":>7} {"vs v82":>7} {"proj real":>10}')
    print('-' * 56)
    res, best_pack = {}, None
    for K in KS:
        topk = fused.topk(K, dim=1).indices
        Xb, nb = pool_features(legs, topk, n_photos, fused)
        Y = (topk == gold[:, None]).float()
        gcols, gnames = genus_features(topk, genus_of_pool, fused)
        Xg = torch.cat([Xb, torch.stack(gcols, dim=2)], dim=2)
        for arm, (X, names) in {'base(34)': (Xb, nb),
                                f'+genus({len(gnames)})': (Xg, nb + gnames)}.items():
            acc = cv(X, Y, folds)
            gain, delta = acc - base_top1, (acc - base_top1) - DEPLOYED_GAIN
            res[f'K{K}_{arm}'] = {'acc': acc, 'gain': gain, 'proj': delta * ANCHOR}
            print(f'{K:>4} {arm:<16} {acc:>7.3f} {gain:>+7.3f} {delta:>+7.3f} {delta*ANCHOR:>+10.3f}',
                  flush=True)
            if best_pack is None or gain > best_pack[0]:
                best_pack = (gain, K, names, X, Y)

    gain, K, names, X, Y = best_pack
    delta = gain - DEPLOYED_GAIN
    print(f'\nbest: K={K} feats={len(names)} gain {gain:+.3f} '
          f'({delta:+.3f} vs deployed) = {delta*ANCHOR:+.3f} projected real')

    if delta > 0:
        nq, k, nf = X.shape
        Xf, Yf = X.reshape(nq * k, nf).numpy(), Y.reshape(nq * k).numpy()
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
        m = LogisticRegression(max_iter=4000, C=1.0)
        m.fit((Xf - mu) / sd, Yf)
        pickle.dump({'model': m, 'mu': mu, 'sd': sd, 'names': names, 'K': K, 'gain': gain,
                     'legs': LEGS, 'deployed_w': DEPLOYED_W, 'pool_size': len(pool)},
                    open(f'{OUT}/rerank_unseen_v85_genus.pkl', 'wb'))
        print(f'wrote {OUT}/rerank_unseen_v85_genus.pkl')
        print('\ntop coefficients:')
        for n, c in sorted(zip(names, m.coef_[0]), key=lambda t: -abs(t[1]))[:12]:
            mark = '  <-- NEW' if n.startswith('genus_') else ''
            print(f'  {n:24s} {c:+.4f}{mark}')
    json.dump(res, open(f'{OUT}/rerank_genus_v85.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
