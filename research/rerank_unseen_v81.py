"""v81: learned re-ranker over the unseen head's top-K. Replaces hand-picked fusion weights.

The unseen head is a sum of 10 legs with constants picked by hand (text 1/0.5/0.75/1/1/1,
bank 4, 336-bank 3, B2 2.5/2.0) applied uniformly to every candidate. Measured holdout ceiling:

    recall@1 33.22   @5 52.07   @10 59.62   @20 67.30   @50 74.76

So the gold novel species is already retrieved for ~2/3 of queries and then mis-ordered. A fixed
weight vector cannot fix that, because the right weighting is candidate-dependent: the bank leg is
authoritative when that class HAS reference photos and pure noise when it does not (currently
uncovered classes are masked to -1e4 and then dbnorm'd, which is a constant, not evidence).

This learns a per-(query, candidate) score from features that already exist -- no new encoder,
no new data. Validated with class-disjoint CV on the gold class, the same protocol as the gate.

  conda activate onet && python research/rerank_unseen_v81.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT  # noqa: E402
from crop_chase53_proxy import HOLD_VIEWS, max_dbnorm  # noqa: E402
from inat_maxpool_proxy import score_maxpool  # noqa: E402
from v50_shiftbank_proxy import (  # noqa: E402
    BANK_336, BANK_CTFT, B2_WF, B2_WL, F336_Q, FROZEN_PROTO, FROZEN_Q, IMG_W, LORA_PROTO,
    LORA_Q, proto_leg, queries,
)

K = 20
NFOLD = 5
TAXABIND_W = 1.0
VIEW_KEYS = ['center', 'squash'] + [f'ostrip{i}' for i in range(5)]
# leg name -> weight in the deployed hand-tuned stack (kept only as a reference/feature scale)
LEGS = ['t_ctft_taxon', 't_L_name', 't_336v2_taxon', 't_ftshift_taxon', 't_ctft_taxctx',
        't_taxabind', 'bank_ct', 'bank_336', 'b2_frozen', 'b2_lora']
DEPLOYED_W = [1.0, 0.5, 0.75, 1.0, 1.0, TAXABIND_W, IMG_W, 3.0, B2_WF, B2_WL]


def build_legs(D, cand):
    """The 10 deployed unseen-head legs on the 2,318 pseudo-novel holdout queries."""
    TtH_c, TnL_c = D.TtH[cand], D.TnL[cand]
    TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'),
                                 weights_only=False)['emb_taxctx'].float(), dim=-1)[cand]
    Ttb = F.normalize(torch.load(os.path.join(OUT, 'text_emb_taxabind_taxctx.pt'),
                                 weights_only=False)['emb_taxctx'].float(), dim=-1)[cand]
    q = {}
    for tag in ['ctftshift', 'ftshift', 'fullft336_v2']:
        idx, feats, _ = load_emb(os.path.join(OUT, f'emb_train_{tag}.pt'))
        q[tag], gold = queries(D, idx, feats)
    qL, gold = queries(D, D.LtI, D.LtF)
    ti, tf, _ = load_emb(os.path.join(OUT, 'emb_train_taxabind.pt'))
    qtb, _ = queries(D, ti, tf)

    legs = {
        't_ctft_taxon': dbnorm(q['ctftshift'] @ TtH_c.t()),
        't_L_name': dbnorm(qL @ TnL_c.t()),
        't_336v2_taxon': dbnorm(q['fullft336_v2'] @ TtH_c.t()),
        't_ftshift_taxon': dbnorm(q['ftshift'] @ TtH_c.t()),
        't_ctft_taxctx': dbnorm(q['ctftshift'] @ TTX.t()),
        't_taxabind': dbnorm(qtb @ Ttb.t()),
    }
    bank = torch.load(BANK_CTFT, weights_only=False)['bank']
    packed = torch.load(HOLD_VIEWS, weights_only=False)
    tens = {k: F.normalize(v.float(), dim=-1) for k, v in packed['views'].items()}
    legs['bank_ct'] = max_dbnorm(tens, VIEW_KEYS, bank, cand)
    i336, f336, _ = load_emb(F336_Q)
    Q336, _ = queries(D, i336, f336)
    legs['bank_336'] = dbnorm(score_maxpool(Q336, torch.load(BANK_336, weights_only=False)['bank'],
                                            cand, topm=4))
    fi, ff, _ = load_emb(FROZEN_Q)
    Qf, _ = queries(D, fi, ff)
    legs['b2_frozen'] = proto_leg(Qf, FROZEN_PROTO, cand)
    li, lf, _ = load_emb(LORA_Q)
    Ql, _ = queries(D, li, lf)
    legs['b2_lora'] = proto_leg(Ql, LORA_PROTO, cand)

    n_photos = torch.zeros(len(cand))
    for j, g in enumerate(cand.tolist()):
        p = bank.get(g)
        if p is not None:
            n_photos[j] = p.shape[0] if hasattr(p, 'shape') else len(p)
    return legs, gold, n_photos


def pair_features(legs, topk, n_photos, fused):
    """Per (query, candidate) features. All within-query normalized forms are included because
    dbnorm's across-image term is batch-coupled and does not transfer holdout->eval."""
    nq, k = topk.shape
    cols, names = [], []
    for name in LEGS:
        M = legs[name]
        v = torch.gather(M, 1, topk)                       # raw leg score
        mx = M.max(1, keepdim=True).values
        mean = M.mean(1, keepdim=True)
        sd = M.std(1, keepdim=True) + 1e-6
        cols += [v, v - mx, (v - mean) / sd]
        names += [f'{name}', f'{name}_gapmax', f'{name}_z']
        # rank of each top-k candidate within this leg (count of higher-scoring candidates)
        r = torch.empty(nq, k, device=M.device)
        for s in range(0, nq, 256):
            e = min(s + 256, nq)
            r[s:e] = (M[s:e].unsqueeze(1) > v[s:e].unsqueeze(2)).sum(2).float()
        cols.append(torch.log1p(r))
        names.append(f'{name}_lograk')
    fv = torch.gather(fused, 1, topk)
    cols += [fv, fv - fused.max(1, keepdim=True).values,
             torch.arange(k, device=fv.device).float().expand(nq, k),
             n_photos[topk], (n_photos[topk] > 0).float()]
    names += ['fused', 'fused_gapmax', 'fused_rank', 'n_photos', 'has_bank']
    return torch.stack(cols, dim=2), names


def main():
    torch.set_num_threads(8)
    cache = os.path.join(OUT, 'rerank_v81_holdout.pt')
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=False)
        X, Y, topk, gold, gcls, names = (d['X'], d['Y'], d['topk'], d['gold'], d['gcls'], d['names'])
        print(f'loaded cache X={tuple(X.shape)}', flush=True)
    else:
        D = FishData()
        cand = D.cand
        legs, gold, n_photos = build_legs(D, cand)
        fused = sum(w * legs[n] for n, w in zip(LEGS, DEPLOYED_W))
        base1 = 100 * (fused.argmax(1) == gold).float().mean().item()
        print(f'deployed fused top-1 = {base1:.3f}', flush=True)
        topk = fused.topk(K, dim=1).indices
        X, names = pair_features(legs, topk, n_photos, fused)
        Y = (topk == gold[:, None]).float()
        gcls = np.array([D.classes[int(cand[g])] for g in gold.tolist()])
        torch.save({'X': X, 'Y': Y, 'topk': topk, 'gold': gold, 'gcls': gcls, 'names': names,
                    'base1': base1}, cache)
        print(f'wrote {cache}', flush=True)

    nq, k, nf = X.shape
    inK = Y.sum(1) > 0
    print(f'queries={nq} K={k} feats={nf} | gold in top-{k}: {100*inK.float().mean():.2f}%', flush=True)
    base_top1 = 100 * (Y[:, 0] > 0).float().mean().item()
    print(f'baseline (rank-1 of deployed fusion) = {base_top1:.3f}\n', flush=True)

    uniq = sorted(set(gcls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in gcls])

    Xf = X.reshape(nq * k, nf).numpy()
    Yf = Y.reshape(nq * k).numpy()
    qfold = np.repeat(folds, k)

    models = {
        'logistic': lambda: LogisticRegression(max_iter=3000, C=1.0),
        'hgb': lambda: HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                                      learning_rate=0.06, min_samples_leaf=40,
                                                      l2_regularization=1.0, random_state=0),
    }
    res = {}
    for name, mk in models.items():
        oof = np.zeros(nq * k)
        for f in range(NFOLD):
            tr, va = qfold != f, qfold == f
            mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
            m = mk()
            m.fit((Xf[tr] - mu) / sd, Yf[tr])
            oof[va] = m.predict_proba((Xf[va] - mu) / sd)[:, 1]
        sc = torch.tensor(oof).reshape(nq, k)
        pick = sc.argmax(1)
        acc = 100 * Y[torch.arange(nq), pick].mean().item()
        res[name] = acc
        print(f'{name:10s} re-ranked top-1 = {acc:6.3f}  ({acc - base_top1:+.3f} vs deployed)',
              flush=True)

    best = max(res, key=res.get)
    gain = res[best] - base_top1
    print(f'\nbest = {best}  {gain:+.3f} holdout-pt  (ceiling at K={k} is '
          f'{100*inK.float().mean().item() - base_top1:+.2f})', flush=True)

    if gain > 0:
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
        m = models[best]()
        m.fit((Xf - mu) / sd, Yf)
        pickle.dump({'model': m, 'kind': best, 'mu': mu, 'sd': sd, 'names': names, 'K': K,
                     'gain': gain, 'legs': LEGS, 'deployed_w': DEPLOYED_W},
                    open(os.path.join(OUT, 'rerank_unseen_v81.pkl'), 'wb'))
        print(f'wrote {OUT}/rerank_unseen_v81.pkl', flush=True)
    json.dump({'base_top1': base_top1, 'res': res, 'best': best, 'gain': gain, 'K': K},
              open(os.path.join(OUT, 'rerank_unseen_v81.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
