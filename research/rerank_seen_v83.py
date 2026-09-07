"""v83: learned re-ranker over the SEEN head's top-K. No leak is possible here.

The seen head is  sum_t w_t * zc( proto_t.q + 2.0*cmax_t + 4.0*taxon_t )  with w = 1.0/2.5/2.5.
Every constant is hand-picked and applied to every class identically. Measured holdout headroom:

    recall@1 89.87   @2 94.73   @3 96.36   @5 97.77   @10 98.85   @20 99.44

so ~9pt sits between rank-1 and top-10, on the coefficient worth 0.502 overall-pt per point --
the largest in the system.

Why this cannot leak the way v81 did: the seen-head candidate pool is the 4,636 kept classes and
the gold of a val_seen image is ALWAYS in it. Every candidate is gold-eligible by construction, so
there is no identifiable subpopulation to prefer. (v81's pool was 9.1% eligible; that was the bug.)

The candidate-conditional signal the fixed weights cannot express: `cmax` is the nearest single
training image and `proto` is the class mean. For a class with 1 training image they are identical;
for a class with 100 they measure different things. A fixed 2.0 cannot know that -- a model given
class frequency can.

Features are pool-size invariant (percentile ranks), as in v82.

  conda activate onet && python research/rerank_seen_v83.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learned_gate_v77 import (  # noqa: E402
    LAM, MEMBERS, NFOLD, OUT, dev, holdout_split, load_train_embs, zc,
)

K = 10
ACTIVE = [t for t, _, w in MEMBERS if w > 0]
SIGNALS = [f'{t}_{c}' for t in ACTIVE for c in ('proto', 'cmax', 'taxon')] + ['block']
CACHE = f'{OUT}/rerank_seen_v83_holdout.pt'


def components(Q, P, TF, TL, Tseen, S):
    """proto / cmax / taxon separately -- the seen head sums them with fixed 1 : 2 : 4."""
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
    """Per (query, candidate) features; all within-query and pool-size invariant."""
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
    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt',
                                 weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[torch.tensor([ci[c] for c in kept], device=dev)]
    n_train = torch.tensor([len(trby[c]) for c in kept], dtype=torch.float32, device=dev)

    sig, block = {}, torch.zeros(ns, S, device=dev)
    for t, hs, w in MEMBERS:
        if w == 0:
            continue
        idx, feats, _ = train[t]
        P = torch.zeros(S, feats.shape[1])
        cnt = torch.zeros(S)
        TF, TL = [], []
        for c in kept:
            for fn in trby[c]:
                f = feats[idx[fn]]
                P[k2i[c]] += f
                cnt[k2i[c]] += 1
                TF.append(f)
                TL.append(k2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
        Q = torch.stack([feats[idx[fn]] for fn in files]).to(dev)
        ps, cm, tx = components(Q, P, torch.stack(TF).to(dev), torch.tensor(TL, device=dev), Tseen, S)
        sig[f'{t}_proto'], sig[f'{t}_cmax'], sig[f'{t}_taxon'] = ps, cm, tx
        block += w * zc(ps + 2.0 * cm + (LAM * tx if hs else 0))
        print(f'  {t} done', flush=True)
    sig['block'] = block

    topk = block.topk(K, dim=1).indices
    X, names = cand_features(sig, topk, n_train, block)
    Y = (topk == gold[:, None]).float()
    return X.cpu(), Y.cpu(), np.array(sp['cls'][:ns]), names


def main():
    torch.set_num_threads(8)
    if os.path.exists(CACHE):
        d = torch.load(CACHE, weights_only=False)
        X, Y, cls, names = d['X'], d['Y'], d['cls'], d['names']
        print(f'loaded cache X={tuple(X.shape)}', flush=True)
    else:
        X, Y, cls, names = build_holdout()
        torch.save({'X': X, 'Y': Y, 'cls': cls, 'names': names}, CACHE)
        print(f'wrote {CACHE}', flush=True)

    nq, k, nf = X.shape
    base = 100 * (Y[:, 0] > 0).float().mean().item()
    print(f'\nqueries={nq} K={k} feats={nf} | gold in top-{k}: '
          f'{100*(Y.sum(1) > 0).float().mean():.2f}%')
    print(f'seen head rank-1 (deployed) = {base:.3f}   [real a_cond 86.91%]\n', flush=True)

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
    gain = acc - base
    print(f'\ndeployed rank-1     {base:6.3f}')
    print(f'LEARNED re-ranker   {acc:6.3f}  ({gain:+.3f})')
    print(f'ceiling at K={k}      {100*(Y.sum(1) > 0).float().mean():6.3f}', flush=True)

    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    m = LogisticRegression(max_iter=4000, C=1.0)
    m.fit((Xf - mu) / sd, Yf)
    pickle.dump({'model': m, 'mu': mu, 'sd': sd, 'names': names, 'K': K, 'gain': gain,
                 'signals': SIGNALS},
                open(f'{OUT}/rerank_seen_v83.pkl', 'wb'))
    print('\ntop coefficients:', flush=True)
    for n, c in sorted(zip(names, m.coef_[0]), key=lambda t: -abs(t[1]))[:12]:
        print(f'  {n:24s} {c:+.4f}', flush=True)
    json.dump({'base': base, 'reranked': acc, 'gain': gain, 'K': K},
              open(f'{OUT}/rerank_seen_v83.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/rerank_seen_v83.pkl', flush=True)


if __name__ == '__main__':
    main()
