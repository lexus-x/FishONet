"""v99: widen the seen re-ranker's candidate depth K (research/rerank_seen_v83.py) from 10 to
15 and 20. Same 32-ish pool-size-invariant features, same 5-fold CV LogisticRegression, same
holdout split -- only K changes. Tests whether the model can still discriminate correctly as
more (harder, lower-margin) candidates enter the pool, exploiting more of the recall@K headroom
that v83's own docstring flags as unexploited:

    recall@1 89.87   @2 94.73   @3 96.36   @5 97.77   @10 98.85   @20 99.44

v83 at K=10 converts that into 91.67% (+1.803pt over 89.87 base). Widening K raises the ceiling
(99.44 at K=20) but also gives the logistic model more distractors per query to rank correctly,
so the effect on realized accuracy (not just ceiling) is the open question -- untested today.

Reuses build_holdout()'s expensive, K-INDEPENDENT half (load 3 member embeddings, build per-class
prototypes, run BOTH the proto/cmax/taxon components AND the block fusion) exactly ONCE via
build_signals() below (a straight copy of rerank_seen_v83.build_holdout() with the K-dependent
tail cut off), then re-derives topk / features / labels / CV-fit per K so the GPU embedding pass
isn't paid 3x. Calls rerank_seen_v83.components() and .cand_features() directly -- no duplicated
math.

PASS BAR (per task): >=+1.0pt over the already-real-tested-negative 91.21% linear reweight,
i.e. >= 92.21% holdout accuracy at some K, to be worth a submission slot. (v83's own K=10 number,
91.67%, already sits below this bar -- so the interesting outcome is whether K=15 or K=20 clears
it, not whether it beats K=10.)

  conda activate onet && python research/rerank_seen_v99_widek.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learned_gate_v77 import LAM, MEMBERS, NFOLD, OUT, dev, holdout_split, load_train_embs, zc  # noqa: E402
from rerank_seen_v83 import cand_features, components  # noqa: E402

SIG_CACHE = f'{OUT}/rerank_seen_v99_sigblock.pt'
KS = [10, 15, 20]
PASS_BAR = 92.21


def build_signals():
    """K-independent half of rerank_seen_v83.build_holdout(): sig dict, block, gold, n_train, cls."""
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
    return sig, block, gold, n_train, np.array(sp['cls'][:ns])


def eval_k(K, sig, block, gold, n_train, cls):
    topk = block.topk(K, dim=1).indices
    X, names = cand_features(sig, topk, n_train, block)
    X, Y = X.cpu(), (topk == gold[:, None]).float().cpu()
    nq, k, nf = X.shape
    base = 100 * (Y[:, 0] > 0).float().mean().item()
    ceiling = 100 * (Y.sum(1) > 0).float().mean().item()

    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cls])
    qfold = np.repeat(folds, k)
    Xf = X.reshape(nq * k, nf).cpu().numpy()
    Yf = Y.reshape(nq * k).cpu().numpy()

    oof = np.zeros(nq * k)
    for f in range(NFOLD):
        tr, va = qfold != f, qfold == f
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
        m = LogisticRegression(max_iter=4000, C=1.0)
        m.fit((Xf[tr] - mu) / sd, Yf[tr])
        oof[va] = m.decision_function((Xf[va] - mu) / sd)
    sc = torch.tensor(oof).reshape(nq, k)
    acc = 100 * Y[torch.arange(nq), sc.argmax(1)].mean().item()
    return base, acc, ceiling


def main():
    torch.set_num_threads(8)
    if os.path.exists(SIG_CACHE):
        d = torch.load(SIG_CACHE, weights_only=False)
        sig, block, gold, n_train, cls = d['sig'], d['block'], d['gold'], d['n_train'], d['cls']
        print(f'loaded cache {SIG_CACHE}', flush=True)
    else:
        sig, block, gold, n_train, cls = build_signals()
        torch.save({'sig': sig, 'block': block, 'gold': gold, 'n_train': n_train, 'cls': cls},
                   SIG_CACHE)
        print(f'wrote {SIG_CACHE}', flush=True)

    print(f'\npass bar: >= {PASS_BAR}%\n', flush=True)
    results = {}
    for K in KS:
        base, acc, ceiling = eval_k(K, sig, block, gold, n_train, cls)
        results[K] = acc
        verdict = 'CLEARS' if acc >= PASS_BAR else 'below bar'
        print(f'K={K:2d}  base(rank-1)={base:.3f}  reranked={acc:.3f} ({acc - base:+.3f})  '
              f'ceiling(recall@{K})={ceiling:.3f}  [{verdict}]', flush=True)

    if 10 in results:
        for K in KS:
            if K != 10:
                print(f'K={K} vs K=10 reranked: {results[K] - results[10]:+.3f}pt', flush=True)


if __name__ == '__main__':
    main()
