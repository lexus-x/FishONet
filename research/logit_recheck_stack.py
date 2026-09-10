"""logit_recheck stage 2: recall@10 improved at tau=+0.25 (98.854 -> 98.904, +0.051pt,
research/logit_recheck_recall10.py). Retrain the seen re-ranker leak-free (identical
methodology to rerank_seen_v83.py: LogisticRegression over pool-size-invariant
proto/cmax/taxon percentile-rank features, K=10, 5-fold class-disjoint CV) but with
candidates drawn from the tau-adjusted fusion (block += tau*log(n_train) before topk)
instead of the deployed fusion. Report true stack-level accuracy vs deployed 91.674
(bar to beat: >=92.674, i.e. +1.0pt).

  conda activate onet && python research/logit_recheck_stack.py
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
from learned_gate_v77 import LAM, MEMBERS, NFOLD, OUT, dev, holdout_split, load_train_embs, zc  # noqa: E402
from rerank_seen_v83 import SIGNALS, K, cand_features, components  # noqa: E402

TAU = 0.25
CACHE = f'{OUT}/logit_recheck_stack_tau{TAU}_holdout.pt'


def build_holdout(tau):
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
    log_n = torch.log(n_train.clamp(min=1))

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
    block = block + tau * log_n  # <-- the ONE change: tau-adjusted fusion picks the candidates
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
        X, Y, cls, names = build_holdout(TAU)
        torch.save({'X': X, 'Y': Y, 'cls': cls, 'names': names}, CACHE)
        print(f'wrote {CACHE}', flush=True)

    nq, k, nf = X.shape
    base = 100 * (Y[:, 0] > 0).float().mean().item()
    print(f'\nqueries={nq} K={k} feats={nf} | gold in top-{k} (tau={TAU} fusion): '
          f'{100*(Y.sum(1) > 0).float().mean():.2f}%')
    print(f'tau={TAU} fusion rank-1 = {base:.3f}   [deployed tau=0 real a_cond 86.91%]\n', flush=True)

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
    print(f'\ntau={TAU} fusion rank-1        {base:6.3f}')
    print(f'LEARNED re-ranker (tau fusion) {acc:6.3f}')
    print(f'ceiling at K={k}                 {100*(Y.sum(1) > 0).float().mean():6.3f}', flush=True)

    DEPLOYED_STACK = 91.674
    BAR = 92.674
    delta = acc - DEPLOYED_STACK
    verdict = 'CLEARS BAR (+1.0pt)' if acc >= BAR else 'DOES NOT CLEAR BAR'
    print(f'\ndeployed stack (tau=0, real)   {DEPLOYED_STACK:6.3f}')
    print(f'this variant stack (holdout)   {acc:6.3f}  ({delta:+.3f})')
    print(f'bar to beat                    {BAR:6.3f}')
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'tau': TAU, 'base': base, 'reranked': acc, 'deployed_stack': DEPLOYED_STACK,
               'delta': delta, 'bar': BAR, 'verdict': verdict, 'K': K},
              open(f'{OUT}/logit_recheck_stack.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/logit_recheck_stack.json', flush=True)


if __name__ == '__main__':
    main()
