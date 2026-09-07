"""v86: give the SEEN re-ranker the encoders it has never seen.

The asymmetry: the unseen head fuses SIX encoders (3x BioCLIP-2.5 + TaxaBind + BioCLIP-2 frozen +
BioCLIP-2 LoRA). The seen head uses THREE (BioCLIP-2.5 ctftshift / ftshift / fullft336shift). The
seen re-ranker therefore reads 32 features that are all functions of those same three encoders.
TaxaBind and BioCLIP-2 never score a seen class anywhere in the pipeline.

v84 established that the residual seen-head gap (91.77 achieved vs 98.85 ceiling@10 = 7.08 proxy-pt)
is NOT a model-capacity limit -- HistGradientBoosting, which can fit any interaction of the existing
32 features, gained +0.093. So the gap needs new EVIDENCE. Two whole encoders that already have
train embeddings on disk, and have never been applied to this head, are the cheapest new evidence
available: no download, no fine-tune, no new external data.

Per-class prototypes and nearest-training-image (cmax) are built under each extra encoder exactly
as the deployed head does for its own three, then exposed as re-ranker features. The RETRIEVER is
unchanged (shortlist still comes from the deployed `block`), so this can only re-order the existing
top-10 -- no coverage or compliance change.

Kill bar: anchor 0.0295 real-pt per proxy-pt on this head, so +0.10 real needs +3.4 proxy-pt over
the deployed +1.803.

  conda activate onet && python research/rerank_seen_v86_encoders.py
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
from rerank_seen_v83 import K, components  # noqa: E402

CACHE = f'{OUT}/rerank_seen_v86_holdout.pt'
DEPLOYED_GAIN = 1.803
ANCHOR = 0.0295
EXTRA = ['taxabind', 'b2frozen', 'b2lora']


def class_signals(idx, feats, kept, k2i, trby, val_files, S):
    """proto + cmax for one encoder, same statistics the deployed seen head uses."""
    feats = F.normalize(feats.float(), dim=-1)
    dim = feats.shape[1]
    P = torch.zeros(S, dim)
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
    Q = torch.stack([feats[idx[fn]] for fn in val_files]).to(dev)
    ps, cm, _ = components(Q, P, torch.stack(TF).to(dev), torch.tensor(TL, device=dev), P, S)
    return ps, cm


def feats_for(sig, signals, topk, n_train, block):
    nq, k = topk.shape
    C = block.shape[1]
    cols, names = [], []
    for name in signals:
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


def build():
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
    base_signals = []
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
        base_signals += [f'{t}_proto', f'{t}_cmax', f'{t}_taxon']
        block += w * zc(ps + 2.0 * cm + (LAM * tx if hs else 0))
        print(f'  base {t} done', flush=True)
    sig['block'] = block
    base_signals.append('block')

    extra_signals = []
    for tag, pack in zip(EXTRA, (tb, b2f, b2l)):
        idx, feats, _ = pack
        ps, cm = class_signals(idx, feats, kept, k2i, trby, files, S)
        sig[f'{tag}_proto'], sig[f'{tag}_cmax'] = ps, cm
        extra_signals += [f'{tag}_proto', f'{tag}_cmax']
        print(f'  EXTRA {tag} done', flush=True)

    topk = block.topk(K, dim=1).indices
    Y = (topk == gold[:, None]).float()
    Xb, nb = feats_for(sig, base_signals, topk, n_train, block)
    Xa, na = feats_for(sig, base_signals + extra_signals, topk, n_train, block)
    return (Xb.cpu(), nb, Xa.cpu(), na, Y.cpu(), np.array(sp['cls'][:ns]))


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
    if os.path.exists(CACHE):
        d = torch.load(CACHE, weights_only=False)
        Xb, nb, Xa, na, Y, cls = d['Xb'], d['nb'], d['Xa'], d['na'], d['Y'], d['cls']
        print(f'loaded cache base={tuple(Xb.shape)} all={tuple(Xa.shape)}', flush=True)
    else:
        Xb, nb, Xa, na, Y, cls = build()
        torch.save({'Xb': Xb, 'nb': nb, 'Xa': Xa, 'na': na, 'Y': Y, 'cls': cls}, CACHE)
        print(f'wrote {CACHE}', flush=True)

    nq, k, _ = Xb.shape
    base = 100 * (Y[:, 0] > 0).float().mean().item()
    ceil = 100 * (Y.sum(1) > 0).float().mean().item()
    print(f'\nqueries={nq} K={k} | deployed rank-1 {base:.3f} | ceiling@{k} {ceil:.3f}', flush=True)

    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cls])

    print(f'\n{"arm":<22} {"feats":>6} {"top-1":>7} {"gain":>7} {"vs v83":>7} {"proj real":>10}')
    print('-' * 64)
    res = {}
    for arm, (X, names) in {'base (3 encoders)': (Xb, nb),
                            '+TaxaBind +B2 x2': (Xa, na)}.items():
        acc = cv(X, Y, folds)
        gain, delta = acc - base, (acc - base) - DEPLOYED_GAIN
        res[arm] = {'acc': acc, 'gain': gain, 'proj': delta * ANCHOR, 'nfeat': X.shape[2]}
        print(f'{arm:<22} {X.shape[2]:>6} {acc:>7.3f} {gain:>+7.3f} {delta:>+7.3f} '
              f'{delta * ANCHOR:>+10.3f}', flush=True)

    a, b_ = res['base (3 encoders)'], res['+TaxaBind +B2 x2']
    lift = b_['gain'] - a['gain']
    print(f'\nnew-encoder lift: {lift:+.3f} proxy-pt = {lift * ANCHOR:+.3f} real overall-pt')
    print('verdict: ' + ('WORTH BUILDING' if b_['proj'] >= 0.05 else 'below bar'), flush=True)

    if b_['gain'] > a['gain']:
        nq, k, nf = Xa.shape
        Xf, Yf = Xa.reshape(nq * k, nf).numpy(), Y.reshape(nq * k).numpy()
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
        m = LogisticRegression(max_iter=4000, C=1.0)
        m.fit((Xf - mu) / sd, Yf)
        pickle.dump({'model': m, 'mu': mu, 'sd': sd, 'names': na, 'K': k, 'gain': b_['gain'],
                     'extra': EXTRA},
                    open(f'{OUT}/rerank_seen_v86.pkl', 'wb'))
        print(f'wrote {OUT}/rerank_seen_v86.pkl')
        print('\ntop coefficients:')
        for n, c in sorted(zip(na, m.coef_[0]), key=lambda t: -abs(t[1]))[:14]:
            mark = '  <-- NEW ENCODER' if n.split('_')[0] in EXTRA else ''
            print(f'  {n:26s} {c:+.4f}{mark}')
    json.dump(res, open(f'{OUT}/rerank_seen_v86.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
