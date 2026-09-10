"""v90 phase-1 analysis: does framing shift degrade the learned gate, and does
shift-jitter training recover it?

Reads the shift-rendered ctftshift holdout embeddings (extract_holdout_shift_v90.py),
rebuilds the 12 v77 gate features with the ctftshift member substituted (protos and
all other members stay at train framing, exactly mirroring deployment where class
prototypes come from training images and only the QUERY is eval-framed), then runs
one class-disjoint 5-fold CV measuring every (train_mode x eval_render) AUC:

  train on norm            -> eval on norm / s135 / s175   (degradation = mechanism)
  train on pooled 3-render -> eval on norm / s135 / s175   (recovery = the fix)

Pre-registered verdict:
  MECHANISM CONFIRMED if norm->s175 AUC drop >= 0.010 AND pooled recovers >= half.
  DEAD if norm->shift drop < 0.005 (geometric stretch does not move gate features;
  content-level shift may still exist but is not fixable by re-rendering).

Partial simulation caveats (both UNDERSTATE the real effect): only 1 of 3 shifted
members (ctftshift w=1.0 of 6.0 in sb_*, but it alone feeds tm_ctft + bank_* +
m_ctftshift = 4 of 12 features); squash TTA views are stretch-invariant.

  conda activate onet && python research/gate_shift_adapt_v90.py
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
from common import OUT
from extract_holdout_shift_v90 import load_train_embs_v90
from learned_gate_v77 import (
    BANK_CTFT, B2_PROTO_FROZEN, B2_PROTO_LORA, FEATS, MEMBERS, NFOLD, assemble, auc,
    bank_scores, dbnorm, deploy_weights, dev, holdout_split,
    operating_point, proto_max, seen_score, standardize, zc,
)

RENDERS = {'norm': None,
           's135': f'{OUT}/emb_holdout_ctft_stretch135.pt',
           's175': f'{OUT}/emb_holdout_ctft_stretch175.pt'}
C = 100.0   # v79 refit


def build_feats(sub_path):
    """v77 build_holdout with the ctftshift query embeddings optionally substituted."""
    import pickle
    from collections import defaultdict
    D = 'data/dl'
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(f'{D}/label_train.json'))
    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(torch.load(f'{OUT}/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load(f'{OUT}/text_emb_h_promptens.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load(f'{OUT}/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)

    train, tb_train, b2f_train, b2l_train = load_train_embs_v90()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    val_files, y, cls = sp['val_files'], sp['y'], sp['cls']
    kept, trby, ci = sp['kept'], sp['trby'], sp['ci']
    k2i = {c: i for i, c in enumerate(kept)}
    Sk = len(kept)
    kept_idx = torch.tensor([ci[c] for c in kept], device=dev)
    other_idx = torch.tensor([i for i in range(len(classes)) if classes[i] not in set(kept)], device=dev)

    def protos(t):
        idx, feats, _ = train[t]
        P = torch.zeros(Sk, feats.shape[1])
        cnt = torch.zeros(Sk)
        TF, TL = [], []
        for c in kept:
            for fn in trby[c]:
                f = feats[idx[fn]]
                P[k2i[c]] += f
                cnt[k2i[c]] += 1
                TF.append(f)
                TL.append(k2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
        return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL, device=dev)

    PR = {t: protos(t) for t, _, _ in MEMBERS}
    Q = {t: torch.stack([train[t][1][train[t][0][fn]] for fn in val_files]).to(dev) for t, _, _ in MEMBERS}
    if sub_path is not None:
        d = torch.load(sub_path, weights_only=False)
        assert d['files'] == val_files, 'row ordering mismatch vs extractor'
        Q['ctftshift'] = F.normalize(d['feats'].float(), dim=-1).to(dev)
    Qb2f = torch.stack([b2f_train[1][b2f_train[0][fn]] for fn in val_files]).to(dev)
    Qb2l = torch.stack([b2l_train[1][b2l_train[0][fn]] for fn in val_files]).to(dev)
    Qtb = torch.stack([tb_train[1][tb_train[0][fn]] for fn in val_files]).to(dev)
    Tseen = TtH[kept_idx]

    seen_block = torch.zeros(len(val_files), Sk, device=dev)
    member_max = {}
    for t, hs, w in MEMBERS:
        if w == 0:
            continue
        sc = seen_score(Q[t], *PR[t], hs, Tseen, Sk)
        member_max[t] = sc.max(1).values
        seen_block += w * zc(sc)

    text_full = (dbnorm(Q['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
                 + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
                 + 1.0 * dbnorm(Q['ctftshift'] @ TTX.t()) + 1.0 * dbnorm(Qtb @ Ttb.t()))
    tm_ctft = ((Q['ctftshift'] @ TtH[kept_idx].t()).max(1).values
               - (Q['ctftshift'] @ TtH[other_idx].t()).max(1).values)
    tm_full = text_full[:, kept_idx].max(1).values - text_full[:, other_idx].max(1).values
    del text_full

    bank_raw = bank_scores(Q['ctftshift'], torch.load(BANK_CTFT, weights_only=False)['bank'], other_idx.tolist())
    b2f = proto_max(Qb2f, B2_PROTO_FROZEN, other_idx)
    b2l = proto_max(Qb2l, B2_PROTO_LORA, other_idx)

    X = assemble(seen_block, member_max, tm_ctft, tm_full, bank_raw, b2f, b2l)
    return X.cpu(), y, cls


def main():
    torch.set_num_threads(8)
    Xs = {}
    y = cls = None
    for name, sub in RENDERS.items():
        cache = f'{OUT}/gate_feats_holdout_v90_{name}.pt'
        if os.path.exists(cache):
            d = torch.load(cache, weights_only=False)
            Xs[name], y, cls = d['X'], d['y'], d['cls']
            print(f'loaded cached {name} {tuple(Xs[name].shape)}', flush=True)
            continue
        if sub is not None and not os.path.exists(sub):
            raise SystemExit(f'missing {sub} — run extract_holdout_shift_v90.py first')
        print(f'building features [{name}] ...', flush=True)
        Xs[name], y, cls = build_feats(sub)
        torch.save({'X': Xs[name], 'y': y, 'cls': cls, 'feats': FEATS}, cache)
        print(f'wrote {cache}', flush=True)

    # sanity: norm rebuild should match the v77 cache (informational — the b2l
    # fallback and today's emb_train_fullft336_v2 regeneration can both shift it)
    v77 = torch.load(f'{OUT}/gate_feats_holdout_v77.pt', weights_only=False)
    if v77['X'].shape == Xs['norm'].shape:
        dmax = (Xs['norm'] - v77['X']).abs().max().item()
        print(f'norm vs v77 cache: max|diff| = {dmax:.2e}', flush=True)
    else:
        print(f'norm {tuple(Xs["norm"].shape)} vs v77 cache {tuple(v77["X"].shape)} — '
              f'shape mismatch (b2l fallback changed the split); internal comparison still valid',
              flush=True)

    w = deploy_weights(y)
    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cls])

    train_modes = {'norm': ['norm'], 'pooled': ['norm', 's135', 's175']}
    oof = {(tm, ev): torch.zeros(len(y)) for tm in train_modes for ev in RENDERS}
    for f in range(NFOLD):
        tr, va = folds != f, folds == f
        wtr = deploy_weights(y[tr])
        Zva = {ev: standardize(Xs[ev][va], deploy_weights(y[va])) for ev in RENDERS}
        for tm, parts in train_modes.items():
            Ztr = torch.cat([standardize(Xs[p][tr], wtr) for p in parts])
            ytr = np.concatenate([y[tr]] * len(parts))
            swt = np.concatenate([wtr.numpy()] * len(parts))
            m = LogisticRegression(max_iter=2000, C=C)
            m.fit(Ztr.numpy(), ytr, sample_weight=swt)
            for ev in RENDERS:
                oof[(tm, ev)][torch.tensor(va)] = torch.tensor(
                    m.predict_proba(Zva[ev].numpy())[:, 1], dtype=torch.float32)
        print(f'  fold {f} done', flush=True)

    print(f'\n=== class-disjoint 5-fold CV, C={C:g}, f=0.60 pinned ===', flush=True)
    print(f'{"train":8s} {"eval":6s} {"AUC":>8s} {"TPR":>8s} {"TNR":>8s} {"proj%":>8s}', flush=True)
    res = {}
    for (tm, ev), sc in oof.items():
        a = auc(sc, y, w)
        t, n, p = operating_point(sc, y, w)
        res[f'{tm}->{ev}'] = {'auc': a, 'tpr': t, 'tnr': n, 'proj': p}
        print(f'{tm:8s} {ev:6s} {a:8.4f} {100 * t:8.2f} {100 * n:8.2f} {p:8.3f}', flush=True)

    drop = res['norm->norm']['auc'] - res['norm->s175']['auc']
    rec = res['pooled->s175']['auc'] - res['norm->s175']['auc']
    print(f'\nnorm->s175 AUC drop = {drop:+.4f}   pooled recovery = {rec:+.4f}', flush=True)
    if drop >= 0.010 and rec >= drop / 2:
        verdict = 'MECHANISM CONFIRMED — phase 2 (all members + re-rankers) justified'
    elif drop < 0.005:
        verdict = 'DEAD — geometric stretch does not move the gate features'
    else:
        verdict = 'WEAK — drop present but jitter training does not recover it'
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'results': res, 'drop_s175': drop, 'recovery_s175': rec, 'verdict': verdict},
              open(f'{OUT}/gate_shift_adapt_v90.json', 'w'), indent=2)
    print(f'wrote {OUT}/gate_shift_adapt_v90.json', flush=True)


if __name__ == '__main__':
    main()
