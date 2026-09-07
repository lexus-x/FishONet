"""v78: v77's learned gate + 5 view-disagreement features.

HANDOFF's root cause for the whole shift problem is framing: training crops are aspect ~1.15,
eval fish ~2.12. v77's gate has no shift feature at all -- it reads one framing of each image.
The 7 crop views v56 already computes give a direct read: if an image's answer SURVIVES
re-cropping, the evidence is real; if it swings, the single-view score is an artifact of framing.

    view_agree      mean cos(view_v, mean view) -- embedding-cloud concentration
    sb_view_std     std over views of the seen-head max
    sb_view_gain    best-view seen max - center-view seen max
    bank_view_std   std over views of the iNat bank max
    bank_view_gain  best-view bank max - center-view bank max

Same evaluation as v77: class-disjoint 5-fold CV, f=0.60 pinned, deployment-weighted.
Needs research/extract_holdout_views_v78.py to have run.

  conda activate onet && python research/learned_gate_v78.py
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
    BANK_CTFT, FEATS, KILL, LAM, NFOLD, OUT, TOPM, auc, deploy_weights, dev, holdout_split,
    load_train_embs, operating_point, seen_score, standardize,
)

VIEW_FEATS = ['view_agree', 'sb_view_std', 'sb_view_gain', 'bank_view_std', 'bank_view_gain']
FEATS78 = FEATS + VIEW_FEATS
HOLD_VIEWS = f'{OUT}/emb_holdout_all_ctft_views7.pt'
VIEW_KEYS = ['center', 'squash'] + [f'ostrip{i}' for i in range(5)]


def bank_perview_max(view_list, bank, other_list):
    """max-over-classes of the top-TOPM-mean bank score, per view. [V, N]."""
    V, Nq = len(view_list), view_list[0].shape[0]
    Qcat = torch.cat(view_list, dim=0)
    best = torch.full((V, Nq), -1e4, device=dev)
    for gidx in other_list:
        photos = bank.get(gidx)
        if photos is None or (hasattr(photos, 'numel') and photos.numel() == 0):
            continue
        if not isinstance(photos, torch.Tensor):
            photos = torch.stack(photos)
        photos = F.normalize(photos.float(), dim=-1).to(dev)
        pooled = (Qcat @ photos.t()).topk(min(TOPM, photos.shape[0]), dim=1).values.mean(dim=1)
        best = torch.maximum(best, pooled.view(V, Nq))
    return best


def view_features(view_list, P, TF, TL, Tseen, S, bank_perview):
    """The 5 view features. view_list[0] MUST be the center view."""
    SB = torch.stack([seen_score(V, P, TF, TL, True, Tseen, S).max(1).values for V in view_list])
    M = F.normalize(torch.stack(view_list).mean(0), dim=-1)
    agree = torch.stack([(V * M).sum(-1) for V in view_list]).mean(0)
    return {
        'view_agree': agree,
        'sb_view_std': SB.std(0),
        'sb_view_gain': SB.max(0).values - SB[0],
        'bank_view_std': bank_perview.std(0),
        'bank_view_gain': bank_perview.max(0).values - bank_perview[0],
    }


def build_view_holdout():
    d = torch.load(HOLD_VIEWS, weights_only=False)
    files = d['files']
    views = {k: F.normalize(d['views'][k].float(), dim=-1).to(dev) for k in VIEW_KEYS}

    train, tb, b2f, b2l = load_train_embs()
    sp = holdout_split(train, tb, b2f, b2l)
    assert sp['val_files'] == files, 'holdout view file order drifted from the split'
    classes, ci, kept, trby = sp['classes'], sp['ci'], sp['kept'], sp['trby']
    k2i = {c: i for i, c in enumerate(kept)}
    Sk = len(kept)
    kept_idx = torch.tensor([ci[c] for c in kept], device=dev)
    other_idx = torch.tensor([i for i in range(len(classes)) if classes[i] not in set(kept)], device=dev)
    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)

    idx, feats, _ = train['ctftshift']
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
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
    TF = torch.stack(TF).to(dev)
    TL = torch.tensor(TL, device=dev)

    vl = [views[k] for k in VIEW_KEYS]
    print('scoring per-view iNat bank ...', flush=True)
    bpv = bank_perview_max(vl, torch.load(BANK_CTFT, weights_only=False)['bank'], other_idx.tolist())
    fd = view_features(vl, P, TF, TL, TtH[kept_idx], Sk, bpv)
    return torch.stack([fd[k] for k in VIEW_FEATS], dim=1).cpu()


def main():
    torch.set_num_threads(8)
    base_cache = f'{OUT}/gate_feats_holdout_v77.pt'
    if not os.path.exists(base_cache):
        raise SystemExit('run research/learned_gate_v77.py first (needs the 12-feature cache)')
    b = torch.load(base_cache, weights_only=False)
    X12, y, cls = b['X'], b['y'], b['cls']

    vcache = f'{OUT}/gate_feats_holdout_v78_views.pt'
    if os.path.exists(vcache):
        Xv = torch.load(vcache, weights_only=False)['Xv']
        print(f'loaded cached view features {tuple(Xv.shape)}', flush=True)
    else:
        Xv = build_view_holdout()
        torch.save({'Xv': Xv, 'feats': VIEW_FEATS}, vcache)
        print(f'wrote {vcache}', flush=True)
    assert Xv.shape[0] == X12.shape[0]
    X = torch.cat([X12, Xv], dim=1)

    w = deploy_weights(y)
    Z = standardize(X, w)
    base = Z[:, FEATS78.index('sb_max')] + 2.0 * Z[:, FEATS78.index('tm_ctft')]
    n12 = len(FEATS)

    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cls])

    arms = {'v77 (12 feat)': list(range(n12)), 'v78 (17 feat)': list(range(len(FEATS78)))}
    oof = {k: torch.zeros(len(y)) for k in arms}
    for f in range(NFOLD):
        tr, va = folds != f, folds == f
        Ztr = standardize(X[tr], deploy_weights(y[tr]))
        Zva = standardize(X[va], deploy_weights(y[va]))
        for name, cols in arms.items():
            m = LogisticRegression(max_iter=2000, C=1.0)
            m.fit(Ztr[:, cols].numpy(), y[tr], sample_weight=deploy_weights(y[tr]).numpy())
            oof[name][torch.tensor(va)] = torch.tensor(
                m.predict_proba(Zva[:, cols].numpy())[:, 1], dtype=torch.float32)
        print(f'  fold {f}: train={tr.sum()} val={va.sum()}', flush=True)

    rows = {}
    bt, bn, bp = operating_point(base, y, w)
    rows['v56_gate'] = {'auc': auc(base, y, w), 'tpr': bt, 'tnr': bn, 'proj': bp}
    for name in arms:
        t, n, p = operating_point(oof[name], y, w)
        rows[name] = {'auc': auc(oof[name], y, w), 'tpr': t, 'tnr': n, 'proj': p}

    print('\n=== class-disjoint 5-fold CV, f=0.60 pinned ===', flush=True)
    print(f'{"gate":14s} {"AUC":>8s} {"TPR":>8s} {"TNR":>8s} {"proj%":>8s} {"vs v56":>8s} {"vs v77":>8s}',
          flush=True)
    p77 = rows['v77 (12 feat)']['proj']
    for name, r in rows.items():
        print(f'{name:14s} {r["auc"]:8.4f} {100*r["tpr"]:8.2f} {100*r["tnr"]:8.2f} '
              f'{r["proj"]:8.3f} {r["proj"]-bp:+8.3f} {r["proj"]-p77:+8.3f}', flush=True)

    gain = rows['v78 (17 feat)']['proj'] - p77
    print(f'\nv78 vs v77 = {gain:+.3f} proxy-pt (kill bar {KILL:+.2f}) -> '
          f'{"CLEARS" if gain >= KILL else "FAILS"}', flush=True)

    m = LogisticRegression(max_iter=2000, C=1.0)
    m.fit(Z.numpy(), y, sample_weight=w.numpy())
    with open(f'{OUT}/learned_gate_v78.pkl', 'wb') as fh:
        pickle.dump({'model': m, 'kind': 'logistic', 'feats': FEATS78, 'cv': rows,
                     'delta_vs_v77': gain, 'clears_kill': bool(gain >= KILL),
                     'seen_frac': 0.60}, fh)
    json.dump({'cv': rows, 'delta_vs_v77': gain, 'clears_kill': bool(gain >= KILL),
               'feats': FEATS78}, open(f'{OUT}/learned_gate_v78.json', 'w'), indent=2)
    print('\ncoefficients (standardized):', flush=True)
    for n, c in sorted(zip(FEATS78, m.coef_[0]), key=lambda t: -abs(t[1])):
        mark = '  <- new' if n in VIEW_FEATS else ''
        print(f'  {n:20s} {c:+.4f}{mark}', flush=True)
    print(f'\nwrote {OUT}/learned_gate_v78.pkl + .json', flush=True)


if __name__ == '__main__':
    main()
