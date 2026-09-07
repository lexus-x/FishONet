"""v80: does NEW information move the gate, where re-weighting did not?

v77->v79 established the rule: re-weighting the same 12 features bought +0.481 routing and gave
back -0.439 in conditionals (net +0.042). Only new evidence compounds. The v77 gate reads exactly
one image bank (ctftshift) and one text margin (ctftshift taxon) -- yet the deployed unseen head
also uses the fullft336shift bank (W_336=3) and a TaxaBind text leg, both orthogonal encoders the
gate has never seen.

    bank336_max/margin   second, different encoder's image->image evidence
    tm_tb                TaxaBind seen-vs-unseen text margin

Same protocol as v77/v79: class-disjoint 5-fold CV, f=0.60 pinned, C=100, deployment-weighted.
Kill bar +0.30 proxy-pt vs the v79 (12-feature, C=100) arm.

  conda activate onet && python research/learned_gate_v80.py
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
    FEATS, KILL, NFOLD, OUT, auc, bank_scores, deploy_weights, dev, holdout_split,
    load_train_embs, operating_point, standardize,
)

C = 100.0
BANK_336 = f'{OUT}/inat_photo_bank_fullft336shift.pt'
NEW = ['bank336_max', 'bank336_margin', 'tm_tb']
CACHE = f'{OUT}/gate_feats_holdout_v80_new.pt'


def build_new_feats():
    train, tb, b2f, b2l = load_train_embs()
    sp = holdout_split(train, tb, b2f, b2l)
    files, classes, ci, kept = sp['val_files'], sp['classes'], sp['ci'], sp['kept']
    kept_idx = torch.tensor([ci[c] for c in kept], device=dev)
    other_idx = torch.tensor([i for i in range(len(classes)) if classes[i] not in set(kept)], device=dev)

    i336, f336, _ = train['fullft336shift']
    Q336 = torch.stack([f336[i336[fn]] for fn in files]).to(dev)
    print('scoring fullft336shift iNat bank ...', flush=True)
    raw = bank_scores(Q336, torch.load(BANK_336, weights_only=False)['bank'], other_idx.tolist())
    top2 = raw.topk(2, dim=1).values

    Ttb = F.normalize(torch.load(f'{OUT}/text_emb_taxabind_taxctx.pt',
                                 weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Qtb = torch.stack([tb[1][tb[0][fn]] for fn in files]).to(dev)
    tm_tb = (Qtb @ Ttb[kept_idx].t()).max(1).values - (Qtb @ Ttb[other_idx].t()).max(1).values

    return torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], tm_tb], dim=1).cpu()


def main():
    torch.set_num_threads(8)
    b = torch.load(f'{OUT}/gate_feats_holdout_v77.pt', weights_only=False)
    X12, y, cls = b['X'], b['y'], b['cls']
    if os.path.exists(CACHE):
        Xn = torch.load(CACHE, weights_only=False)['Xn']
        print(f'loaded cached new features {tuple(Xn.shape)}', flush=True)
    else:
        Xn = build_new_feats()
        torch.save({'Xn': Xn, 'feats': NEW}, CACHE)
        print(f'wrote {CACHE}', flush=True)

    X = torch.cat([X12, Xn], dim=1)
    ALL = FEATS + NEW
    n12 = len(FEATS)
    w = deploy_weights(y)
    Z = standardize(X, w)
    base = Z[:, ALL.index('sb_max')] + 2.0 * Z[:, ALL.index('tm_ctft')]

    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cls])

    i336 = [ALL.index('bank336_max'), ALL.index('bank336_margin')]
    itb = [ALL.index('tm_tb')]
    arms = {
        'v79 (12)': list(range(n12)),
        '+bank336 (14)': list(range(n12)) + i336,
        '+tm_tb (13)': list(range(n12)) + itb,
        '+both (15)': list(range(n12)) + i336 + itb,
    }
    oof = {k: torch.zeros(len(y)) for k in arms}
    for f in range(NFOLD):
        tr, va = folds != f, folds == f
        Ztr = standardize(X[tr], deploy_weights(y[tr]))
        Zva = standardize(X[va], deploy_weights(y[va]))
        for name, cols in arms.items():
            m = LogisticRegression(max_iter=20000, C=C)
            m.fit(Ztr[:, cols].numpy(), y[tr], sample_weight=deploy_weights(y[tr]).numpy())
            oof[name][torch.tensor(va)] = torch.tensor(
                m.predict_proba(Zva[:, cols].numpy())[:, 1], dtype=torch.float32)
        print(f'  fold {f}', flush=True)

    rows = {'v56_gate': dict(zip(['tpr', 'tnr', 'proj'], operating_point(base, y, w)),
                             auc=auc(base, y, w))}
    for name in arms:
        rows[name] = dict(zip(['tpr', 'tnr', 'proj'], operating_point(oof[name], y, w)),
                          auc=auc(oof[name], y, w))

    print(f'\n=== class-disjoint 5-fold CV, f=0.60 pinned, C={C:g} ===', flush=True)
    print(f'{"arm":16s} {"AUC":>8s} {"TPR":>8s} {"TNR":>8s} {"proj%":>8s} {"vs v79":>8s}', flush=True)
    p79 = rows['v79 (12)']['proj']
    for name, r in rows.items():
        print(f'{name:16s} {r["auc"]:8.4f} {100*r["tpr"]:8.2f} {100*r["tnr"]:8.2f} '
              f'{r["proj"]:8.3f} {r["proj"]-p79:+8.3f}', flush=True)

    best = max(arms, key=lambda k: rows[k]['proj'])
    gain = rows[best]['proj'] - p79
    print(f'\nbest = {best}  {gain:+.3f} proxy-pt vs v79 (kill bar {KILL:+.2f}) -> '
          f'{"CLEARS" if gain >= KILL and best != "v79 (12)" else "FAILS"}', flush=True)

    if gain >= KILL and best != 'v79 (12)':
        cols = arms[best]
        m = LogisticRegression(max_iter=20000, C=C)
        m.fit(Z[:, cols].numpy(), y, sample_weight=w.numpy())
        feats = [ALL[i] for i in cols]
        pickle.dump({'model': m, 'kind': f'logistic C={C:g}', 'feats': feats, 'cv': rows,
                     'delta': gain, 'clears_kill': True, 'seen_frac': 0.60, 'cols': cols},
                    open(f'{OUT}/learned_gate_v80.pkl', 'wb'))
        print(f'wrote {OUT}/learned_gate_v80.pkl  feats={feats}', flush=True)
        for n, c in sorted(zip(feats, m.coef_[0]), key=lambda t: -abs(t[1])):
            print(f'  {n:20s} {c:+.4f}{"  <- new" if n in NEW else ""}', flush=True)
    json.dump({'cv': rows, 'best': best, 'gain_vs_v79': gain}, open(f'{OUT}/learned_gate_v80.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
