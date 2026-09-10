"""v100: does multi-view TTA sharpen the fullft336shift leg's seen-vs-pseudo-novel signal?

SCOPE CORRECTION vs the assignment's SEEN-closed-set framing -- read this before running:
outputs/emb_holdout_fullft336_cropviews.pt covers ONLY the 2,318 pseudo-novel (val_uns) rows.
Verified by set intersection against learned_gate_v77.holdout_split(): 0/2318 overlap with the
11,866-row val_seen split that research/seen_reweight_v98.py's 89.87/91.20/91.21 numbers are
measured on; 2318/2318 overlap with val_uns. Pseudo-novel classes get ZERO training images
(holdout_split routes 100% of a pseudo class's images to val_uns, none to trby) -- there is no
seen-route prototype for their true label, so closed-set "accuracy" on this file is undefined.
THIS FILE CANNOT MOVE THE 91.21 / 92.21 SEEN-closed-set NUMBER. (research/rerank_seen_v83.py is
seen-only for the same reason: candidate pool = kept classes, val_uns rows never enter it either.)

What it CAN test, flagged as unexploited in learned_gate_v77.py's own docstring: the novelty
gate's per-member "how seen-like is this row" feature (m_fullft336shift = seen_score(...).max(1))
is computed from a single center-view embedding for every val row, including the 2,318
pseudo-novel ones -- "the 7-view crop-max stays in the unseen HEAD ... and never enters the
gate." This script recomputes that one feature 3 ways for the pseudo-novel rows only (center /
mean-pooled-7-view / max-pooled-7-view) and measures ROC-AUC separating val_seen (label 1,
center-view, unchanged) from val_uns (label 0, swapped) -- i.e. does TTA make true novel rows
look LESS like some known class, which is exactly the gate's job at f=0.60.

Pass bar (gate-AUC space, NOT the 92.21 SEEN closed-set bar -- genuinely different metric, see
above): best variant clears center-view baseline by >= +0.02 AUC. Below that, kill it under the
same logic v93/v96 used for weak single-feature effects -- not worth a fused re-fit.
If it clears: next (separate, heavier) step is swapping sb_max/sb_margin/sb_lse_gap/
m_fullft336shift into learned_gate_v77.build_holdout()'s val_uns rows, retraining the 12-feature
logistic gate, and re-projecting through COEF_TPR/COEF_TNR at f=0.60 for a real overall-pt
number -- gated on this passing first, and even then it is a gate-quality lever (moves TNR at
the novelty gate), not a fix to the SEEN fusion research/seen_reweight_v98.py measures.

  conda activate onet && python research/tta_fullft336_gate_auc_v100.py
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, 'research')
from learned_gate_v77 import OUT, dev, holdout_split, load_train_embs, seen_score

CROPVIEWS = f'{OUT}/emb_holdout_fullft336_cropviews.pt'
AUC_KILL = 0.02


def main():
    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby, kept = sp['trby'], sorted(sp['kept'])
    k2i = {c: i for i, c in enumerate(kept)}
    S = len(kept)
    val_seen = [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 1]
    val_uns = [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 0]
    print(f'val_seen {len(val_seen)} | val_uns {len(val_uns)} | kept classes {S}', flush=True)

    idx, feats, _ = train['fullft336shift']

    # kept-class prototypes, single-view -- identical construction to seen_reweight_v98 / build_holdout
    P = torch.zeros(S, feats.shape[1])
    cnt = torch.zeros(S)
    TFl, TLl = [], []
    for c in kept:
        for fn in trby[c]:
            f = feats[idx[fn]]
            P[k2i[c]] += f
            cnt[k2i[c]] += 1
            TFl.append(f)
            TLl.append(k2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
    TF = torch.stack(TFl).to(dev)
    TL = torch.tensor(TLl).to(dev)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(),
                       dim=-1).to(dev)
    kept_idx = torch.tensor([sp['ci'][c] for c in kept]).to(dev)
    Tseen = TtH[kept_idx]

    def score_max(qF):
        return seen_score(qF, P, TF, TL, True, Tseen, S).max(1).values.cpu()

    # label-1 half: val_seen, single-view, unchanged -- matches production exactly
    qF_seen = torch.stack([feats[idx[fn]] for fn in val_seen])
    s_seen = score_max(qF_seen)

    # label-0 half: val_uns, 3 variants from the cached 7-view crop TTA
    cv = torch.load(CROPVIEWS, weights_only=False)
    cv_idx = {fn: i for i, fn in enumerate(cv['files'])}
    missing = [fn for fn in val_uns if fn not in cv_idx]
    assert not missing, f'{len(missing)} val_uns rows missing from {CROPVIEWS}'
    order = [cv_idx[fn] for fn in val_uns]
    views = {k: v[order] for k, v in cv['views'].items()}

    # (a) center-view baseline: should reproduce what the deployed gate feature currently computes
    s_uns_center = score_max(views['center'])

    # (b) mean-pooled embedding across all 7 views, renormalized
    mean_emb = F.normalize(sum(views.values()) / len(views), dim=-1)
    s_uns_mean = score_max(mean_emb)

    # (c) per-view score-then-max: true TTA-max, matches the v56 "crop-max" recipe used for the
    # unseen head at eval time (never validated against ground truth before -- this file lets us)
    per_view_scores = [seen_score(v, P, TF, TL, True, Tseen, S) for v in views.values()]
    s_uns_max = torch.stack(per_view_scores).max(0).values.max(1).values.cpu()

    y = torch.cat([torch.ones(len(val_seen)), torch.zeros(len(val_uns))]).numpy()

    def auc(s_uns):
        return roc_auc_score(y, torch.cat([s_seen, s_uns]).numpy())

    a_center, a_mean, a_max = auc(s_uns_center), auc(s_uns_mean), auc(s_uns_max)
    print(f'AUC center-view (repro) : {a_center:.4f}')
    print(f'AUC mean-pooled 7-view  : {a_mean:.4f}  (delta {a_mean - a_center:+.4f})')
    print(f'AUC max-pooled  7-view  : {a_max:.4f}  (delta {a_max - a_center:+.4f})')
    best_delta = max(a_mean, a_max) - a_center
    verdict = ('CLEARS -> worth the fused re-fit + f=0.60 re-projection' if best_delta >= AUC_KILL
               else 'DEAD -- TTA does not sharpen this gate feature enough to matter')
    print(f'\nVERDICT: {verdict}  (best delta {best_delta:+.4f}, kill bar +{AUC_KILL})', flush=True)


if __name__ == '__main__':
    main()
