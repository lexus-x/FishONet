"""v99: rank-based fusion (Borda count / reciprocal-rank fusion) across the 3 seen
legs (ctftshift/ftshift/fullft336shift), instead of the z-scored linear sum used by
the deployed fusion and by every v98 weight-grid variant.

Motivation: v98's 125-combo *linear-weight* grid is exhausted and its best point
(91.21% holdout) already real-tested WORSE (77.37% vs deployed's 77.5%). That failure
is evidence against this particular linear family, not against combining the 3 legs
at all. Rank fusion is a genuinely different combination RULE (order statistics, not
a magnitude-weighted sum) -- classic IR fact: Borda/RRF are more robust than a raw
score sum when per-leg score distributions have different tails/outliers, which is
exactly the situation here (proto+cmax+taxon scores are not calibrated to a common
scale across ctftshift/ftshift/fullft336shift beyond the z-score first moment).

Method: for each leg t and each holdout row, rank the S seen classes by that leg's
score (rank 0 = best). Combine ranks two ways:
  - Borda:  points_t = (S-1) - rank_t            (linear in rank)
  - RRF:    points_t = 1 / (K + rank_t + 1)       (K in {10, 60}, standard IR heuristic)
each summed either UNWEIGHTED (1/1/1, testing the rule itself) or with the EXISTING
deployed weights (1.0/2.5/2.5, testing whether the current weighting still helps in
rank-space) -- one point each, not a new grid search. argmax over the combined score
gives the predicted class; accuracy compared to the deployed linear-fusion accuracy
on the identical holdout split/rows.

Pre-registered kill bar: best rank-fusion variant must beat 91.21% (today's best
already-failed linear point) by >= +1.0pt, i.e. >= 92.21% holdout, to be worth a
submission slot -- a smaller number is not interesting given 91.21 already failed
real transfer.

  conda activate onet && python research/seen_rankfusion_v99.py
"""
from __future__ import annotations

import json
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, MEMBERS, dev, holdout_split, load_train_embs, zc
from seen_inat_v95 import member_scores

DEPLOYED = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
EQUAL = {t: 1.0 for t in DEPLOYED}
RRF_K = [10, 60]
BEST_TODAY = 91.21          # v98's best linear reweight, already real-tested WORSE
KILL_LIFT = 1.0


def rank_of(scoremat):
    """rank[i,j] = 0 for the best class in row i, S-1 for the worst. Order-preserving
    for any monotonic per-row rescaling, so it's fine to rank the z-scored Zbase
    tensors directly (z-scoring does not change within-row order)."""
    return scoremat.argsort(dim=1, descending=True).argsort(dim=1).float()


def main():
    torch.set_num_threads(8)
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby, val_seen = sp['trby'], [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    ci = {c: i for i, c in enumerate(classes)}
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zbase = {}
    for t, _, _ in MEMBERS:
        if t not in DEPLOYED:
            continue
        idx, feats, _ = train[t]
        P = torch.zeros(S, feats.shape[1])
        cnt = torch.zeros(S)
        TFl, TLl = [], []
        for c in seen:
            for fn in trby[c]:
                if fn not in idx:
                    continue
                f = feats[idx[fn]]
                P[s2i[c]] += f
                cnt[s2i[c]] += 1
                TFl.append(f)
                TLl.append(s2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
        TF = torch.stack(TFl).to(dev)
        TL = torch.tensor(TLl).to(dev)
        qF = torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev)
        base = member_scores(qF, P, TF, TL, Tseen, S)
        Zbase[t] = zc(base)
        acc = 100 * (base.argmax(1).cpu() == yv).float().mean().item()
        print(f'  {t:16s} solo val acc {acc:6.2f}', flush=True)
        del TF, TL, qF, base
        torch.cuda.empty_cache()

    def lin_acc(w):
        f = sum(w[t] * Zbase[t] for t in DEPLOYED)
        return 100 * (f.argmax(1).cpu() == yv).float().mean().item()

    def borda_acc(w):
        pts = sum(w[t] * ((S - 1) - rank_of(Zbase[t])) for t in DEPLOYED)
        return 100 * (pts.argmax(1).cpu() == yv).float().mean().item()

    def rrf_acc(w, k):
        pts = sum(w[t] / (k + rank_of(Zbase[t]) + 1.0) for t in DEPLOYED)
        return 100 * (pts.argmax(1).cpu() == yv).float().mean().item()

    rows = {}
    rows['deployed_linear_1.0_2.5_2.5'] = lin_acc(DEPLOYED)
    rows['borda_equal'] = borda_acc(EQUAL)
    rows['borda_deployed_w'] = borda_acc(DEPLOYED)
    for k in RRF_K:
        rows[f'rrf_equal_k{k}'] = rrf_acc(EQUAL, k)
        rows[f'rrf_deployed_w_k{k}'] = rrf_acc(DEPLOYED, k)

    print('\n=== rank-fusion vs deployed linear fusion (holdout closed-set accuracy) ===', flush=True)
    for name, acc in rows.items():
        print(f'  {name:28s} {acc:6.2f}', flush=True)

    best_key = max(rows, key=lambda k: rows[k] if k != 'deployed_linear_1.0_2.5_2.5' else -1)
    best_acc = rows[best_key]
    lift = best_acc - BEST_TODAY
    verdict = (f'CLEARS — {best_key} {best_acc:.2f} beats {BEST_TODAY} by {lift:+.2f}pt'
               if lift >= KILL_LIFT else
               f'DEAD — best rank-fusion variant ({best_key} {best_acc:.2f}) does not beat '
               f'{BEST_TODAY} by >= +{KILL_LIFT}pt')
    print(f'\nbest-today (v98 linear, already failed real) {BEST_TODAY:.2f}  '
          f'best rank-fusion {best_key} {best_acc:.2f}  lift {lift:+.2f}pt '
          f'(kill >= +{KILL_LIFT})', flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'rows': rows, 'best_today': BEST_TODAY, 'best': best_key, 'best_acc': best_acc,
               'lift': lift, 'verdict': verdict},
              open(f'{OUT}/seen_rankfusion_v99.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_rankfusion_v99.json', flush=True)


if __name__ == '__main__':
    main()
