"""v106: unsupervised k-reciprocal / Jaccard neighbor-consistency re-ranking over the
CLASS-PROTOTYPE graph, for the SEEN closed-set classifier.

Genuinely new axis vs everything else in flight today (js_shrink/shrink = move a
PROTOTYPE's location toward a taxon anchor; selfcal = rescale one class's own score by
its own LOO self-similarity; whiten = one global linear transform of embedding space;
bagging = variance reduction of the SAME mean-prototype estimator; softknn = a 3rd
aggregation strategy between proto-mean and cmax; moe_fusion = per-QUERY leg reweight;
leg_agreement = discrete cross-LEG vote count; concat_fusion = representation-level
fusion; harmonic_fusion = non-additive combination of the 3 LEGS). None of them look at
the relationship BETWEEN classes. This does: a static, unsupervised class-class
similarity graph (built once from ftshift's training prototypes, no learning, no
external data) is used to ask, for each of a query's own top-K candidates, "does this
candidate's neighborhood in class-space overlap with the query's OWN candidate set?" --
the k-reciprocal / Jaccard re-ranking principle from person re-ID (Zhong et al. 2017),
applied here at the class level instead of the gallery-image level.

MECHANISM:
  1. Base score = deployed 3-leg fusion (proto + 2*cmax + 4*taxon per member, zc'd,
     summed with weights 1.0/2.5/2.5) -- identical formula to seen_inat_v95/seen_reweight_v98.
  2. Class graph G (S x S): cosine sim between ftshift class prototypes (train-fold
     only, same leak-free trby split). NB[c] = top-Kg nearest OTHER classes to c.
  3. Per query: topK = query's own top-K candidates by base score. For each candidate
     c in topK, jaccard(c) = |NB[c] ∩ topK| / |NB[c] ∪ topK| (approx union = Kg+K-|∩|).
  4. final = base_score; final[q, topK] += ALPHA * jaccard  -->  argmax.
  ALPHA=0 must reproduce the base fusion argmax exactly (sanity check).

Pre-registered grid (no number seen before this line): K in {10, 20}, Kg in {5, 15},
ALPHA in {0, 0.5, 1, 2, 4, 8}. Kill bar: best combo lifts val_seen closed-set argmax
accuracy >= +1.0pt over the repo-best 91.674% (v83 seen re-ranker), i.e. >= 92.674%.

  conda activate onet && python research/seen_kreciprocal_v106.py
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, MEMBERS, dev, holdout_split, load_train_embs, zc
from seen_inat_v95 import BASE_W, member_scores

KS = [10, 20]
KGS = [5, 15]
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
KILL_LIFT_OVER_V83 = 1.0
V83_RERANK = 91.674


def main():
    torch.set_num_threads(8)
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby, val_seen = sp['trby'], [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen]).to(dev)
    Q = len(val_seen)
    print(f'val_seen rows {Q} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zbase = {}
    P_ftshift = None
    for t, _, _ in MEMBERS:
        if t not in BASE_W:
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
        acc = 100 * (base.argmax(1) == yv).float().mean().item()
        print(f'  {t:16s} solo val acc {acc:6.2f}', flush=True)
        if t == 'ftshift':
            P_ftshift = P
        del TF, TL, qF, base
        torch.cuda.empty_cache()

    fused = sum(BASE_W[t] * Zbase[t] for t in BASE_W)  # (Q, S)
    base_acc = 100 * (fused.argmax(1) == yv).float().mean().item()
    print(f'\ndeployed base fusion val acc: {base_acc:.2f} (sanity vs known ~89.87)', flush=True)

    # class-class graph from ftshift prototypes (train-fold only, no query/val info)
    G = P_ftshift @ P_ftshift.t()  # (S, S)
    G.fill_diagonal_(-1e9)

    print('\n=== pre-registered grid (val_seen closed-set accuracy) ===', flush=True)
    rows = {}
    for Kg in KGS:
        NB = G.topk(Kg, dim=1).indices  # (S, Kg)
        for K in KS:
            topK_vals, topK_idx = fused.topk(K, dim=1)  # (Q, K)
            isTopK = torch.zeros(Q, S, dtype=torch.bool, device=dev)
            isTopK.scatter_(1, topK_idx, True)

            nb_of_cands = NB[topK_idx]  # (Q, K, Kg) -- neighbor sets of each query's own top-K candidates
            row_idx = torch.arange(Q, device=dev).view(Q, 1, 1).expand(Q, K, Kg)
            gathered = isTopK[row_idx, nb_of_cands]  # (Q, K, Kg) bool: is this neighbor also in my topK?
            inter = gathered.sum(-1).float()  # (Q, K)
            union = (Kg + K - inter).clamp(min=1.0)
            jaccard = inter / union  # (Q, K)

            for ALPHA in ALPHAS:
                final = fused.clone()
                final.scatter_add_(1, topK_idx, ALPHA * jaccard)
                acc = 100 * (final.argmax(1) == yv).float().mean().item()
                key = f'K={K} Kg={Kg} alpha={ALPHA}'
                rows[key] = acc
                if ALPHA == 0.0:
                    assert abs(acc - base_acc) < 1e-4, f'ALPHA=0 sanity check failed: {acc} vs {base_acc}'
            del topK_vals, topK_idx, isTopK, nb_of_cands, row_idx, gathered, inter, union, jaccard
            torch.cuda.empty_cache()

    best_key = max(rows, key=rows.get)
    best_acc = rows[best_key]
    lift = best_acc - base_acc
    lift_over_v83 = best_acc - V83_RERANK
    top8 = sorted(rows.items(), key=lambda kv: -kv[1])[:8]
    for k, v in top8:
        print(f'  {k:28s} -> {v:6.2f}', flush=True)

    verdict = ('CLEARS' if lift_over_v83 >= KILL_LIFT_OVER_V83 else
               'DEAD -- no (K, Kg, alpha) reaches v83+1.0')
    print(f'\nbase_fusion {base_acc:.2f}  best {best_key} {best_acc:.2f}  '
          f'(lift over base {lift:+.2f}, lift over v83-rerank {lift_over_v83:+.2f}, '
          f'kill bar >= +{KILL_LIFT_OVER_V83} over {V83_RERANK})', flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'rows': rows, 'base_acc': base_acc, 'best_key': best_key, 'best_acc': best_acc,
                'lift_over_base': lift, 'lift_over_v83': lift_over_v83, 'verdict': verdict},
               open(f'{OUT}/seen_kreciprocal_v106.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_kreciprocal_v106.json', flush=True)


if __name__ == '__main__':
    main()
