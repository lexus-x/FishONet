"""v107: replace the deployed weighted SCORE fusion (1.0/2.5/2.5 on proto+2*cmax+4*taxon
per leg) with a uniform EMBEDDING-level mean: for each image, L2-normalize each leg's
embedding, average with equal weight (1/3 each), L2-normalize the result, and run ONE
proto+cmax+taxon classification on that single merged embedding -- no per-leg weights at all.

  test_embedding_mean = L2_norm( 1/3 * sum_i L2_norm(embedding_i) )

Tested for both bare fused-argmax accuracy AND recall@10 (the fair retriever-level bar,
per today's audit -- a retriever change must clear recall@10 before a re-ranker retrain
is worth it).

  conda activate onet && python research/seen_embmean_v107.py
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

torch.set_num_threads(8)
LEGS = ['ctftshift', 'ftshift', 'fullft336shift']
BASE_W = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
LAM = 4.0
DEPLOYED_ACC = 89.870
DEPLOYED_RECALL10 = 98.854
RERANK_BAR = 91.674
KILL = 1.0


def member_scores(qF, P, TF, TL, Tseen, S):
    out = torch.empty(qF.shape[0], S, device=dev)
    for i in range(0, qF.shape[0], 2000):
        e = qF[i:i + 2000]
        sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        out[i:i + 2000] = e @ P.t() + 2.0 * cmax + LAM * (e @ Tseen.t())
    return out


def main():
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby = sp['trby']
    val_seen = [f for f, y_ in zip(sp['val_files'], sp['y']) if y_ == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    # per-leg raw embeddings for train exemplars and val queries, in a common file order
    per_leg_train, per_leg_val = {}, {}
    for t in LEGS:
        idx, feats, _ = train[t]
        TFl, TLl = [], []
        for c in seen:
            for fn in trby[c]:
                if fn not in idx:
                    continue
                TFl.append(feats[idx[fn]])
                TLl.append(s2i[c])
        per_leg_train[t] = (torch.stack(TFl).to(dev), torch.tensor(TLl).to(dev))
        per_leg_val[t] = torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev)
    TL = per_leg_train[LEGS[0]][1]  # identical ordering/labels across legs by construction

    # ---- baseline: deployed weighted SCORE fusion (sanity check vs known 89.870/98.854) ----
    Zbase = {}
    for t in LEGS:
        TF, _ = per_leg_train[t]
        qF = per_leg_val[t]
        P = torch.zeros(S, TF.shape[1], device=dev)
        cnt = torch.zeros(S, device=dev)
        P.index_add_(0, TL, TF)
        cnt.index_add_(0, TL, torch.ones_like(TL, dtype=torch.float32))
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
        Zbase[t] = zc(member_scores(qF, P, TF, TL, Tseen, S))
    fused = sum(BASE_W[t] * Zbase[t] for t in LEGS)
    base_acc = 100 * (fused.argmax(1).cpu() == yv).float().mean().item()
    top10 = fused.topk(10, dim=1).indices.cpu()
    base_recall10 = 100 * (top10 == yv.unsqueeze(1)).any(1).float().mean().item()
    print(f'deployed weighted-score fusion: acc {base_acc:.3f} (ref {DEPLOYED_ACC}) '
          f'recall@10 {base_recall10:.3f} (ref {DEPLOYED_RECALL10})', flush=True)

    # ---- embedding-mean fusion: L2norm each leg, average equal-weight, L2norm again ----
    TF_mean = F.normalize(sum(F.normalize(per_leg_train[t][0], dim=-1) for t in LEGS) / 3.0, dim=-1)
    qF_mean = F.normalize(sum(F.normalize(per_leg_val[t], dim=-1) for t in LEGS) / 3.0, dim=-1)
    P_mean = torch.zeros(S, TF_mean.shape[1], device=dev)
    cnt = torch.zeros(S, device=dev)
    P_mean.index_add_(0, TL, TF_mean)
    cnt.index_add_(0, TL, torch.ones_like(TL, dtype=torch.float32))
    P_mean = F.normalize(P_mean / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    scores_mean = member_scores(qF_mean, P_mean, TF_mean, TL, Tseen, S)
    mean_acc = 100 * (scores_mean.argmax(1).cpu() == yv).float().mean().item()
    top10_mean = scores_mean.topk(10, dim=1).indices.cpu()
    mean_recall10 = 100 * (top10_mean == yv.unsqueeze(1)).any(1).float().mean().item()
    print(f'\nembedding-mean fusion (uniform, no leg weights): acc {mean_acc:.3f}  '
          f'recall@10 {mean_recall10:.3f}', flush=True)

    lift_acc = mean_acc - base_acc
    lift_recall = mean_recall10 - base_recall10
    print(f'\nlift vs deployed: acc {lift_acc:+.3f}pt  recall@10 {lift_recall:+.3f}pt', flush=True)
    verdict = ('CLEARS recall@10 -- worth chaining into a re-ranker retrain'
               if lift_recall > 0 else
               'DEAD -- recall@10 does not improve, cannot beat the 91.674 stack via re-ranking')
    print(f'VERDICT: {verdict}', flush=True)

    out = {'deployed_acc': base_acc, 'deployed_recall10': base_recall10,
           'embmean_acc': mean_acc, 'embmean_recall10': mean_recall10,
           'lift_acc': lift_acc, 'lift_recall10': lift_recall, 'verdict': verdict}
    json.dump(out, open(f'{OUT}/seen_embmean_v107.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/seen_embmean_v107.json', flush=True)


if __name__ == '__main__':
    main()
