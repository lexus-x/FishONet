"""v108: teammate proposal -- on the embedding-mean base (v107), replace the fixed
proto/cmax/taxon weights (1.0/2.0/4.0) with:

    final = (score_1 + score_2) + (0.9 * score_3 + 0.1 * score_4)

  score_1 = proto (class-mean cosine sim)
  score_2 = cmax  (nearest single training-image cosine sim)
  score_3 = taxon (query vs scientific-name text embedding)   -- was weight 4.0, now ~0.9
  score_4 = query vs PROVIDED class description text embedding (outputs/text_emb_h_traits_provided.pt
            -- the cached embedding of data/dl/descriptions.json's morphological descriptions,
            LLM-compressed per research/unseen_text.py; no raw-description embedding was cached
            separately, so this is the closest available "text descriptions" resource)

All four scores computed on the SAME merged query representation (test_embedding_mean =
L2norm(mean(L2norm(leg) for leg in [ctftshift, ftshift, fullft336shift])), per v107) --
this is a follow-up on the embedding-mean idea, not the original per-leg fusion.

  conda activate onet && python research/seen_embmean_desc_v108.py
"""
from __future__ import annotations

import json
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, MEMBERS, dev, holdout_split, load_train_embs

torch.set_num_threads(8)
LEGS = ['ctftshift', 'ftshift', 'fullft336shift']
DEPLOYED_ACC = 89.870
DEPLOYED_RECALL10 = 98.854
EMBMEAN_ACC = 90.199
EMBMEAN_RECALL10 = 98.972
RERANK_BAR = 91.674
KILL = 1.0


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

    # taxon text anchor (existing)
    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen_taxon = TtH[kept_idx]

    # FishBase multi-bullet description text anchor -- score_4 (per teammate: multiple
    # descriptions per class -> L2 each, average, L2 again; already done in
    # src/build_text_fishbase.py, this just indexes the result)
    dtr = torch.load(f'{OUT}/text_emb_h_fishbase.pt', weights_only=False)
    tr_classes = dtr['classes']
    tci = {c: i for i, c in enumerate(tr_classes)}
    cov_mask = dtr['coverage_mask']
    missing = [c for c in seen if c not in tci or not cov_mask[tci[c]]]
    print(f'fishbase coverage: {len(seen)-len(missing)}/{len(seen)} seen classes '
          f'({len(missing)} missing -> zero-vector fallback)', flush=True)
    Ttr_full = dtr['emb_fishbase'].float()  # already L2/avg/L2'd per class
    Tseen_traits = torch.stack([
        Ttr_full[tci[c]] if (c in tci and cov_mask[tci[c]]) else torch.zeros(Ttr_full.shape[1])
        for c in seen
    ]).to(dev)

    # per-leg raw embeddings -> merged (embedding-mean) representation, same as v107
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
    TL = per_leg_train[LEGS[0]][1]

    TF_mean = F.normalize(sum(F.normalize(per_leg_train[t][0], dim=-1) for t in LEGS) / 3.0, dim=-1)
    qF_mean = F.normalize(sum(F.normalize(per_leg_val[t], dim=-1) for t in LEGS) / 3.0, dim=-1)

    P = torch.zeros(S, TF_mean.shape[1], device=dev)
    cnt = torch.zeros(S, device=dev)
    P.index_add_(0, TL, TF_mean)
    cnt.index_add_(0, TL, torch.ones_like(TL, dtype=torch.float32))
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)

    def scores(qF):
        n = qF.shape[0]
        s1 = torch.empty(n, S, device=dev)   # proto
        s2 = torch.empty(n, S, device=dev)   # cmax
        for i in range(0, n, 2000):
            e = qF[i:i + 2000]
            s1[i:i + 2000] = e @ P.t()
            sim = e @ TF_mean.t()
            c = torch.full((e.shape[0], S), -1e9, device=dev)
            c.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
            s2[i:i + 2000] = c
        s3 = qF @ Tseen_taxon.t()   # taxon
        s4 = qF @ Tseen_traits.t()  # description/traits
        return s1, s2, s3, s4

    s1, s2, s3, s4 = scores(qF_mean)
    final = (s1 + s2) + (0.9 * s3 + 0.1 * s4)
    acc = 100 * (final.argmax(1).cpu() == yv).float().mean().item()
    top10 = final.topk(10, dim=1).indices.cpu()
    recall10 = 100 * (top10 == yv.unsqueeze(1)).any(1).float().mean().item()

    # ablation: same formula but WITHOUT score_4 (0.9*s3 alone, i.e. does adding traits help
    # at all, isolated from the proto/cmax weight change embmean already made)
    final_no4 = (s1 + s2) + (0.9 * s3)
    acc_no4 = 100 * (final_no4.argmax(1).cpu() == yv).float().mean().item()
    top10_no4 = final_no4.topk(10, dim=1).indices.cpu()
    recall10_no4 = 100 * (top10_no4 == yv.unsqueeze(1)).any(1).float().mean().item()

    print(f'\nformula: (proto+cmax) + (0.9*taxon + 0.1*traits)   acc {acc:.3f}  recall@10 {recall10:.3f}')
    print(f'ablation, no score_4:  (proto+cmax) + (0.9*taxon)     acc {acc_no4:.3f}  recall@10 {recall10_no4:.3f}')
    print(f'\nrefs: deployed {DEPLOYED_ACC}/{DEPLOYED_RECALL10}   embmean-v107 {EMBMEAN_ACC}/{EMBMEAN_RECALL10}')

    for name, a, r in [('full (score1-4)', acc, recall10), ('no-score4', acc_no4, recall10_no4)]:
        print(f'  {name:16s} lift-vs-deployed acc {a-DEPLOYED_ACC:+.3f} recall@10 {r-DEPLOYED_RECALL10:+.3f}'
              f'  | lift-vs-embmean acc {a-EMBMEAN_ACC:+.3f} recall@10 {r-EMBMEAN_RECALL10:+.3f}')

    verdict = 'CLEARS recall@10 -- worth chaining into reranker' if max(recall10, recall10_no4) > DEPLOYED_RECALL10 else 'DEAD'
    print(f'\nVERDICT: {verdict}', flush=True)
    json.dump({'acc_full': acc, 'recall10_full': recall10, 'acc_no4': acc_no4,
               'recall10_no4': recall10_no4, 'deployed_acc': DEPLOYED_ACC,
               'deployed_recall10': DEPLOYED_RECALL10, 'verdict': verdict},
              open(f'{OUT}/seen_embmean_desc_v108.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_embmean_desc_v108.json', flush=True)


if __name__ == '__main__':
    main()
