"""logit_recheck: does tau*log(n_train) additive class-bias improve RECALL@10 (not just
top-1)? v99 found tau=+0.25 best for top-1 (+0.04pt, dead by top-1 kill bar) but never
checked recall@10 -- a bias could shuffle marginal candidates into/out of the top-10
even while barely moving argmax. Instrument: probe2_retriever_recall.py pattern (fused
member_scores, recall@k via topk vs gold). Base fusion = deployed 1.0/2.5/2.5 weighted
proto+2*cmax+4*taxon sum, identical to logit_adjust_v99 / rerank_seen_v83.

Gate: sweep tau in a few values around 0. If recall@10 > baseline (98.854) for any tau,
proceed to stage 2: retrain the rerank_seen_v83 leak-free re-ranker using candidates
from the tau-adjusted fusion, report true stack accuracy vs 91.674 (bar: >=92.674).
If no tau improves recall@10, stop here -- re-ranking a same-or-worse pool cannot help.

  conda activate onet && python research/logit_recheck_recall10.py
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
from seen_inat_v95 import member_scores

BASE_W = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
TAU_GRID = [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0]


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


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
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    n_train = torch.tensor([len(trby[c]) for c in seen], dtype=torch.float32)
    log_n = torch.log(n_train.clamp(min=1)).to(dev)
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zbase = {}
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
        del TF, TL, qF, base
        torch.cuda.empty_cache()

    fused = sum(BASE_W[t] * Zbase[t] for t in BASE_W)  # [N, S], deployed (tau=0)

    print('\n=== tau sweep: recall@k on fused + tau*log(n_train) ===', flush=True)
    rows = {}
    for tau in TAU_GRID:
        adj = fused + tau * log_n
        top10 = adj.topk(10, dim=1).indices.cpu()
        rec = {k: 100 * (top10[:, :k] == yv.unsqueeze(1)).any(1).float().mean().item()
               for k in (1, 5, 10)}
        rows[tau] = rec
        print(f'  tau={tau:+.2f} -> top1 {rec[1]:.3f}  recall@5 {rec[5]:.3f}  recall@10 {rec[10]:.3f}', flush=True)

    base_r10 = rows[0.0][10]
    best_tau = max(rows, key=lambda t: rows[t][10])
    best_r10 = rows[best_tau][10]
    lift = best_r10 - base_r10
    verdict = 'IMPROVES' if lift > 1e-9 and best_tau != 0.0 else 'DEAD'
    print(f'\nbaseline (tau=0) recall@10 = {base_r10:.3f}  [deployed real: 98.854]')
    print(f'best tau={best_tau:+.2f} recall@10 = {best_r10:.3f}  (lift {lift:+.3f})')
    print(f'VERDICT: {verdict} -- ' +
          ('proceed to stack retrain' if verdict == 'IMPROVES' else 'recall@10 does not improve, stop here (no re-rank follow-up)'),
          flush=True)
    json.dump({'rows': {str(k): v for k, v in rows.items()}, 'base_r10': base_r10,
               'best_tau': best_tau, 'best_r10': best_r10, 'lift': lift, 'verdict': verdict},
              open(f'{OUT}/logit_recheck_recall10.json', 'w'), indent=2)
    print(f'wrote {OUT}/logit_recheck_recall10.json', flush=True)


if __name__ == '__main__':
    main()
