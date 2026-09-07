"""WiSE-style blend: ctftshift vs ctftshift_inat embeddings (alpha sweep on stack proxy).

  python research/ctft_wise_blend_proxy.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT  # noqa: E402
from inat_ft_stack_proxy import queries, text_stack, mean_proto  # noqa: E402
from inat_maxpool_proxy import score_maxpool, top1  # noqa: E402

PROTO_BASE = os.path.join(OUT, 'inat_protos_ctftshift_full.pt')
PROTO_INAT = os.path.join(OUT, 'inat_protos_ctftshift_inat_full.pt')
BANK_BASE = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')
REF_MEAN, REF_MP35, REF_MP4 = 43.49, 45.17, 45.08


def blend_feats(f0, f1, alpha: float) -> torch.Tensor:
    if alpha <= 0:
        return f0
    if alpha >= 1:
        return f1
    return F.normalize((1 - alpha) * f0 + alpha * f1, dim=-1)


def blend_proto_path(alpha: float):
    d0 = torch.load(PROTO_BASE, weights_only=False)
    d1 = torch.load(PROTO_INAT, weights_only=False)
    P0 = d0['protos'].float()
    P1 = d1['protos'].float()
    P = F.normalize((1 - alpha) * P0 + alpha * P1, dim=-1)
    return P


def mean_from_P(cand, q, P):
    Pc = P[cand]
    has = Pc.norm(dim=-1) > 0.5
    S = q @ Pc.t()
    S = S.masked_fill(~has.unsqueeze(0), -1e4)
    return dbnorm(S), has


def cv_delta(S0, gold, has, D, Sm, w):
    rng = random.Random(0)
    cov_classes = [c for c in D.pseudo if has[D.cand_pos[D.ci[c]]]]
    rng.shuffle(cov_classes)
    half = len(cov_classes) // 2
    val_cls = set(cov_classes[half:])
    cand_list = D.cand.tolist()
    q_cls = [D.classes[cand_list[g]] for g in gold.tolist()]
    val_m = torch.tensor([c in val_cls for c in q_cls])
    val_base = top1((S0 + 3.0 * Sm)[val_m], gold[val_m]) if w == 3 else top1(S0[val_m], gold[val_m])
    val_t = top1((S0 + w * Sm)[val_m], gold[val_m])
    return val_t - val_base


def main():
    t0 = time.time()
    D = FishData()
    cand = D.cand
    bank = torch.load(BANK_BASE, weights_only=False)['bank']

    idx0, f0, files0 = load_emb(os.path.join(OUT, 'emb_train_ctftshift.pt'))
    idx1, f1, files1 = load_emb(os.path.join(OUT, 'emb_train_ctftshift_inat.pt'))
    assert files0 == files1, 'train file order mismatch'

    q0, gold = queries(D, idx0, f0)
    q1, gold1 = queries(D, idx1, f1)
    assert gold.equal(gold1)

    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]
    rows = []
    best = {'alpha': 0.0, 'mean_w3': 0.0, 'maxpool_w35': 0.0, 'maxpool_w4': 0.0}

    for alpha in alphas:
        q = blend_feats(q0, q1, alpha)
        S0, _ = text_stack(D, cand, q)
        base = top1(S0, gold)
        P = blend_proto_path(alpha)
        Smean, has = mean_from_P(cand, q, P)
        mean_w3 = top1(S0 + 3.0 * Smean, gold)
        Smp4 = score_maxpool(q, bank, cand, topm=4)
        mp35 = top1(S0 + 3.5 * Smp4, gold)
        mp4 = top1(S0 + 4.0 * Smp4, gold)
        row = {
            'alpha': alpha,
            'baseline_text': base,
            'mean_w3': mean_w3,
            'delta_mean_w3': mean_w3 - REF_MEAN,
            'maxpool_w35': mp35,
            'delta_maxpool_w35': mp35 - REF_MP35,
            'maxpool_w4': mp4,
            'delta_maxpool_w4': mp4 - REF_MP4,
        }
        rows.append(row)
        print(
            f'a={alpha:.2f} text={base:.2f} mean@3={mean_w3:.2f} ({row["delta_mean_w3"]:+.2f}) '
            f'mp@3.5={mp35:.2f} ({row["delta_maxpool_w35"]:+.2f}) mp@4={mp4:.2f}',
            flush=True,
        )
        if mean_w3 > best['mean_w3']:
            best.update({'alpha': alpha, 'mean_w3': mean_w3, 'maxpool_w35': mp35, 'maxpool_w4': mp4})

    # CV at best alpha for mean and maxpool
    ba = best['alpha']
    q = blend_feats(q0, q1, ba)
    S0, _ = text_stack(D, cand, q)
    P = blend_proto_path(ba)
    Smean, has = mean_from_P(cand, q, P)
    Smp4 = score_maxpool(q, bank, cand, topm=4)

    out = {
        'method': 'embedding_blend_ctftshift + alpha*inat_delta',
        'bank': BANK_BASE,
        'ref': {'mean_w3': REF_MEAN, 'maxpool_w35': REF_MP35, 'maxpool_w4': REF_MP4},
        'alphas': rows,
        'best_by_mean_w3': best,
        'cv_at_best_alpha': {
            'alpha': ba,
            'mean_val_delta': cv_delta(S0, gold, has, D, Smean, 3.0),
            'mp_val_delta': cv_delta(S0, gold, has, D, Smp4, 3.5),
        },
        'projected_real_002_per_proxy_pt': {
            'from_best_mean_delta': 50.49 + 0.02 * (best['mean_w3'] - REF_MEAN),
            'from_best_mp_delta': 50.49 + 0.02 * (best['maxpool_w35'] - REF_MP35),
        },
        'clears_kill_03': (best['mean_w3'] - REF_MEAN) >= 0.3 or (best['maxpool_w35'] - REF_MP35) >= 0.3,
    }
    path = os.path.join(OUT, 'ctft_wise_blend_proxy_results.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'[{time.time()-t0:.0f}s] best alpha={ba} mean={best["mean_w3"]:.2f} wrote {path}', flush=True)


if __name__ == '__main__':
    main()
