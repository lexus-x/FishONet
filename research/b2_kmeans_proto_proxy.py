"""k-means multi-prototype on the frozen BioCLIP-2 iNat photo bank (untested).

Reference: v41 frozen B2 mean-proto @ w=2.5 on the v41 stack (ref 46.29, dual_b2_proxy).
This isolates whether k-means multi-prototype (query = max over K cluster centers) beats
the single mean proto — a distinct construction from B2-maxpool (dead) and ToL-bank (dead).

  /home/ubuntu/miniconda3/envs/onet/bin/python research/b2_kmeans_proto_proxy.py
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT  # noqa: E402
from inat_maxpool_proxy import score_maxpool, top1  # noqa: E402
from v41_taxctx_weight_cv import stack_base  # noqa: E402

REF_FROZEN25 = 46.28990590572357  # dual_b2_proxy frozen@2.5
IMG_W = 4.0
B2_W = 2.5
BANK_CTFT = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')
BANK_B2 = os.path.join(OUT, 'inat_photo_bank_bioclip2.pt')
MEAN_PROTO = os.path.join(OUT, 'inat_protos_bioclip2.pt')
QENC_B2 = os.path.join(OUT, 'emb_train_bioclip2.pt')


def queries(D, idx, feats):
    q, y = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx:
                q.append(feats[idx[fn]])
                y.append(D.cand_pos[D.ci[c]])
    return torch.stack(q), torch.tensor(y)


def kmeans_centers(photos, K, iters=20, seed=0):
    """photos [n,D] normalized. Returns [K,D] normalized cluster centers."""
    n = photos.shape[0]
    if n <= K:
        return photos
    torch.manual_seed(seed)
    idx = torch.randperm(n)[:K]
    centers = photos[idx].clone()
    for _ in range(iters):
        sim = photos @ centers.t()  # [n,K]
        assign = sim.argmax(1)
        newc = []
        for k in range(K):
            m = assign == k
            if m.sum() == 0:
                newc.append(centers[k])
            else:
                newc.append(photos[m].mean(0))
        centers = torch.stack([F.normalize(c, dim=-1) for c in newc])
    return centers


def multi_proto_leg(Q, bank, cand, K):
    """Score Q with max-cosine over K cluster centers per class. Returns dbnorm(S), has."""
    N, C = Q.shape[0], len(cand)
    S = torch.full((N, C), -1e4)
    has = torch.zeros(C, dtype=torch.bool)
    cand_list = cand.tolist()
    for j, gidx in enumerate(cand_list):
        photos = bank.get(gidx)
        if photos is None or photos.numel() == 0:
            continue
        photos = F.normalize(photos.float(), dim=-1)
        centers = kmeans_centers(photos, K)
        sim = Q @ centers.t()  # [N,K]
        S[:, j] = sim.max(dim=1).values
        has[j] = True
    return dbnorm(S), has


def mean_proto_leg(Q, proto_path, cand):
    P = F.normalize(torch.load(proto_path, weights_only=False)['protos'].float(), dim=-1)[cand]
    has = P.norm(dim=-1) > 0.5
    S = Q @ P.t()
    S = S.masked_fill(~has.unsqueeze(0), -1e4)
    return dbnorm(S), has


def main():
    D = FishData()
    cand = D.cand
    S0, gold, qH = stack_base(D, cand, 1.0)
    bank_ctft = torch.load(BANK_CTFT, weights_only=False)['bank']
    Smp = score_maxpool(qH, bank_ctft, cand, topm=4)
    base = S0 + IMG_W * Smp

    n_idx, n_feat, _ = load_emb(QENC_B2)
    Qb2, _ = queries(D, n_idx, n_feat)

    S_mean, _ = mean_proto_leg(Qb2, MEAN_PROTO, cand)
    ref_mean = top1(base + B2_W * S_mean, gold)
    print(f'ref mean-proto @ w={B2_W}: {ref_mean:.2f} (target {REF_FROZEN25:.2f})', flush=True)

    best, best_cfg = ref_mean, {'K': 1, 'w': B2_W}
    rows = []
    bank_b2 = torch.load(BANK_B2, weights_only=False)['bank']
    for K in (2, 3, 4, 5, 6):
        S_km, _ = multi_proto_leg(Qb2, bank_b2, cand, K)
        for w in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
            acc = top1(base + w * S_km, gold)
            d = acc - ref_mean
            rows.append({'K': K, 'w': w, 'proxy': acc, 'delta_vs_mean25': d})
            print(f'  K={K} w={w}: {acc:.2f} ({d:+.2f})', flush=True)
            if acc > best:
                best, best_cfg = acc, {'K': K, 'w': w}

    delta = best - ref_mean
    out = {
        'recipe': 'frozen B2 k-means multi-prototype vs mean-proto',
        'ref_mean_w25': ref_mean,
        'ref_target': REF_FROZEN25,
        'best': best,
        'best_cfg': best_cfg,
        'delta_vs_mean25': delta,
        'clears_kill_03': delta >= 0.3,
        'proj_real_008': 50.57 + 0.08 * delta,
        'sweep': rows,
    }
    op = os.path.join(OUT, 'b2_kmeans_proto_proxy.json')
    json.dump(out, open(op, 'w'), indent=1)
    print(f'BEST {best:.2f} cfg={best_cfg} d={delta:+.2f}', flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()