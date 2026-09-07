"""Honest v40 proxy: add BioCLIP-2 iNat mean-proto leg on ctftshift maxpool stack.

Refs (v40): mean@w3=43.49, maxpool top4@w3.5=45.17, maxpool top4@w4=45.08.
Kill bar: +0.3 proxy-pt on mean or maxpool track.

  python research/inat_bioclip2_v40_proxy.py
"""
from __future__ import annotations

import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT  # noqa: E402
from inat_maxpool_proxy import score_maxpool, top1, v36_stack, mean_proto_baseline  # noqa: E402

REF_MEAN, REF_MP35, REF_MP4 = 43.49, 45.17, 45.08
PROTO_B2 = os.path.join(OUT, 'inat_protos_bioclip2.pt')
QENC_B2 = os.path.join(OUT, 'emb_train_bioclip2.pt')
BANK = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')


def proto_leg(Q, proto_path, cand):
    P = F.normalize(torch.load(proto_path, weights_only=False)['protos'].float(), dim=-1)[cand]
    has = P.norm(dim=-1) > 0.5
    S = Q @ P.t()
    S = S.masked_fill(~has.unsqueeze(0), -1e4)
    return dbnorm(S), has


def queries(D, idx, feats):
    q, y = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx:
                q.append(feats[idx[fn]])
                y.append(D.cand_pos[D.ci[c]])
    return torch.stack(q), torch.tensor(y)


def cv_pick(S0, gold, has_ctft, D, cand, Sm, tune_cls, val_cls, weights):
    cand_list = cand.tolist()
    q_cls = [D.classes[cand_list[g]] for g in gold.tolist()]
    tune_m = torch.tensor([c in tune_cls for c in q_cls])
    val_m = torch.tensor([c in val_cls for c in q_cls])
    base_tune = top1(S0[tune_m], gold[tune_m])
    best_w, best_tune = 0.0, base_tune
    for w in weights:
        acc = top1((S0 + w * Sm)[tune_m], gold[tune_m])
        if acc > best_tune:
            best_tune, best_w = acc, w
    val_base = top1(S0[val_m], gold[val_m])
    val_tuned = top1((S0 + best_w * Sm)[val_m], gold[val_m])
    return best_w, val_base, val_tuned


def main():
    for p in (PROTO_B2, QENC_B2, BANK):
        if not os.path.exists(p):
            raise SystemExit(f'missing {p} — wait for embed jobs')

    D = FishData()
    cand = D.cand
    bank = torch.load(BANK, weights_only=False)['bank']
    S0, gold, qH = v36_stack(D, cand)
    Smean, has_c = mean_proto_baseline(cand, qH)
    Smp4 = score_maxpool(qH, bank, cand, topm=4)

    v40_mean = top1(S0 + 3.0 * Smean, gold)
    v40_mp35 = top1(S0 + 3.5 * Smp4, gold)
    v40_mp4 = top1(S0 + 4.0 * Smp4, gold)
    print(f'v40 ref mean@3={v40_mean:.2f} mp@3.5={v40_mp35:.2f} mp@4={v40_mp4:.2f}', flush=True)

    n_idx, n_feat, _ = load_emb(QENC_B2)
    Qb2, _ = queries(D, n_idx, n_feat)
    S_b2, has_b2 = proto_leg(Qb2, PROTO_B2, cand)

    best_mean, best_w_mean = v40_mean, 0.0
    best_mp35, best_w_mp35 = v40_mp35, 0.0
    best_mp4, best_w_mp4 = v40_mp4, 0.0
    for w in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]:
        m = top1(S0 + 3.0 * Smean + w * S_b2, gold)
        p35 = top1(S0 + 3.5 * Smp4 + w * S_b2, gold)
        p4 = top1(S0 + 4.0 * Smp4 + w * S_b2, gold)
        print(f'  +{w:g}*b2: mean={m:.2f} mp35={p35:.2f} mp4={p4:.2f}', flush=True)
        if m > best_mean:
            best_mean, best_w_mean = m, w
        if p35 > best_mp35:
            best_mp35, best_w_mp35 = p35, w
        if p4 > best_mp4:
            best_mp4, best_w_mp4 = p4, w

    rng = random.Random(0)
    cov = [c for c in D.pseudo if has_c[D.cand_pos[D.ci[c]]]]
    rng.shuffle(cov)
    half = len(cov) // 2
    tune_cls, val_cls = set(cov[:half]), set(cov[half:])
    w_cv, vb, vt = cv_pick(S0 + 3.0 * Smean, gold, has_c, D, cand, S_b2, tune_cls, val_cls,
                           [0.0, 0.5, 1.0, 1.5, 2.0])

    d_mean = best_mean - REF_MEAN
    d_mp35 = best_mp35 - REF_MP35
    d_mp4 = best_mp4 - REF_MP4
    out = {
        'model': 'hf-hub:imageomics/bioclip-2',
        'proto': PROTO_B2,
        'qenc': QENC_B2,
        'v40_recomputed': {'mean_w3': v40_mean, 'maxpool_w35': v40_mp35, 'maxpool_w4': v40_mp4},
        'ref_targets': {'mean_w3': REF_MEAN, 'maxpool_w35': REF_MP35, 'maxpool_w4': REF_MP4},
        'best_mean': {'acc': best_mean, 'w_b2': best_w_mean, 'delta_vs_ref': d_mean},
        'best_maxpool_w35': {'acc': best_mp35, 'w_b2': best_w_mp35, 'delta_vs_ref': d_mp35},
        'best_maxpool_w4': {'acc': best_mp4, 'w_b2': best_w_mp4, 'delta_vs_ref': d_mp4},
        'cv_mean_track': {'w_b2': w_cv, 'val_base': vb, 'val_tuned': vt, 'val_delta': vt - vb},
        'projected_real_002_mp': 50.49 + 0.02 * d_mp35,
        'projected_real_012_mean': 50.49 + 0.12 * d_mean,
        'clears_kill_03': max(d_mean, d_mp35, d_mp4) >= 0.3,
    }
    op = os.path.join(OUT, 'inat_bioclip2_v40_proxy.json')
    json.dump(out, open(op, 'w'), indent=1)
    print(f'best mean {best_mean:.2f} ({d_mean:+.2f}) mp35 {best_mp35:.2f} ({d_mp35:+.2f}) kill={out["clears_kill_03"]}', flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
