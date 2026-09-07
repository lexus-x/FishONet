"""Honest v41 proxy: v40 ctft maxpool @ IMG_W=4 + B2 mean @ B2_W=2.5 + optional B2 maxpool leg.

Kill bar: +0.3 proxy-pt vs v41 ref (~+0.024 real @ 0.08 transfer on mean track;
use 0.08 for B2-family legs, not maxpool 0.02).

  python research/inat_bioclip2_v41_maxpool_proxy.py
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
from inat_maxpool_proxy import score_maxpool, top1, v36_stack  # noqa: E402

PROTO_B2 = os.path.join(OUT, 'inat_protos_bioclip2.pt')
QENC_B2 = os.path.join(OUT, 'emb_train_bioclip2.pt')
BANK_CTFT = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')
BANK_B2 = os.path.join(OUT, 'inat_photo_bank_bioclip2.pt')
IMG_W = 4.0
B2_MEAN_W = 2.5
REF_V41 = 46.29  # ortho holdout best @ mp4 + b2 mean w2.5


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


def main():
    for p in (PROTO_B2, QENC_B2, BANK_CTFT, BANK_B2):
        if not os.path.exists(p):
            raise SystemExit(f'missing {p} — run: python research/inat_maxpool_proxy.py --enc b2 --build-bank')

    D = FishData()
    cand = D.cand
    bank_ctft = torch.load(BANK_CTFT, weights_only=False)['bank']
    bank_b2 = torch.load(BANK_B2, weights_only=False)['bank']
    S0, gold, qH = v36_stack(D, cand)
    Smp_ctft = score_maxpool(qH, bank_ctft, cand, topm=4)

    n_idx, n_feat, _ = load_emb(QENC_B2)
    Qb2, _ = queries(D, n_idx, n_feat)
    S_b2_mean, has_b2 = proto_leg(Qb2, PROTO_B2, cand)
    S_b2_mp = score_maxpool(Qb2, bank_b2, cand, topm=4)

    v41_base = S0 + IMG_W * Smp_ctft + B2_MEAN_W * S_b2_mean
    v41_ref = top1(v41_base, gold)
    print(f'v41 base recomputed={v41_ref:.2f} (ref {REF_V41:.2f})', flush=True)

    best, best_w = v41_ref, 0.0
    rows = []
    for w in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        acc = top1(v41_base + w * S_b2_mp, gold)
        d = acc - v41_ref
        rows.append({'b2_mp_w': w, 'proxy': acc, 'delta_vs_v41': d})
        print(f'  v41 + {w:g}*b2_maxpool_top4: {acc:.2f} ({d:+.2f})', flush=True)
        if acc > best:
            best, best_w = acc, w

    alone_mp = top1(S_b2_mp, gold)
    print(f'b2 maxpool top4 alone={alone_mp:.2f}', flush=True)

    rng = random.Random(0)
    cov = [c for c in D.pseudo if has_b2[D.cand_pos[D.ci[c]]]]
    rng.shuffle(cov)
    half = len(cov) // 2
    tune_cls, val_cls = set(cov[:half]), set(cov[half:])
    cand_list = cand.tolist()
    q_cls = [D.classes[cand_list[g]] for g in gold.tolist()]
    tune_m = torch.tensor([c in tune_cls for c in q_cls])
    val_m = torch.tensor([c in val_cls for c in q_cls])

    cv_w, cv_tune = B2_MEAN_W, top1(v41_base[tune_m], gold[tune_m])
    for w in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        acc = top1((v41_base + w * S_b2_mp)[tune_m], gold[tune_m])
        if acc > cv_tune:
            cv_tune, cv_w = acc, w
    val_base = top1(v41_base[val_m], gold[val_m])
    val_tuned = top1((v41_base + cv_w * S_b2_mp)[val_m], gold[val_m])

    delta = best - v41_ref
    out = {
        'model': 'hf-hub:imageomics/bioclip-2',
        'bank_b2': BANK_B2,
        'v41_recomputed': v41_ref,
        'ref_v41_proxy': REF_V41,
        'b2_maxpool_alone_top4': alone_mp,
        'best': best,
        'best_b2_mp_w': best_w,
        'delta_vs_v41': delta,
        'clears_kill_03': delta >= 0.3,
        'projected_real_008': 50.57 + 0.08 * delta,
        'cv': {'w_b2_mp': cv_w, 'val_base': val_base, 'val_tuned': val_tuned,
               'val_delta': val_tuned - val_base},
        'sweep': rows,
    }
    op = os.path.join(OUT, 'inat_bioclip2_v41_maxpool_proxy.json')
    json.dump(out, open(op, 'w'), indent=1)
    print(f'BEST {best:.2f} d={delta:+.2f} kill={out["clears_kill_03"]} proj_real={out["projected_real_008"]:.2f}', flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
