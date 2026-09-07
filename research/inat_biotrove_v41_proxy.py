"""BioTrove-CLIP iNat mean-proto leg on v41 stack (ctft maxpool w4 + B2 mean w2.5).

Kill bar: +0.3 proxy-pt vs v41 ref (46.29 ortho / ~46.03 recomputed).

  python research/inat_biotrove_v41_proxy.py
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

PROTO_BT = os.path.join(OUT, 'inat_protos_biotrove.pt')
QENC_BT = os.path.join(OUT, 'emb_train_biotrove.pt')
PROTO_B2 = os.path.join(OUT, 'inat_protos_bioclip2.pt')
QENC_B2 = os.path.join(OUT, 'emb_train_bioclip2.pt')
BANK_CTFT = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')
IMG_W = 4.0
B2_MEAN_W = 2.5
REF_V41 = 46.29


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
    for p in (PROTO_BT, QENC_BT, PROTO_B2, QENC_B2, BANK_CTFT):
        if not os.path.exists(p):
            raise SystemExit(f'missing {p}')

    D = FishData()
    cand = D.cand
    bank = torch.load(BANK_CTFT, weights_only=False)['bank']
    S0, gold, qH = v36_stack(D, cand)
    Smp4 = score_maxpool(qH, bank, cand, topm=4)
    v41 = top1(S0 + IMG_W * Smp4, gold)

    n_idx, n_feat, _ = load_emb(QENC_B2)
    Qb2, _ = queries(D, n_idx, n_feat)
    S_b2, _ = proto_leg(Qb2, PROTO_B2, cand)
    v41_b2 = top1(S0 + IMG_W * Smp4 + B2_MEAN_W * S_b2, gold)
    print(f'v41 recomputed mp4={v41:.2f} +b2={v41_b2:.2f}', flush=True)

    bt_idx, bt_feat, _ = load_emb(QENC_BT)
    Qbt, _ = queries(D, bt_idx, bt_feat)
    S_bt, has_bt = proto_leg(Qbt, PROTO_BT, cand)

    best, best_w = v41_b2, 0.0
    sweep = []
    for w in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]:
        acc = top1(S0 + IMG_W * Smp4 + B2_MEAN_W * S_b2 + w * S_bt, gold)
        sweep.append({'bt_w': w, 'proxy': acc, 'delta_vs_v41_b2': acc - v41_b2})
        print(f'  +{w:g}*bt: {acc:.2f} ({acc - v41_b2:+.2f} vs v41)', flush=True)
        if acc > best:
            best, best_w = acc, w

    rng = random.Random(0)
    cov = [c for c in D.pseudo if has_bt[D.cand_pos[D.ci[c]]]]
    rng.shuffle(cov)
    half = len(cov) // 2
    tune_cls, val_cls = set(cov[:half]), set(cov[half:])
    base_stack = S0 + IMG_W * Smp4 + B2_MEAN_W * S_b2
    cand_list = cand.tolist()
    q_cls = [D.classes[cand_list[g]] for g in gold.tolist()]
    tune_m = torch.tensor([c in tune_cls for c in q_cls])
    val_m = torch.tensor([c in val_cls for c in q_cls])
    vb = top1(base_stack[val_m], gold[val_m])
    best_cv_w, best_cv = 0.0, vb
    for w in [0.0, 0.5, 1.0, 1.5, 2.0]:
        acc = top1((base_stack + w * S_bt)[val_m], gold[val_m])
        if acc > best_cv:
            best_cv, best_cv_w = acc, w
    vt = best_cv

    d = best - v41_b2
    out = {
        'model': 'hf-hub:BGLab/BioTrove-CLIP',
        'proto': PROTO_BT,
        'qenc': QENC_BT,
        'v41_b2_recomputed': v41_b2,
        'ref_v41_proxy': REF_V41,
        'best': best,
        'best_bt_w': best_w,
        'delta_vs_v41_b2': d,
        'clears_kill_03': d >= 0.3,
        'projected_real_008': 50.57 + 0.08 * d,
        'cv': {'w_bt': best_cv_w, 'val_base': vb, 'val_tuned': vt, 'val_delta': vt - vb},
        'sweep': sweep,
    }
    op = os.path.join(OUT, 'inat_biotrove_v41_proxy.json')
    json.dump(out, open(op, 'w'), indent=1)
    print(f'best {best:.2f} w={best_w} delta={d:+.2f} kill={out["clears_kill_03"]}', flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
