"""Class-disjoint CV + holdout sweep for B2_W on v41 deploy recipe (mp4 + BioCLIP-2 mean proto)."""
from __future__ import annotations

import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, load_emb, OUT  # noqa: E402
from inat_maxpool_proxy import score_maxpool, top1, v36_stack, mean_proto_baseline  # noqa: E402

PROTO_B2 = os.path.join(OUT, 'inat_protos_bioclip2.pt')
QENC_B2 = os.path.join(OUT, 'emb_train_bioclip2.pt')
BANK = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')
IMG_W = 4.0
V41_B2 = 2.5
REF_V41_PROXY = 46.29  # ortho best @ mp4+w2.5 mean track approx


def proto_leg(Q, proto_path, cand):
    from common import dbnorm

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


def cv_pick(S0, gold, D, cand, S_add, tune_cls, val_cls, weights):
    cand_list = cand.tolist()
    q_cls = [D.classes[cand_list[g]] for g in gold.tolist()]
    tune_m = torch.tensor([c in tune_cls for c in q_cls])
    val_m = torch.tensor([c in val_cls for c in q_cls])
    base = S0 + IMG_W * S_add
    base_tune = top1(base[tune_m], gold[tune_m])
    best_w, best_tune = V41_B2, base_tune
    for w in weights:
        acc = top1((base + w * S_b2)[tune_m], gold[tune_m])
        if acc > best_tune:
            best_tune, best_w = acc, w
    val_base = top1(base[val_m], gold[val_m])
    val_tuned = top1((base + best_w * S_b2)[val_m], gold[val_m])
    return best_w, val_base, val_tuned


def main():
    D = FishData()
    cand = D.cand
    bank = torch.load(BANK, weights_only=False)['bank']
    S0, gold, qH = v36_stack(D, cand)
    Smp4 = score_maxpool(qH, bank, cand, topm=4)
    n_idx, n_feat, _ = load_emb(QENC_B2)
    Qb2, _ = queries(D, n_idx, n_feat)
    global S_b2
    S_b2, _ = proto_leg(Qb2, PROTO_B2, cand)

    weights = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    base = S0 + IMG_W * Smp4
    v41_ref = top1(base + V41_B2 * S_b2, gold)
    rows = []
    for w in weights:
        acc = top1(base + w * S_b2, gold)
        d = acc - v41_ref
        proj = 50.57 + 0.08 * (acc - v41_ref)
        rows.append({'b2_w': w, 'proxy': acc, 'delta_vs_v41_b25': d, 'proj_real_008': proj})
        print(f'B2_W={w:g} proxy={acc:.2f} d_vs_b25={d:+.2f} proj_real={proj:.3f}', flush=True)

    rng = random.Random(0)
    cov = [c for c in D.pseudo if True]
    rng.shuffle(cov)
    half = len(cov) // 2
    tune_cls, val_cls = set(cov[:half]), set(cov[half:])
    w_cv, vb, vt = cv_pick(S0, gold, D, cand, Smp4, tune_cls, val_cls, weights)

    out = {
        'recipe': 'S0 + 4*maxpool_top4 + B2_W*b2_mean_proto',
        'v41_ref_proxy_b25': v41_ref,
        'real_anchor': {'overall': 50.57, 'transfer_overall_per_proxy_pt': 0.08},
        'sweep': rows,
        'cv': {'picked_w': w_cv, 'val_base': vb, 'val_tuned': vt, 'val_delta': vt - vb},
    }
    best = max(rows, key=lambda r: r['proxy'])
    out['build_zip_if'] = [
        r for r in rows
        if r['proj_real_008'] >= 50.57 + 0.05 and r['proxy'] >= best['proxy'] - 0.01
    ]
    op = os.path.join(OUT, 'inat_bioclip2_b2w_sweep.json')
    json.dump(out, open(op, 'w'), indent=1)
    print(f'CV pick w={w_cv} val_delta={vt - vb:+.3f} wrote {op}', flush=True)


if __name__ == '__main__':
    main()
