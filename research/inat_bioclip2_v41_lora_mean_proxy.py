"""Honest v41 proxy with LoRA-tuned BioCLIP-2 mean proto leg (env overrides paths).

  B2_PROTO=outputs/inat_protos_bioclip2_lora_v2.pt \\
  B2_QENC=outputs/emb_train_bioclip2_lora_v2.pt \\
  python research/inat_bioclip2_v41_lora_mean_proxy.py
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

REF_V41 = 46.03
IMG_W = 4.0
B2_W_DEFAULT = 2.5
PROTO_B2 = os.environ.get('B2_PROTO', os.path.join(OUT, 'inat_protos_bioclip2_lora.pt'))
QENC_B2 = os.environ.get('B2_QENC', os.path.join(OUT, 'emb_train_bioclip2_lora.pt'))
BANK = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')


def queries(D, idx, feats):
    q, y = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx:
                q.append(feats[idx[fn]])
                y.append(D.cand_pos[D.ci[c]])
    return torch.stack(q), torch.tensor(y)


def proto_leg(Q, proto_path, cand):
    P = F.normalize(torch.load(proto_path, weights_only=False)['protos'].float(), dim=-1)[cand]
    has = P.norm(dim=-1) > 0.5
    S = Q @ P.t()
    S = S.masked_fill(~has.unsqueeze(0), -1e4)
    return dbnorm(S), has


def main():
    for p in (PROTO_B2, QENC_B2, BANK):
        if not os.path.exists(p):
            raise SystemExit(f'missing {p}')

    D = FishData()
    cand = D.cand
    S0, gold, qH = stack_base(D, cand, 1.0)
    bank = torch.load(BANK, weights_only=False)['bank']
    Smp = score_maxpool(qH, bank, cand, topm=4)
    v41_frozen_idx, v41_frozen_feat, _ = load_emb(os.path.join(OUT, 'emb_train_bioclip2.pt'))
    Qf, _ = queries(D, v41_frozen_idx, v41_frozen_feat)
    S_frozen, _ = proto_leg(Qf, os.path.join(OUT, 'inat_protos_bioclip2.pt'), cand)
    ref = top1(S0 + IMG_W * Smp + B2_W_DEFAULT * S_frozen, gold)

    n_idx, n_feat, _ = load_emb(QENC_B2)
    Qb2, _ = queries(D, n_idx, n_feat)
    S_b2, _ = proto_leg(Qb2, PROTO_B2, cand)
    base = S0 + IMG_W * Smp

    best, best_w = ref, B2_W_DEFAULT
    rows = []
    for w in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        acc = top1(base + w * S_b2, gold)
        d = acc - ref
        rows.append({'b2_w': w, 'proxy': acc, 'delta_vs_ref': d})
        print(f'  B2_W={w:g}: {acc:.2f} ({d:+.2f})', flush=True)
        if acc > best:
            best, best_w = acc, w

    delta = best - ref
    out = {
        'recipe': 'v41 stack with B2 LoRA mean proto + LoRA train queries',
        'proto': PROTO_B2,
        'qenc': QENC_B2,
        'ref_v41_frozen': ref,
        'best_w': best_w,
        'best_proxy': best,
        'delta_vs_ref': delta,
        'clears_kill_03': delta >= 0.3,
        'proj_real_008': 50.57 + 0.08 * delta,
        'zip_recommend': delta >= 0.3,
        'sweep': rows,
    }
    tag = os.environ.get('OUT_TAG', 'inat_bioclip2_v41_lora_mean_proxy')
    op = os.path.join(OUT, f'{tag}.json')
    json.dump(out, open(op, 'w'), indent=1)
    print(f'REF {ref:.2f} BEST {best:.2f} @ w={best_w} d={delta:+.2f} kill={out["clears_kill_03"]}', flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
