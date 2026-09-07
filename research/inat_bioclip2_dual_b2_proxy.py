"""v41 stack: simultaneous frozen B2 + LoRA-v2 mean proto legs (class-disjoint CV + holdout).

  conda activate onet && python research/inat_bioclip2_dual_b2_proxy.py
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
BANK = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')
FROZEN_PROTO = os.path.join(OUT, 'inat_protos_bioclip2.pt')
FROZEN_Q = os.path.join(OUT, 'emb_train_bioclip2.pt')
LORA_PROTO = os.path.join(OUT, 'inat_protos_bioclip2_lora_v2.pt')
LORA_Q = os.path.join(OUT, 'emb_train_bioclip2_lora_v2.pt')


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
    for p in (FROZEN_PROTO, FROZEN_Q, LORA_PROTO, LORA_Q, BANK):
        if not os.path.exists(p):
            raise SystemExit(f'missing {p}')

    D = FishData()
    cand = D.cand
    S0, gold, qH = stack_base(D, cand, 1.0)
    bank = torch.load(BANK, weights_only=False)['bank']
    Smp = score_maxpool(qH, bank, cand, topm=4)
    base = S0 + IMG_W * Smp

    fi, ff, _ = load_emb(FROZEN_Q)
    Qf, _ = queries(D, fi, ff)
    S_frozen, _ = proto_leg(Qf, FROZEN_PROTO, cand)

    li, lf, _ = load_emb(LORA_Q)
    Ql, _ = queries(D, li, lf)
    S_lora, _ = proto_leg(Ql, LORA_PROTO, cand)

    ref_single = top1(base + 2.5 * S_frozen, gold)
    ref_lora_only = top1(base + 1.5 * S_lora, gold)

    best, best_cfg = ref_lora_only, {'mode': 'lora_only', 'wf': 0.0, 'wl': 1.5}
    rows = []

    for wf in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        for wl in (0.0, 0.5, 1.0, 1.5, 2.0):
            if wf == 0.0 and wl == 0.0:
                continue
            acc = top1(base + wf * S_frozen + wl * S_lora, gold)
            d = acc - ref_single
            rows.append({'wf': wf, 'wl': wl, 'proxy': acc, 'delta_vs_frozen25': d})
            if acc > best:
                best, best_cfg = acc, {'mode': 'dual', 'wf': wf, 'wl': wl}

    delta = best - ref_single
    out = {
        'recipe': 'v41 dual B2 frozen + LoRA-v2 mean proto',
        'ref_v41_frozen_w25': ref_single,
        'ref_lora_only_w15': ref_lora_only,
        'best_proxy': best,
        'best_cfg': best_cfg,
        'delta_vs_frozen25': delta,
        'clears_kill_03': delta >= 0.3,
        'proj_real_008': 50.57 + 0.08 * delta,
        'sweep': rows,
    }
    op = os.path.join(OUT, 'inat_bioclip2_dual_b2_proxy.json')
    json.dump(out, open(op, 'w'), indent=1)
    print(f'REF frozen@2.5={ref_single:.2f} lora@1.5={ref_lora_only:.2f}', flush=True)
    print(f'BEST {best:.2f} cfg={best_cfg} d={delta:+.2f}', flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
