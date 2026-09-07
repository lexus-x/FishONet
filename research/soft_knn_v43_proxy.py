"""Soft-kNN / logsumexp photo retrieval vs maxpool on v43-like stack.

Bank format: dict gidx -> [k,D] (same as score_maxpool).

  conda activate onet && python research/soft_knn_v43_proxy.py
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

IMG_W = 4.0
WF, WL = 0.5, 1.0
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
    return dbnorm(S)


def score_soft_knn(Q, bank, cand, k: int, temp: float):
    """Per-class logsumexp of top-k photo sims."""
    N = Q.shape[0]
    C = len(cand)
    S = torch.full((N, C), -1e4)
    for j, gidx in enumerate(cand.tolist()):
        photos = bank.get(gidx)
        if photos is None or photos.numel() == 0:
            continue
        sim = Q @ photos.t()
        kk = min(k, sim.shape[1])
        topv = sim.topk(kk, dim=1).values
        S[:, j] = (topv / temp).logsumexp(dim=1)
    return dbnorm(S)


def main():
    D = FishData()
    cand = D.cand
    S0, gold, qH = stack_base(D, cand, 1.0)
    bank = torch.load(BANK, weights_only=False)['bank']
    Smp = score_maxpool(qH, bank, cand, topm=4)
    base_mp = S0 + IMG_W * Smp

    fi, ff, _ = load_emb(FROZEN_Q)
    Qf, _ = queries(D, fi, ff)
    Sf = proto_leg(Qf, FROZEN_PROTO, cand)
    li, lf, _ = load_emb(LORA_Q)
    Ql, _ = queries(D, li, lf)
    Sp = proto_leg(Ql, LORA_PROTO, cand)
    ref = top1(base_mp + WF * Sf + WL * Sp, gold)
    print(f'REF={ref:.4f}', flush=True)

    rows = []
    best, best_cfg = ref, {'mode': 'ref'}
    for k in (4, 8, 16):
        for temp in (0.05, 0.07, 0.1):
            print(f'scoring knn k={k} temp={temp}...', flush=True)
            Sk = score_soft_knn(qH, bank, cand, k=k, temp=temp)
            for w in (2.0, 3.0, 4.0, 5.0):
                acc = top1(S0 + w * Sk + WF * Sf + WL * Sp, gold)
                row = {'k': k, 'temp': temp, 'w': w, 'proxy': acc, 'delta': acc - ref}
                rows.append(row)
                if acc > best:
                    best, best_cfg = acc, row
            for w in (1.0, 2.0):
                acc = top1(base_mp + w * Sk + WF * Sf + WL * Sp, gold)
                row = {
                    'k': k, 'temp': temp, 'w_blend': w, 'proxy': acc, 'delta': acc - ref,
                }
                rows.append(row)
                if acc > best:
                    best, best_cfg = acc, row

    delta = best - ref
    out = {
        'ref': ref,
        'best': best,
        'best_cfg': best_cfg,
        'delta': delta,
        'clears_kill_03': delta >= 0.3,
        'proj_real_036': 50.75564278704613 + 0.36 * delta,
        'sweep_top': sorted(rows, key=lambda r: -r['proxy'])[:15],
    }
    op = os.path.join(OUT, 'soft_knn_v43_proxy.json')
    json.dump(out, open(op, 'w'), indent=2)
    print(json.dumps({k: out[k] for k in out if k != 'sweep_top'}, indent=2), flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
