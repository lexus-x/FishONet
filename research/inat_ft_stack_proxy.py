"""Honest pseudo-unseen proxy: ctftshift_inat replaces ctftshift in v36 text stack + iNat legs.

Compares mean-proto @ w3 (v37 ref 43.53) and maxpool top4 @ w3.5 (v40 ref 45.17)
against the same stack with the original ctftshift encoder.

  python research/inat_ft_stack_proxy.py
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
from inat_maxpool_proxy import score_maxpool, top1  # noqa: E402

CKPT = 'outputs/ctft_shift_inat.pt'
PROTO_INAT = os.path.join(OUT, 'inat_protos_ctftshift_inat_full.pt')
BANK_INAT = os.path.join(OUT, 'inat_photo_bank_ctftshift_inat.pt')
PROTO_BASE = os.path.join(OUT, 'inat_protos_ctftshift_full.pt')
BANK_BASE = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')


def queries(D, idx, feats):
    q, y = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx:
                q.append(feats[idx[fn]])
                y.append(D.cand_pos[D.ci[c]])
    return torch.stack(q), torch.tensor(y)


def text_stack(D, cand, q_ctft):
    TtH_c = D.TtH[cand]
    TnL_c = D.TnL[cand]
    TTX = F.normalize(
        torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(),
        dim=-1,
    )
    TTX_c = TTX[cand]
    legs = {}
    for tag in ['ftshift', 'fullft336_v2']:
        idx, feats, _ = load_emb(os.path.join(OUT, f'emb_train_{tag}.pt'))
        legs[tag], _ = queries(D, idx, feats)
    qL, gold = queries(D, D.LtI, D.LtF)
    S0 = (
        dbnorm(q_ctft @ TtH_c.t())
        + 0.5 * dbnorm(qL @ TnL_c.t())
        + 0.75 * dbnorm(legs['fullft336_v2'] @ TtH_c.t())
        + 1.0 * dbnorm(legs['ftshift'] @ TtH_c.t())
        + 1.0 * dbnorm(q_ctft @ TTX_c.t())
    )
    return S0, gold


def mean_proto(cand, qenc, proto_path):
    d = torch.load(proto_path, weights_only=False)
    P = F.normalize(d['protos'].float(), dim=-1)
    Pc = P[cand]
    has = Pc.norm(dim=-1) > 0.5
    S = qenc @ Pc.t()
    S = S.masked_fill(~has.unsqueeze(0), -1e4)
    return dbnorm(S), has


def eval_encoder(D, cand, enc_tag, proto_path, bank):
    idx, feats, _ = load_emb(os.path.join(OUT, f'emb_train_{enc_tag}.pt'))
    q_ctft, gold = queries(D, idx, feats)
    S0, _ = text_stack(D, cand, q_ctft)
    base = top1(S0, gold)
    Smean, has = mean_proto(cand, q_ctft, proto_path)
    mean_w3 = top1(S0 + 3.0 * Smean, gold)
    Smp4 = score_maxpool(q_ctft, bank, cand, topm=4)
    mp_w35 = top1(S0 + 3.5 * Smp4, gold)
    mp_w4 = top1(S0 + 4.0 * Smp4, gold)
    return {
        'enc': enc_tag,
        'proto': proto_path,
        'baseline_text': base,
        'mean_w3': mean_w3,
        'maxpool_top4_w35': mp_w35,
        'maxpool_top4_w4': mp_w4,
        'has': has,
        'S0': S0,
        'gold': gold,
        'q': q_ctft,
        'Smean': Smean,
        'Smp4': Smp4,
    }


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
    bank_base = torch.load(BANK_BASE, weights_only=False)['bank']
    bank_inat = torch.load(BANK_INAT, weights_only=False)['bank']

    ref = eval_encoder(D, cand, 'ctftshift', PROTO_BASE, bank_base)
    ft = eval_encoder(D, cand, 'ctftshift_inat', PROTO_INAT, bank_inat)

    res = {
        'ckpt': CKPT,
        'ref_ctftshift': {k: ref[k] for k in ref if k not in ('has', 'S0', 'gold', 'q', 'Smean', 'Smp4')},
        'inat_ft': {k: ft[k] for k in ft if k not in ('has', 'S0', 'gold', 'q', 'Smean', 'Smp4')},
        'delta_mean_w3': ft['mean_w3'] - ref['mean_w3'],
        'delta_maxpool_w35': ft['maxpool_top4_w35'] - ref['maxpool_top4_w35'],
        'ref_targets': {'mean_w3': 43.53, 'maxpool_w35': 45.17},
        'cv': {
            'ref_mean_val_delta': cv_delta(ref['S0'], ref['gold'], ref['has'], D, ref['Smean'], 3.0),
            'ft_mean_val_delta': cv_delta(ft['S0'], ft['gold'], ft['has'], D, ft['Smean'], 3.0),
            'ref_mp_val_delta': cv_delta(ref['S0'], ref['gold'], ref['has'], D, ref['Smp4'], 3.5),
            'ft_mp_val_delta': cv_delta(ft['S0'], ft['gold'], ft['has'], D, ft['Smp4'], 3.5),
        },
        'projected_real_overall_012': {
            'from_mean_delta': 50.49 + 0.12 * (ft['mean_w3'] - ref['mean_w3']),
            'from_mp_delta': 50.49 + 0.12 * (ft['maxpool_top4_w35'] - ref['maxpool_top4_w35']),
        },
    }

    out = os.path.join(OUT, 'inat_ft_stack_proxy_results.json')
    json.dump(res, open(out, 'w'), indent=1)
    print(f'[{time.time()-t0:.0f}s] ref mean@3={ref["mean_w3"]:.2f} mp@3.5={ref["maxpool_top4_w35"]:.2f}', flush=True)
    print(f'     ft mean@3={ft["mean_w3"]:.2f} ({res["delta_mean_w3"]:+.2f}) mp@3.5={ft["maxpool_top4_w35"]:.2f} ({res["delta_maxpool_w35"]:+.2f})', flush=True)
    print(f'     projected real (0.12/pt): mean {res["projected_real_overall_012"]["from_mean_delta"]:.2f}% mp {res["projected_real_overall_012"]["from_mp_delta"]:.2f}%', flush=True)
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
