"""Holdout: 336 crop-max vs v56 recipe (ctft overlap-max + 336 mean-TTA bank).

Needs outputs/emb_holdout_fullft336_cropviews.pt from extract_eval_336_crops.py.

  conda activate onet && python research/v57_336crop_proxy.py
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT
from crop_views_proxy import BANK, score_bank_raw
from inat_maxpool_proxy import score_maxpool, top1
from v41_taxctx_weight_cv import stack_base
from v50_shiftbank_proxy import BANK_336, B2_WF, B2_WL, F336_Q, IMG_W, proto_leg, queries
from crop_views_proxy import FROZEN_PROTO, FROZEN_Q, LORA_PROTO, LORA_Q
from crop_chase53_proxy import BASE_KEYS, HOLD_VIEWS, OVERLAP_KEYS, max_dbnorm

KILL = 0.3
HOLD_336 = os.path.join(OUT, 'emb_holdout_fullft336_cropviews.pt')
KEYS_336 = ['center', 'squash'] + [f'ostrip{i}' for i in range(5)]
BANK_336_CROP = os.path.join(OUT, 'inat_photo_bank_fullft336_cropviews.pt')


def csls(S, k=10):
    qn = S.topk(min(k, S.shape[1]), dim=1).values.mean(1, keepdim=True)
    cn = S.topk(min(k, S.shape[0]), dim=0).values.mean(0, keepdim=True)
    return 2 * S - qn - cn


def softmax_pool(raws, t):
    stacked = torch.stack(raws, dim=0)
    if t <= 0:
        return stacked.max(0).values
    w = torch.softmax(stacked / t, dim=0)
    return (w * stacked).sum(0)


def main():
    D = FishData()
    cand = D.cand
    S0, gold, qH = stack_base(D, cand, 1.0)
    bank_ct = torch.load(BANK, weights_only=False)['bank']
    Sct = score_maxpool(qH, bank_ct, cand, topm=4)
    Sf = proto_leg(queries(D, *load_emb(FROZEN_Q)[:2])[0], FROZEN_PROTO, cand)
    Sl = proto_leg(queries(D, *load_emb(LORA_Q)[:2])[0], LORA_PROTO, cand)
    base_rest = S0 + B2_WF * Sf + B2_WL * Sl
    S336 = score_maxpool(queries(D, *load_emb(F336_Q)[:2])[0],
                         torch.load(BANK_336, weights_only=False)['bank'], cand, topm=4)
    ref50 = top1(base_rest + IMG_W * Sct, gold)
    packed = torch.load(HOLD_VIEWS, weights_only=False)
    ct_views = {k: F.normalize(v.float(), dim=-1) for k, v in packed['views'].items()}
    S_ov = max_dbnorm(ct_views, ['center', 'squash'] + OVERLAP_KEYS, bank_ct, cand)
    ref56 = top1(base_rest + IMG_W * S_ov + 3.0 * S336, gold)
    print(f'REF v50-like={ref50:.4f}  REF v56-like={ref56:.4f}', flush=True)

    rows = []
    best, best_name = ref56, 'v56-like'
    if os.path.exists(HOLD_336):
        h336 = torch.load(HOLD_336, weights_only=False)
        t336 = {k: F.normalize(h336['views'][k].float(), dim=-1) for k in KEYS_336}
        bank336 = torch.load(BANK_336, weights_only=False)['bank']
        raws = [score_bank_raw(t336[k], bank336, cand, topm=4) for k in KEYS_336]
        for name, Sraw in {
            '336_max': torch.stack(raws).max(0).values,
            '336_soft_t0.05': softmax_pool(raws, 0.05),
            '336_soft_t0.1': softmax_pool(raws, 0.1),
            '336_csls_max': csls(torch.stack(raws).max(0).values, 10),
        }.items():
            S = dbnorm(Sraw)
            for wct, w336 in ((4.0, 0.0), (4.0, 2.0), (4.0, 3.0), (0.0, 3.0), (0.0, 4.0)):
                acc = top1(base_rest + wct * S_ov + w336 * S, gold)
                d56 = acc - ref56
                d50 = acc - ref50
                rows.append({'mode': name, 'w_ct': wct, 'w_336': w336, 'proxy': acc,
                             'd_v56': d56, 'd_v50': d50})
                print(f'{name} ct{wct:g}+336{w336:g}: {acc:.4f} (vs56 {d56:+.3f})', flush=True)
                if acc > best:
                    best, best_name = acc, f'{name}|ct{wct}|336{w336}'
        if os.path.exists(BANK_336_CROP):
            bcrop = torch.load(BANK_336_CROP, weights_only=False)['bank']
            Sqc = max_dbnorm(t336, KEYS_336, bcrop, cand)
            acc = top1(base_rest + IMG_W * S_ov + 3.0 * Sqc, gold)
            rows.append({'mode': '336Q_x_336cropbank', 'w_ct': 4.0, 'w_336': 3.0,
                         'proxy': acc, 'd_v56': acc - ref56, 'd_v50': acc - ref50})
            print(f'336Q x cropbank: {acc:.4f} (vs56 {acc-ref56:+.3f})', flush=True)
            if acc > best:
                best, best_name = acc, '336Q_x_336cropbank'
    else:
        print(f'missing {HOLD_336} — extract still running', flush=True)

    out = {
        'ref50': ref50,
        'ref56': ref56,
        'best': best,
        'best_name': best_name,
        'delta_vs_v56': best - ref56,
        'clears_kill_vs_v56': (best - ref56) >= KILL,
        'proj_real_012': 51.61643067433057 + 0.12 * (best - ref56),
        'rows': sorted(rows, key=lambda r: -r['proxy'])[:25],
    }
    op = os.path.join(OUT, 'v57_336crop_proxy.json')
    json.dump(out, open(op, 'w'), indent=2)
    print(json.dumps({k: out[k] for k in out if k != 'rows'}, indent=2), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
