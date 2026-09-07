"""Holdout: aspect-adaptive crop-max + margin blend vs v56 recipe.

Uses cached holdout crop views. AR from the image itself (legal at deploy).

  conda activate onet && python -u research/v58_adaptive_proxy.py
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT
from crop_chase53_proxy import HOLD_VIEWS, OVERLAP_KEYS, max_dbnorm
from crop_views_proxy import BANK, FROZEN_PROTO, FROZEN_Q, IMG_ROOT, LORA_PROTO, LORA_Q, index_images, score_bank_raw
from inat_maxpool_proxy import score_maxpool, top1
from v41_taxctx_weight_cv import stack_base
from v50_shiftbank_proxy import BANK_336, B2_WF, B2_WL, F336_Q, IMG_W, proto_leg, queries

KILL = 0.3


def aspect(path):
    with Image.open(path) as im:
        w, h = im.size
    return max(w, h) / max(1, min(w, h))


def main():
    D = FishData()
    cand = D.cand
    S0, gold, qH = stack_base(D, cand, 1.0)
    bank = torch.load(BANK, weights_only=False)['bank']
    Sct = score_maxpool(qH, bank, cand, topm=4)
    Sf = proto_leg(queries(D, *load_emb(FROZEN_Q)[:2])[0], FROZEN_PROTO, cand)
    Sl = proto_leg(queries(D, *load_emb(LORA_Q)[:2])[0], LORA_PROTO, cand)
    base_rest = S0 + B2_WF * Sf + B2_WL * Sl
    S336 = score_maxpool(
        queries(D, *load_emb(F336_Q)[:2])[0],
        torch.load(BANK_336, weights_only=False)['bank'], cand, topm=4,
    )
    packed = torch.load(HOLD_VIEWS, weights_only=False)
    views = {k: F.normalize(v.float(), dim=-1) for k, v in packed['views'].items()}
    files = packed['files']
    S_ov = max_dbnorm(views, ['center', 'squash'] + OVERLAP_KEYS, bank, cand)
    ref50 = top1(base_rest + IMG_W * Sct, gold)
    ref56 = top1(base_rest + IMG_W * S_ov + 3.0 * S336, gold)
    print(f'REF50={ref50:.4f} REF56={ref56:.4f}', flush=True)

    imgidx = index_images(IMG_ROOT)
    ar = torch.zeros(len(files))
    miss = 0
    for i, fn in enumerate(files):
        p = imgidx.get(fn)
        if p is None:
            miss += 1
            ar[i] = 1.0
            continue
        try:
            ar[i] = aspect(p)
        except Exception:
            miss += 1
            ar[i] = 1.0
    print(f'AR n={len(files)} miss={miss} med={ar.median():.3f} p90={ar.quantile(0.9):.3f}', flush=True)

    rows = []
    best, best_name = ref56, 'v56'
    for thr in (1.2, 1.4, 1.6, 1.8, 2.0, 2.2):
        raw_sq = score_bank_raw(views['squash'], bank, cand)
        raw_ov = torch.stack([score_bank_raw(views[k], bank, cand) for k in ['center', 'squash'] + OVERLAP_KEYS]).max(0).values
        raw = torch.where(ar.unsqueeze(1) >= thr, raw_ov, raw_sq)
        S = dbnorm(raw)
        acc = top1(base_rest + IMG_W * S + 3.0 * S336, gold)
        d = acc - ref56
        rows.append({'mode': f'ar>={thr}', 'proxy': acc, 'd_v56': d, 'frac_strips': float((ar >= thr).float().mean())})
        print(f'ar>={thr:.1f} strips_frac={(ar>=thr).float().mean():.3f}: {acc:.4f} ({d:+.3f})', flush=True)
        if acc > best:
            best, best_name = acc, f'ar>={thr}'

    # margin blend: take overlap max only when its top1-top2 gap exceeds squash
    raw_sq = score_bank_raw(views['squash'], bank, cand)
    raw_ov = torch.stack(
        [score_bank_raw(views[k], bank, cand) for k in ['center', 'squash'] + OVERLAP_KEYS]
    ).max(0).values
    top2_ov = raw_ov.topk(2, dim=1).values
    top2_sq = raw_sq.topk(2, dim=1).values
    gap_ov = top2_ov[:, 0] - top2_ov[:, 1]
    gap_sq = top2_sq[:, 0] - top2_sq[:, 1]
    for dgap in (0.0, 0.01, 0.02, 0.04, 0.06):
        pick = (gap_ov >= gap_sq + dgap).unsqueeze(1)
        raw = torch.where(pick, raw_ov, raw_sq)
        acc = top1(base_rest + IMG_W * dbnorm(raw) + 3.0 * S336, gold)
        d = acc - ref56
        rows.append({'mode': f'margin_dgap={dgap}', 'proxy': acc, 'd_v56': d,
                     'frac_ov': float(pick.float().mean())})
        print(f'margin Δ≥{dgap}: frac_ov={pick.float().mean():.3f} {acc:.4f} ({d:+.3f})', flush=True)
        if acc > best:
            best, best_name = acc, f'margin_dgap={dgap}'

    # softmax view pool (less peaky than hard max)
    raws = [score_bank_raw(views[k], bank, cand) for k in ['center', 'squash'] + OVERLAP_KEYS]
    stacked = torch.stack(raws, 0)
    for t in (0.02, 0.05, 0.1, 0.2):
        w = torch.softmax(stacked / t, dim=0)
        acc = top1(base_rest + IMG_W * dbnorm((w * stacked).sum(0)) + 3.0 * S336, gold)
        d = acc - ref56
        rows.append({'mode': f'soft_t={t}', 'proxy': acc, 'd_v56': d})
        print(f'soft t={t}: {acc:.4f} ({d:+.3f})', flush=True)
        if acc > best:
            best, best_name = acc, f'soft_t={t}'

    out = {
        'ref50': ref50, 'ref56': ref56, 'best': best, 'best_name': best_name,
        'delta_vs_v56': best - ref56,
        'clears_kill_vs_v56': (best - ref56) >= KILL,
        'rows': sorted(rows, key=lambda r: -r['proxy'])[:20],
    }
    op = os.path.join(OUT, 'v58_adaptive_proxy.json')
    json.dump(out, open(op, 'w'), indent=2)
    print(json.dumps({k: out[k] for k in out if k != 'rows'}, indent=2), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
