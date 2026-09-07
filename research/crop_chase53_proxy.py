"""Holdout: stack crop-max (ctft iNat bank) with 336 iNat bank, and probe tighter crops.

Runs after eval crop extract frees the GPU. Kill bar +0.3 vs v50-like ref.

  conda activate onet && python research/crop_chase53_proxy.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT
from crop_v2_views import views_for_v2
from crop_views_proxy import (
    BANK, CTFT_Q, DEV, FROZEN_PROTO, FROZEN_Q, IMG_ROOT, KILL, LORA_PROTO, LORA_Q,
    encode_pil_batch, index_images, load_ctft, score_bank_raw,
)
from inat_maxpool_proxy import score_maxpool, top1
from v41_taxctx_weight_cv import stack_base
from v50_shiftbank_proxy import BANK_336, B2_WF, B2_WL, F336_Q, IMG_W, proto_leg, queries

HOLD_VIEWS = os.path.join(OUT, 'emb_holdout_ctft_cropviews_v2.pt')

BASE_KEYS = ['center', 'squash', 'letterbox', 'strip0', 'strip1', 'strip2']
TIGHT_KEYS = ['tight_letterbox', 'tight_squash', 'tstrip0', 'tstrip1', 'tstrip2']
OVERLAP_KEYS = [f'ostrip{i}' for i in range(5)]
ALL_KEYS = BASE_KEYS + TIGHT_KEYS + OVERLAP_KEYS


def pad_views(vs):
    if 'strip1' not in vs:
        vs['strip1'] = vs['strip0']
    if 'strip2' not in vs:
        vs['strip2'] = vs.get('strip1', vs['strip0'])
    if 'tstrip1' not in vs:
        vs['tstrip1'] = vs.get('tstrip0', vs['strip0'])
    if 'tstrip2' not in vs:
        vs['tstrip2'] = vs.get('tstrip1', vs.get('tstrip0', vs['strip0']))
    last_o = vs.get('ostrip0', vs['strip0'])
    for i in range(5):
        k = f'ostrip{i}'
        if k not in vs:
            vs[k] = last_o
        else:
            last_o = vs[k]
    return vs


def max_dbnorm(tensors, keys, bank, cand):
    raws = [score_bank_raw(tensors[k], bank, cand, topm=4) for k in keys if k in tensors]
    stacked = torch.stack(raws, dim=0)
    return dbnorm(stacked.max(0).values)


def main():
    print(f'DEV={DEV}', flush=True)
    D = FishData()
    cand = D.cand
    S0, gold, qH = stack_base(D, cand, 1.0)
    bank = torch.load(BANK, weights_only=False)['bank']
    Sct = score_maxpool(qH, bank, cand, topm=4)
    fi, ff, _ = load_emb(FROZEN_Q)
    Qf, _ = queries(D, fi, ff)
    Sf = proto_leg(Qf, FROZEN_PROTO, cand)
    li, lf, _ = load_emb(LORA_Q)
    Ql, _ = queries(D, li, lf)
    Sl = proto_leg(Ql, LORA_PROTO, cand)
    base_rest = S0 + B2_WF * Sf + B2_WL * Sl
    ref = top1(base_rest + IMG_W * Sct, gold)
    print(f'REF v50-like={ref:.4f}', flush=True)

    S336 = score_maxpool(
        queries(D, *load_emb(F336_Q)[:2])[0],
        torch.load(BANK_336, weights_only=False)['bank'],
        cand,
        topm=4,
    )
    acc336 = top1(base_rest + IMG_W * Sct + 2.0 * S336, gold)
    print(f'+2*336 bank (mean TTA Q)={acc336:.4f} ({acc336 - ref:+.3f})', flush=True)

    idx, feats, files = load_emb(CTFT_Q)
    q_files, q_gold = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx:
                q_files.append(fn)
                q_gold.append(D.cand_pos[D.ci[c]])
    q_gold = torch.tensor(q_gold)
    assert torch.equal(q_gold, gold)
    n = len(q_files)

    if os.path.exists(HOLD_VIEWS):
        packed = torch.load(HOLD_VIEWS, weights_only=False)
        assert packed['files'] == q_files
        tensors = {k: F.normalize(v.float(), dim=-1) for k, v in packed['views'].items()}
        print(f'loaded {HOLD_VIEWS}', flush=True)
    else:
        imgidx = index_images(IMG_ROOT)
        model = load_ctft()
        packs = {k: [None] * n for k in ALL_KEYS}
        miss = 0
        t0 = time.time()
        buf_t, buf_k, buf_i = [], [], []

        def flush():
            if not buf_t:
                return
            f = encode_pil_batch(model, buf_t)
            for row, key, qi in zip(f, buf_k, buf_i):
                packs[key][qi] = row
            buf_t.clear()
            buf_k.clear()
            buf_i.clear()

        for i, fn in enumerate(q_files):
            p = imgidx.get(fn)
            cached = feats[idx[fn]]
            if p is None:
                miss += 1
                for k in ALL_KEYS:
                    packs[k][i] = cached
                continue
            try:
                img = Image.open(p).convert('RGB')
                vs = pad_views(views_for_v2(img))
            except Exception:
                miss += 1
                for k in ALL_KEYS:
                    packs[k][i] = cached
                continue
            for k in ALL_KEYS:
                buf_t.append(vs[k])
                buf_k.append(k)
                buf_i.append(i)
            if len(buf_t) >= 64:
                flush()
            if i % 200 == 0:
                print(f'  encode {i}/{n} miss={miss} [{time.time() - t0:.0f}s]', flush=True)
        flush()
        tensors = {k: F.normalize(torch.stack(rows).float(), dim=-1) for k, rows in packs.items()}
        torch.save({'files': q_files, 'views': tensors, 'view_keys': ALL_KEYS}, HOLD_VIEWS)
        print(f'wrote {HOLD_VIEWS} miss={miss} [{time.time() - t0:.0f}s]', flush=True)

    recipes = {
        'max(all_no_flip)': BASE_KEYS,
        'max(+overlap5)': BASE_KEYS + OVERLAP_KEYS,
        'max(+tight)': BASE_KEYS + TIGHT_KEYS,
        'max(+overlap+tight)': ALL_KEYS,
        'max(tight_only)': ['center', 'squash'] + TIGHT_KEYS,
        'max(overlap_only)': ['center', 'squash'] + OVERLAP_KEYS,
    }
    rows = []
    best, best_name = ref, 'ref'
    scored = {}
    for name, keys in recipes.items():
        S = max_dbnorm(tensors, keys, bank, cand)
        scored[name] = S
        acc = top1(base_rest + IMG_W * S, gold)
        d = acc - ref
        rows.append({'mode': name, 'w_336': 0.0, 'proxy': acc, 'delta': d})
        print(f'{name}: {acc:.4f} ({d:+.3f})', flush=True)
        if acc > best:
            best, best_name = acc, name

    for w336 in (0.0, 1.0, 2.0, 3.0):
        for name, S in scored.items():
            acc = top1(base_rest + IMG_W * S + w336 * S336, gold)
            d = acc - ref
            rows.append({'mode': name, 'w_336': w336, 'proxy': acc, 'delta': d})
            print(f'{name} +{w336:g}*336: {acc:.4f} ({d:+.3f})', flush=True)
            tag = f'{name}|w336={w336}'
            if acc > best:
                best, best_name = acc, tag

    delta = best - ref
    out = {
        'ref': ref,
        'best': best,
        'best_name': best_name,
        'delta': delta,
        'clears_kill_03': delta >= KILL,
        'plus336_meanQ': acc336,
        'proj_real_020': 51.44259077526987 + 0.20 * delta,
        'proj_real_036': 51.44259077526987 + 0.36 * delta,
        'rows': sorted(rows, key=lambda r: -r['proxy'])[:30],
    }
    op = os.path.join(OUT, 'crop_chase53_proxy.json')
    json.dump(out, open(op, 'w'), indent=2)
    print(json.dumps({k: out[k] for k in out if k != 'rows'}, indent=2), flush=True)
    print(f'wrote {op}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
