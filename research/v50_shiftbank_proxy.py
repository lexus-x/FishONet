"""Holdout proxy: add ftshift + fullft336shift iNat photo-bank legs on the v46/v50 stack.

Unseen route today only has the ctftshift photo bank. These two extra banks use the
same shift-FT encoders already in the seen ensemble (HANDOFF 2026-08-08 gap).

Kill bar: +0.3 holdout-pt vs the current dual-B2 + ctft-maxpool ref.
  conda activate onet && python research/v50_shiftbank_proxy.py
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT
from inat_maxpool_proxy import score_maxpool, top1
from v41_taxctx_weight_cv import stack_base

IMG_W = 4.0
B2_WF = 2.5
B2_WL = 2.0
KILL = 0.3
BANK_CTFT = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')
BANK_FT = os.path.join(OUT, 'inat_photo_bank_ftshift.pt')
BANK_336 = os.path.join(OUT, 'inat_photo_bank_fullft336shift.pt')
FROZEN_PROTO = os.path.join(OUT, 'inat_tol_merged_b2_a05.pt')
FROZEN_Q = os.path.join(OUT, 'emb_train_bioclip2.pt')
LORA_PROTO = os.path.join(OUT, 'inat_tol_merged_b2lora_a0.5.pt')
LORA_Q = os.path.join(OUT, 'emb_train_bioclip2_lora_v2.pt')
FT_Q = os.path.join(OUT, 'emb_train_ftshift.pt')
F336_Q = os.path.join(OUT, 'emb_train_fullft336shift.pt')


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


def load_bank(path):
    return torch.load(path, weights_only=False)['bank']


def main():
    needed = [BANK_CTFT, BANK_FT, FROZEN_PROTO, FROZEN_Q, LORA_PROTO, LORA_Q, FT_Q]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        raise SystemExit('missing:\n  ' + '\n  '.join(missing))

    D = FishData()
    cand = D.cand
    S0, gold, qH = stack_base(D, cand, 1.0)
    Sct = score_maxpool(qH, load_bank(BANK_CTFT), cand, topm=4)

    fi, ff, _ = load_emb(FROZEN_Q)
    Qf, _ = queries(D, fi, ff)
    Sf = proto_leg(Qf, FROZEN_PROTO, cand)

    li, lf, _ = load_emb(LORA_Q)
    Ql, _ = queries(D, li, lf)
    Sl = proto_leg(Ql, LORA_PROTO, cand)

    ref_s = S0 + IMG_W * Sct + B2_WF * Sf + B2_WL * Sl
    ref = top1(ref_s, gold)
    print(f'REF v50-like (ctft bank + dual B2 denser)={ref:.4f}', flush=True)

    fti, ftf, _ = load_emb(FT_Q)
    Qft, _ = queries(D, fti, ftf)
    Sft = score_maxpool(Qft, load_bank(BANK_FT), cand, topm=4)

    S336 = None
    if os.path.exists(BANK_336) and os.path.exists(F336_Q):
        i336, f336, _ = load_emb(F336_Q)
        Q336, _ = queries(D, i336, f336)
        S336 = score_maxpool(Q336, load_bank(BANK_336), cand, topm=4)
        print('fullft336shift bank loaded', flush=True)
    else:
        print('fullft336shift bank not ready — scoring ftshift only', flush=True)

    rows = []
    best, best_cfg = ref, {'mode': 'ref'}
    weights = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

    for w_ft in weights:
        acc = top1(ref_s + w_ft * Sft, gold)
        rows.append({'w_ft': w_ft, 'w_336': 0.0, 'proxy': acc, 'delta': acc - ref})
        print(f'  +{w_ft:g}*ftshift_bank {acc:.4f} ({acc - ref:+.3f})', flush=True)
        if acc > best:
            best, best_cfg = acc, {'w_ft': w_ft, 'w_336': 0.0}

    if S336 is not None:
        for w_ft in (0.0, 2.0, 4.0):
            for w_336 in weights:
                if w_ft == 0.0 and w_336 == 0.0:
                    continue
                acc = top1(ref_s + w_ft * Sft + w_336 * S336, gold)
                rows.append({
                    'w_ft': w_ft, 'w_336': w_336, 'proxy': acc, 'delta': acc - ref,
                })
                print(
                    f'  +{w_ft:g}*ft +{w_336:g}*336 {acc:.4f} ({acc - ref:+.3f})',
                    flush=True,
                )
                if acc > best:
                    best, best_cfg = acc, {'w_ft': w_ft, 'w_336': w_336}

    delta = best - ref
    out = {
        'ref': ref,
        'best': best,
        'best_cfg': best_cfg,
        'delta': delta,
        'clears_kill_03': delta >= KILL,
        'proj_real_020': 51.44259077526987 + 0.20 * delta,
        'proj_real_036': 51.44259077526987 + 0.36 * delta,
        'had_336': S336 is not None,
        'sweep_top': sorted(rows, key=lambda r: -r['proxy'])[:20],
    }
    op = os.path.join(OUT, 'v50_shiftbank_proxy.json')
    json.dump(out, open(op, 'w'), indent=2)
    print(json.dumps({k: out[k] for k in out if k != 'sweep_top'}, indent=2), flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
