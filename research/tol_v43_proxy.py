"""Proxy: v43 dual-B2 stack ± TreeOfLife BioCLIP-2 mean protos (gap-fill or additive).

  conda activate onet && python research/tol_v43_proxy.py
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
TOL_PROTO = os.path.join(OUT, 'tol_protos_bioclip2.pt')
REF_V43 = 46.80759310722351


def queries(D, idx, feats):
    q, y = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx:
                q.append(feats[idx[fn]])
                y.append(D.cand_pos[D.ci[c]])
    return torch.stack(q), torch.tensor(y)


def proto_leg(Q, P_all, cand):
    P = F.normalize(P_all.float(), dim=-1)[cand]
    has = P.norm(dim=-1) > 0.5
    S = Q @ P.t()
    S = S.masked_fill(~has.unsqueeze(0), -1e4)
    return dbnorm(S), has


def merge_gapfill(inat, tol):
    """Keep iNat where present; fill zeros from ToL."""
    out = inat.clone()
    miss = out.norm(dim=-1) < 0.5
    fill = (tol.norm(dim=-1) > 0.5) & miss
    out[fill] = F.normalize(tol[fill].float(), dim=-1)
    return out, int(fill.sum()), int((out.norm(dim=-1) > 0.5).sum())


def main():
    for p in (FROZEN_PROTO, FROZEN_Q, LORA_PROTO, LORA_Q, BANK, TOL_PROTO):
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
    Pf = torch.load(FROZEN_PROTO, weights_only=False)['protos']
    Sf, _ = proto_leg(Qf, Pf, cand)

    li, lf, _ = load_emb(LORA_Q)
    Ql, _ = queries(D, li, lf)
    Pl = torch.load(LORA_PROTO, weights_only=False)['protos']
    Sl, _ = proto_leg(Ql, Pl, cand)

    ref = top1(base + WF * Sf + WL * Sl, gold)
    print(f'REF v43-like={ref:.4f} (expected~{REF_V43:.4f})', flush=True)

    Pt = torch.load(TOL_PROTO, weights_only=False)['protos']
    St, has_t = proto_leg(Qf, Pt, cand)  # ToL emb space = frozen B2
    alone = top1(base + 2.5 * St, gold)

    merged, nfill, ncov = merge_gapfill(Pf, Pt)
    Sm, _ = proto_leg(Qf, merged, cand)
    gap = top1(base + WF * Sm + WL * Sl, gold)

    rows = []
    best, best_cfg = ref, {'mode': 'ref'}
    for wt in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5):
        for mode, Sextra in [('tol_add', St), ('gapfill_fused', Sm - Sf)]:
            # gapfill_fused: use merged in place of frozen via replacing Sf weight
            if mode == 'gapfill_fused':
                acc = top1(base + WF * Sm + WL * Sl + wt * St, gold) if wt > 0 else gap
                if wt == 0:
                    acc = gap
            else:
                acc = top1(base + WF * Sf + WL * Sl + wt * St, gold)
            d = acc - ref
            rows.append({'mode': mode, 'wt': wt, 'proxy': acc, 'delta': d})
            if acc > best:
                best, best_cfg = acc, {'mode': mode, 'wt': wt}

    # cleaner sweeps
    rows2 = []
    for wt in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0):
        acc = top1(base + WF * Sf + WL * Sl + wt * St, gold)
        rows2.append({'mode': 'add_tol', 'wt': wt, 'proxy': acc, 'delta': acc - ref})
        if acc > best:
            best, best_cfg = acc, {'mode': 'add_tol', 'wt': wt}
    for wt in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        acc = top1(base + WF * Sm + WL * Sl + wt * 0, gold) if False else top1(
            base + WF * Sm + WL * Sl, gold
        )
        # only one gapfill-without-extra needed; already have gap
        break
    for wf in (0.5, 1.0, 1.5, 2.0, 2.5):
        acc = top1(base + wf * Sm + WL * Sl, gold)
        rows2.append({'mode': 'gapfill_frozen_w', 'wf': wf, 'proxy': acc, 'delta': acc - ref})
        if acc > best:
            best, best_cfg = acc, {'mode': 'gapfill_frozen_w', 'wf': wf}

    delta = best - ref
    out = {
        'ref_v43': ref,
        'tol_alone_w25': alone,
        'gapfill_proxy': gap,
        'gapfill_nfill': nfill,
        'gapfill_ncov_frozen_slot': ncov,
        'tol_cand_cov': int(has_t.sum()),
        'best_proxy': best,
        'best_cfg': best_cfg,
        'delta_vs_v43': delta,
        'clears_kill_03': delta >= 0.3,
        'proj_real_036': 50.75564278704613 + 0.36 * delta,
        'proj_real_008': 50.75564278704613 + 0.08 * delta,
        'sweep': rows2,
    }
    op = os.path.join(OUT, 'tol_v43_proxy.json')
    json.dump(out, open(op, 'w'), indent=2)
    print(json.dumps({k: out[k] for k in out if k != 'sweep'}, indent=2), flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
