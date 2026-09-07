"""Proxy: v36 text stack + separate iNat and gap image legs (no shared proto matrix).

  S = S0 + w_inat * dbnorm(Q @ P_inat) + w_gap * dbnorm(Q @ P_gap)
Gap leg masked to classes with gap proto and NO iNat proto.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT  # noqa: E402


def top1(S, gold):
    return (S.argmax(1) == gold).float().mean().item() * 100


def build_s0(D, cand):
    TtH_c = D.TtH[cand]
    TnL_c = D.TnL[cand]
    TTX = F.normalize(
        torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(),
        dim=-1,
    )
    TTX_c = TTX[cand]

    def queries(idx, feats):
        q, y = [], []
        for c in D.pseudo:
            for fn in D.by[c]:
                if fn in idx:
                    q.append(feats[idx[fn]])
                    y.append(D.cand_pos[D.ci[c]])
        return torch.stack(q), torch.tensor(y)

    legs = {}
    for tag in ['ctftshift', 'ftshift', 'fullft336_v2']:
        idx, feats, _ = load_emb(os.path.join(OUT, f'emb_train_{tag}.pt'))
        q, gold = queries(idx, feats)
        legs[tag] = q
    qL, gold = queries(D.LtI, D.LtF)
    S0 = (
        dbnorm(legs['ctftshift'] @ TtH_c.t())
        + 0.5 * dbnorm(qL @ TnL_c.t())
        + 0.75 * dbnorm(legs['fullft336_v2'] @ TtH_c.t())
        + 1.0 * dbnorm(legs['ftshift'] @ TtH_c.t())
        + 1.0 * dbnorm(legs['ctftshift'] @ TTX_c.t())
    )
    return S0, gold, legs['ctftshift']


def img_leg(Q, cand, path, mask_fn):
    d = torch.load(path, weights_only=False)
    P = F.normalize(d['protos'].float(), dim=-1)[cand]
    has = mask_fn(P)
    S = Q @ P.t()
    S = S.masked_fill(~has.unsqueeze(0), -1e4)
    return dbnorm(S), has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inat', default=os.path.join(OUT, 'inat_protos_ctftshift_full.pt'))
    ap.add_argument('--gap', default=os.path.join(OUT, 'gap_protos_ctftshift.pt'))
    ap.add_argument('--out', default=os.path.join(OUT, 'gap_fill_proxy_results.json'))
    a = ap.parse_args()

    D = FishData()
    cand = D.cand
    S0, gold, Q = build_s0(D, cand)
    base = top1(S0, gold)
    print(f'S0 baseline: {base:.2f}', flush=True)

    Pin = F.normalize(torch.load(a.inat, weights_only=False)['protos'].float(), dim=-1)[cand]
    has_inat = Pin.norm(dim=-1) > 0.5
    S_inat, _ = img_leg(Q, cand, a.inat, lambda P: P.norm(dim=-1) > 0.5)

    has_gap_only = torch.zeros(len(cand), dtype=torch.bool)
    if os.path.exists(a.gap):
        Pg = F.normalize(torch.load(a.gap, weights_only=False)['protos'].float(), dim=-1)[cand]
        has_gap_only = (Pg.norm(dim=-1) > 0.5) & (~has_inat)
        S_gap, _ = img_leg(Q, cand, a.gap, lambda P: (P.norm(dim=-1) > 0.5) & (~has_inat))
    else:
        S_gap = torch.zeros_like(S0)

    ref_inat = top1(S0 + 3.0 * S_inat, gold)
    print(f'v37 reference (w_inat=3, w_gap=0): {ref_inat:.2f}', flush=True)

    gold_inat = has_inat[gold]
    gold_gap = has_gap_only[gold]
    print(f'gold covered: inat {int(gold_inat.sum())} gap-only {int(gold_gap.sum())}', flush=True)

    best = ref_inat
    best_cfg = (3.0, 0.0)
    grid = {}
    for w_i in [2.5, 3.0, 3.5]:
        for w_g in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            acc = top1(S0 + w_i * S_inat + w_g * S_gap, gold)
            grid[f'{w_i}_{w_g}'] = acc
            if acc > best:
                best, best_cfg = acc, (w_i, w_g)

    # class-disjoint CV on inat-covered gold
    rng = random.Random(0)
    cov_classes = [c for c in D.pseudo if has_inat[D.cand_pos[D.ci[c]]]]
    rng.shuffle(cov_classes)
    half = len(cov_classes) // 2
    tune_cls, val_cls = set(cov_classes[:half]), set(cov_classes[half:])
    cand_list = cand.tolist()
    q_cls = [D.classes[cand_list[g]] for g in gold.tolist()]
    tune_m = torch.tensor([c in tune_cls for c in q_cls])
    val_m = torch.tensor([c in val_cls for c in q_cls])

    best_wi, best_wg = 3.0, 0.0
    best_tune = top1(S0[tune_m], gold[tune_m])
    for w_i in [2.5, 3.0, 3.5]:
        for w_g in [0.0, 0.5, 1.0, 1.5, 2.0]:
            acc = top1((S0 + w_i * S_inat + w_g * S_gap)[tune_m], gold[tune_m])
            if acc > best_tune:
                best_tune, best_wi, best_wg = acc, w_i, w_g
    val_base = top1(S0[val_m], gold[val_m])
    val_tuned = top1((S0 + best_wi * S_inat + best_wg * S_gap)[val_m], gold[val_m])

    res = {
        'baseline': base,
        'v37_ref_w_inat_3': ref_inat,
        'best': best,
        'best_w_inat': best_cfg[0],
        'best_w_gap': best_cfg[1],
        'best_delta_vs_v37_ref': best - ref_inat,
        'gold_inat': int(gold_inat.sum()),
        'gold_gap_only': int(gold_gap.sum()),
        'grid': grid,
        'cv': {
            'w_inat': best_wi,
            'w_gap': best_wg,
            'val_base': val_base,
            'val_tuned': val_tuned,
            'val_delta': val_tuned - val_base,
        },
        'gap': a.gap,
    }
    json.dump(res, open(a.out, 'w'), indent=1)
    print(f'best {best:.2f} @ w_inat={best_cfg[0]} w_gap={best_cfg[1]} '
          f'(delta vs v37 ref {best-ref_inat:+.2f})', flush=True)
    print(f'CV val {val_base:.2f}->{val_tuned:.2f} wi={best_wi} wg={best_wg}', flush=True)
    print(f'wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
