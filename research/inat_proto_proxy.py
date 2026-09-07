"""Proxy A/B: does adding iNaturalist image prototypes lift the v36 unseen stack?

Baseline = v36 unseen text stack MINUS TaxaBind leg (saturation_check: 31.45 on the
pseudo-unseen hard-sim proxy). We add an image-prototype leg (query @ per-class iNat
prototype, dbnorm, masked for uncovered classes) and measure top-1 over the full cand
space. Reports: coverage, image-alone, blends, covered-gold subset, and a
class-disjoint CV of the best blend weight.

  python research/inat_proto_proxy.py --protos outputs/inat_protos_h.pt --qenc h
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT  # noqa: E402


def top1(S, gold):
    return (S.argmax(1) == gold).float().mean().item() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protos', default=os.path.join(OUT, 'inat_protos_h.pt'))
    ap.add_argument('--protos2', default='', help='optional 2nd proto set (e.g. ctft)')
    ap.add_argument('--qenc', choices=['h', 'ctftshift'], default='h',
                    help='query encoder matching --protos')
    ap.add_argument('--qenc2', choices=['h', 'ctftshift'], default='ctftshift')
    a = ap.parse_args()

    t0 = time.time()
    D = FishData()
    cand = D.cand
    cand_list = cand.tolist()
    print(f'[{time.time()-t0:.0f}s] cand={len(cand)} pseudo={len(D.pseudo)}', flush=True)

    # ---- build v36 unseen text stack (minus TaxaBind), exactly as saturation_check ----
    def queries(idx, feats):
        q, y = [], []
        for c in D.pseudo:
            for fn in D.by[c]:
                if fn in idx:
                    q.append(feats[idx[fn]])
                    y.append(D.cand_pos[D.ci[c]])
        return torch.stack(q), torch.tensor(y)

    TtH_c = D.TtH[cand]
    TnL_c = D.TnL[cand]
    TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'),
                                 weights_only=False)['emb_taxctx'].float(), dim=-1)
    TTX_c = TTX[cand]

    legs = {}
    for tag in ['ctftshift', 'ftshift', 'fullft336_v2']:
        idx, feats, _ = load_emb(os.path.join(OUT, f'emb_train_{tag}.pt'))
        q, gold = queries(idx, feats)
        legs[tag] = q
    qL, gold = queries(D.LtI, D.LtF)
    hIdx, hFeat, _ = load_emb(os.path.join(OUT, 'emb_train_h.pt'))
    qH, _ = queries(hIdx, hFeat)

    S0 = (dbnorm(legs['ctftshift'] @ TtH_c.t()) + 0.5 * dbnorm(qL @ TnL_c.t())
          + 0.75 * dbnorm(legs['fullft336_v2'] @ TtH_c.t())
          + 1.0 * dbnorm(legs['ftshift'] @ TtH_c.t())
          + 1.0 * dbnorm(legs['ctftshift'] @ TTX_c.t()))
    base = top1(S0, gold)
    print(f'\nv36 text stack (minus TB) baseline top-1 = {base:.2f}', flush=True)

    qmap = {'h': qH, 'ctftshift': legs['ctftshift']}

    def img_leg(protos_path, qenc):
        d = torch.load(protos_path, weights_only=False)
        P = F.normalize(d['protos'].float(), dim=-1)
        Pc = P[cand]
        has = (Pc.norm(dim=-1) > 0.5)
        Q = qmap[qenc]
        Simg = Q @ Pc.t()
        Simg = Simg.masked_fill(~has.unsqueeze(0), -1e4)
        return dbnorm(Simg), has, d

    Simg, has, d = img_leg(a.protos, a.qenc)
    cov_cand = int(has.sum())
    # gold coverage
    gold_cand_idx = torch.tensor([D.cand_pos[D.ci[c]] for c in D.pseudo])
    # each query's gold covered?
    gold_covered = has[gold]
    print(f'iNat protos({a.qenc}): cand covered {cov_cand}/{len(cand)} '
          f'({100*cov_cand/len(cand):.1f}%); gold-query covered '
          f'{int(gold_covered.sum())}/{len(gold)} ({100*gold_covered.float().mean():.1f}%)', flush=True)

    a_img = top1(Simg, gold)
    print(f'image protos alone top-1 = {a_img:.2f}', flush=True)

    res = {'baseline': base, 'img_alone': a_img, 'cand_covered': cov_cand,
           'cand_total': len(cand), 'gold_covered': int(gold_covered.sum()),
           'gold_total': len(gold), 'n_queries': len(gold), 'blends': {},
           'protos': a.protos, 'qenc': a.qenc}

    print('--- blends: S0 + w*img ---', flush=True)
    best = base; bestw = 0.0
    for w in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        acc = top1(S0 + w * Simg, gold)
        res['blends'][str(w)] = acc
        if acc > best:
            best, bestw = acc, w
        print(f'  S0 + {w:g}*img: {acc:.2f} ({acc-base:+.2f})', flush=True)

    # optional 2nd proto set fused too
    if a.protos2 and os.path.exists(a.protos2):
        Simg2, has2, _ = img_leg(a.protos2, a.qenc2)
        print('--- blends: S0 + w*(img+img2) ---', flush=True)
        for w in [0.5, 1.0, 1.5, 2.0]:
            acc = top1(S0 + w * (Simg + Simg2), gold)
            res['blends'][f'dual_{w}'] = acc
            if acc > best:
                best, bestw = acc, w
            print(f'  S0 + {w:g}*(img+img2): {acc:.2f} ({acc-base:+.2f})', flush=True)

    # ---- covered-gold subset: does img help where a gold proto exists? ----
    if gold_covered.any():
        sel = gold_covered
        b_sub = top1(S0[sel], gold[sel])
        for w in [0.5, 1.0, 2.0]:
            acc = top1((S0 + w * Simg)[sel], gold[sel])
            res.setdefault('covered_subset', {})[str(w)] = acc
            print(f'  [covered-gold n={int(sel.sum())}] S0+{w:g}*img: {acc:.2f} (base {b_sub:.2f}, {acc-b_sub:+.2f})', flush=True)
        res['covered_subset_base'] = b_sub

    # ---- class-disjoint CV of best blend weight ----
    # split covered gold classes into tune/val; fit w on tune, report val delta.
    import random
    rng = random.Random(0)
    cov_classes = [c for c in D.pseudo if has[D.cand_pos[D.ci[c]]]]
    rng.shuffle(cov_classes)
    half = len(cov_classes) // 2
    tune_cls, val_cls = set(cov_classes[:half]), set(cov_classes[half:])
    q_cls = [D.classes[cand_list[g]] for g in gold.tolist()]
    tune_m = torch.tensor([c in tune_cls for c in q_cls])
    val_m = torch.tensor([c in val_cls for c in q_cls])
    best_w_tune, best_a_tune = 0.0, top1(S0[tune_m], gold[tune_m])
    for w in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        acc = top1((S0 + w * Simg)[tune_m], gold[tune_m])
        if acc > best_a_tune:
            best_a_tune, best_w_tune = acc, w
    val_base = top1(S0[val_m], gold[val_m])
    val_tuned = top1((S0 + best_w_tune * Simg)[val_m], gold[val_m])
    res['cv'] = {'best_w_tune': best_w_tune, 'val_base': val_base, 'val_tuned': val_tuned,
                 'val_delta': val_tuned - val_base, 'n_val': int(val_m.sum())}
    print(f'\nCV (class-disjoint, covered-gold): best_w={best_w_tune}, '
          f'val {val_base:.2f} -> {val_tuned:.2f} ({val_tuned-val_base:+.2f}, n_val={int(val_m.sum())})',
          flush=True)

    res['best_blend'] = best
    res['best_delta'] = best - base
    res['best_w'] = bestw
    out = os.path.join(OUT, 'inat_proto_proxy_results.json')
    json.dump(res, open(out, 'w'), indent=1)
    print(f'\nbest blend {best:.2f} ({best-base:+.2f}) @ w={bestw}; wrote {out}', flush=True)


if __name__ == '__main__':
    main()
