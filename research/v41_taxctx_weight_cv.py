"""v41 full-stack proxy: sweep ctftshift@taxctx text-leg weight (class-disjoint CV).

Refs: v41 frozen B2 mean @ w=2.5, ctft maxpool @ w=4, taxctx leg @ w=1.0.
Kill: +0.3 proxy-pt vs ~46.03; zip EV ~50.57 + 0.08*delta.

  python research/v41_taxctx_weight_cv.py
"""
from __future__ import annotations

import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT  # noqa: E402
from inat_maxpool_proxy import score_maxpool, top1  # noqa: E402

REF_V41 = 46.03
IMG_W = 4.0
B2_W = 2.5
TAXABIND_W = 1.0
PROTO_B2 = os.path.join(OUT, 'inat_protos_bioclip2.pt')
QENC_B2 = os.path.join(OUT, 'emb_train_bioclip2.pt')
BANK = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')


def queries(D, idx, feats):
    q, y = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx:
                q.append(feats[idx[fn]])
                y.append(D.cand_pos[D.ci[c]])
    return torch.stack(q), torch.tensor(y)


def stack_base(D, cand, w_taxctx: float):
    TtH_c = D.TtH[cand]
    TnL_c = D.TnL[cand]
    TTX = F.normalize(
        torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(),
        dim=-1,
    )
    TTX_c = TTX[cand]
    Ttb = F.normalize(
        torch.load(os.path.join(OUT, 'text_emb_taxabind_taxctx.pt'), weights_only=False)['emb_taxctx'].float(),
        dim=-1,
    )[cand]
    legs = {}
    for tag in ['ctftshift', 'ftshift', 'fullft336_v2']:
        idx, feats, _ = load_emb(os.path.join(OUT, f'emb_train_{tag}.pt'))
        q, gold = queries(D, idx, feats)
        legs[tag] = q
    qL, gold = queries(D, D.LtI, D.LtF)
    tb_idx, tb_feat, _ = load_emb(os.path.join(OUT, 'emb_train_taxabind.pt'))
    Qtb, _ = queries(D, tb_idx, tb_feat)
    S0 = (
        dbnorm(legs['ctftshift'] @ TtH_c.t())
        + 0.5 * dbnorm(qL @ TnL_c.t())
        + 0.75 * dbnorm(legs['fullft336_v2'] @ TtH_c.t())
        + 1.0 * dbnorm(legs['ftshift'] @ TtH_c.t())
        + w_taxctx * dbnorm(legs['ctftshift'] @ TTX_c.t())
        + TAXABIND_W * dbnorm(Qtb @ Ttb.t())
    )
    return S0, gold, legs['ctftshift']


def proto_leg(Q, cand):
    P = F.normalize(torch.load(PROTO_B2, weights_only=False)['protos'].float(), dim=-1)[cand]
    has = P.norm(dim=-1) > 0.5
    S = Q @ P.t()
    S = S.masked_fill(~has.unsqueeze(0), -1e4)
    return dbnorm(S), has


def v41_score(D, cand, w_taxctx):
    S0, gold, qH = stack_base(D, cand, w_taxctx)
    bank = torch.load(BANK, weights_only=False)['bank']
    Smp = score_maxpool(qH, bank, cand, topm=4)
    n_idx, n_feat, _ = load_emb(QENC_B2)
    Qb2, _ = queries(D, n_idx, n_feat)
    S_b2, _ = proto_leg(Qb2, cand)
    S = S0 + IMG_W * Smp + B2_W * S_b2
    return S, gold


def main():
    D = FishData()
    cand = D.cand
    ref_s, gold = v41_score(D, cand, 1.0)
    ref = top1(ref_s, gold)
    print(f'v41 ref w_taxctx=1.0: {ref:.2f} (target {REF_V41:.2f})', flush=True)

    best, best_w = ref, 1.0
    rows = []
    for w in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        S, _ = v41_score(D, cand, w)
        acc = top1(S, gold)
        d = acc - ref
        rows.append({'w_taxctx': w, 'proxy': acc, 'delta_vs_w1': d})
        print(f'  w_taxctx={w:g}: {acc:.2f} ({d:+.2f})', flush=True)
        if acc > best:
            best, best_w = acc, w

    _, _, qH = stack_base(D, cand, 1.0)
    n_idx, n_feat, _ = load_emb(QENC_B2)
    Qb2, _ = queries(D, n_idx, n_feat)
    _, has_b2 = proto_leg(Qb2, cand)
    rng = random.Random(0)
    cov = [c for c in D.pseudo if has_b2[D.cand_pos[D.ci[c]]]]
    rng.shuffle(cov)
    half = len(cov) // 2
    tune_cls, val_cls = set(cov[:half]), set(cov[half:])
    cand_list = cand.tolist()
    q_cls = [D.classes[cand_list[g]] for g in gold.tolist()]
    tune_m = torch.tensor([c in tune_cls for c in q_cls])
    val_m = torch.tensor([c in val_cls for c in q_cls])

    cv_w, cv_tune = 1.0, ref
    for w in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        S, _ = v41_score(D, cand, w)
        acc = top1(S[tune_m], gold[tune_m])
        if acc > cv_tune:
            cv_tune, cv_w = acc, w
    S_base, _ = v41_score(D, cand, 1.0)
    S_cv, _ = v41_score(D, cand, cv_w)
    val_base = top1(S_base[val_m], gold[val_m])
    val_tuned = top1(S_cv[val_m], gold[val_m])

    delta = best - ref
    out = {
        'recipe': 'v41 full stack taxctx weight sweep',
        'ref_w1': ref,
        'ref_target': REF_V41,
        'best': best,
        'best_w_taxctx': best_w,
        'delta_vs_w1': delta,
        'clears_kill_03': delta >= 0.3,
        'projected_real_008': 50.57 + 0.08 * delta,
        'zip_recommend': delta >= 0.3,
        'cv': {'w_taxctx': cv_w, 'val_base': val_base, 'val_tuned': val_tuned,
               'val_delta': val_tuned - val_base},
        'sweep': rows,
    }
    op = os.path.join(OUT, 'v41_taxctx_weight_cv.json')
    json.dump(out, open(op, 'w'), indent=1)
    print(f'BEST {best:.2f} w={best_w} d={delta:+.2f} cv_val_delta={val_tuned-val_base:+.2f}', flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
