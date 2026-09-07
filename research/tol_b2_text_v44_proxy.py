"""v44 stack + TreeOfLife BioCLIP-2 training-text emb leg (unseen-route).

ToL txt_emb_bioclip-2 covers ~13965/17393 classes (authorship-stripped binomial match).
Queries: frozen B2 image embs (same 768-d space).

  conda activate onet && python research/tol_b2_text_v44_proxy.py
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
WF, WL = 2.0, 1.0  # v44 denser defaults
BANK = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')
MERGED = os.path.join(OUT, 'inat_tol_merged_b2_a05.pt')
LORA_PROTO = os.path.join(OUT, 'inat_protos_bioclip2_lora_v2.pt')
FROZEN_Q = os.path.join(OUT, 'emb_train_bioclip2.pt')
LORA_Q = os.path.join(OUT, 'emb_train_bioclip2_lora_v2.pt')
TOL_TXT = os.path.join(OUT, 'tol_text_emb_b2.pt')
OUR_B2_TXT = os.path.join(OUT, 'text_emb_b2_taxctx.pt')


def queries(D, idx, feats):
    q, y = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx:
                q.append(feats[idx[fn]])
                y.append(D.cand_pos[D.ci[c]])
    return torch.stack(q), torch.tensor(y)


def proto_scores(Q, P_all, cand):
    P = F.normalize(P_all.float(), dim=-1)[cand]
    has = P.norm(dim=-1) > 0.5
    S = Q @ P.t()
    S = S.masked_fill(~has.unsqueeze(0), -1e4)
    return dbnorm(S), int(has.sum())


def main():
    D = FishData()
    cand = D.cand
    S0, gold, qH = stack_base(D, cand, 1.0)
    bank = torch.load(BANK, weights_only=False)['bank']
    base = S0 + IMG_W * score_maxpool(qH, bank, cand, topm=4)

    fi, ff, _ = load_emb(FROZEN_Q)
    Qf, _ = queries(D, fi, ff)
    Pm = torch.load(MERGED, weights_only=False)['protos']
    Sm, _ = proto_scores(Qf, Pm, cand)
    li, lf, _ = load_emb(LORA_Q)
    Ql, _ = queries(D, li, lf)
    Pl = torch.load(LORA_PROTO, weights_only=False)['protos']
    Sl, _ = proto_scores(Ql, Pl, cand)

    ref = top1(base + WF * Sm + WL * Sl, gold)
    print(f'REF v44-like={ref:.4f}', flush=True)

    Tt = torch.load(TOL_TXT, weights_only=False)['emb_taxon']
    St, ncov = proto_scores(Qf, Tt, cand)
    print(f'ToL B2 text cand cov={ncov}/{len(cand)}', flush=True)
    alone = top1(base + 2.0 * St, gold)
    print(f'ToL text alone @2={alone:.4f}', flush=True)

    rows = []
    best, best_cfg = ref, {'mode': 'ref'}
    for wt in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        acc = top1(base + WF * Sm + WL * Sl + wt * St, gold)
        row = {'wt': wt, 'proxy': acc, 'delta': acc - ref}
        rows.append(row)
        print(row, flush=True)
        if acc > best:
            best, best_cfg = acc, row

    # also try replacing/reducing image proto weight in favor of text
    for wf in (1.0, 1.5, 2.0):
        for wt in (1.0, 2.0, 3.0):
            acc = top1(base + wf * Sm + WL * Sl + wt * St, gold)
            row = {'wf': wf, 'wt': wt, 'proxy': acc, 'delta': acc - ref}
            rows.append(row)
            if acc > best:
                best, best_cfg = acc, row

    # our own b2 taxctx text if loadable
    if os.path.exists(OUR_B2_TXT):
        To = torch.load(OUR_B2_TXT, weights_only=False)
        key = 'emb_taxctx' if 'emb_taxctx' in To else 'emb_taxon'
        So, n2 = proto_scores(Qf, To[key], cand)
        print(f'our B2 taxctx cand cov={n2}', flush=True)
        for wt in (0.5, 1.0, 2.0):
            acc = top1(base + WF * Sm + WL * Sl + wt * So, gold)
            row = {'mode': 'our_b2_txt', 'wt': wt, 'proxy': acc, 'delta': acc - ref}
            rows.append(row)
            print(row, flush=True)
            if acc > best:
                best, best_cfg = acc, row

    delta = best - ref
    out = {
        'ref_v44': ref,
        'tol_text_alone_w2': alone,
        'tol_text_cand_cov': ncov,
        'best': best,
        'best_cfg': best_cfg,
        'delta': delta,
        'clears_kill_03': delta >= 0.3,
        'clears_kill_1': delta >= 1.0,
        'proj_real_039': 50.77246600308426 + 0.039 * delta,
        'proj_real_36': 50.77246600308426 + 0.36 * delta,
        'sweep_top': sorted(rows, key=lambda r: -r['proxy'])[:15],
    }
    op = os.path.join(OUT, 'tol_b2_text_v44_proxy.json')
    json.dump(out, open(op, 'w'), indent=2)
    print(json.dumps({k: out[k] for k in out if k != 'sweep_top'}, indent=2), flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
