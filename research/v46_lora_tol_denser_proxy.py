"""v46 hunt: LoRA∩ToL denser merge + dual denser weight CV on v44 stack.

  conda activate onet && python research/v46_lora_tol_denser_proxy.py
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
BANK = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')
INAT_F = os.path.join(OUT, 'inat_protos_bioclip2.pt')
INAT_L = os.path.join(OUT, 'inat_protos_bioclip2_lora_v2.pt')
TOL = os.path.join(OUT, 'tol_protos_bioclip2.pt')
MERGED_F = os.path.join(OUT, 'inat_tol_merged_b2_a05.pt')
Q_F = os.path.join(OUT, 'emb_train_bioclip2.pt')
Q_L = os.path.join(OUT, 'emb_train_bioclip2_lora_v2.pt')
REF_V44 = 47.23899960517883


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
    return dbnorm(S)


def denser_merge(a, b, alpha):
    A = F.normalize(a.float(), dim=-1)
    B = F.normalize(b.float(), dim=-1)
    ha, hb = A.norm(dim=-1) > 0.5, B.norm(dim=-1) > 0.5
    out = torch.zeros_like(A)
    both, only_a, only_b = ha & hb, ha & ~hb, hb & ~ha
    out[only_a] = A[only_a]
    out[only_b] = B[only_b]
    out[both] = F.normalize((1 - alpha) * A[both] + alpha * B[both], dim=-1)
    return out, int(both.sum())


def main():
    D = FishData()
    cand = D.cand
    S0, gold, qH = stack_base(D, cand, 1.0)
    bank = torch.load(BANK, weights_only=False)['bank']
    base = S0 + IMG_W * score_maxpool(qH, bank, cand, topm=4)

    fi, ff, _ = load_emb(Q_F)
    Qf, _ = queries(D, fi, ff)
    li, lf, _ = load_emb(Q_L)
    Ql, _ = queries(D, li, lf)

    Pf0 = torch.load(INAT_F, weights_only=False)['protos']
    Pl0 = torch.load(INAT_L, weights_only=False)['protos']
    Pt = torch.load(TOL, weights_only=False)['protos']
    Pf = torch.load(MERGED_F, weights_only=False)['protos']  # α=0.5 frozen denser

    Sf = proto_scores(Qf, Pf, cand)
    Sl0 = proto_scores(Ql, Pl0, cand)
    ref = top1(base + 2.0 * Sf + 1.0 * Sl0, gold)
    print(f'REF v44-like={ref:.4f} (expect~{REF_V44:.4f})', flush=True)

    rows = []
    best, best_cfg = ref, {'mode': 'ref'}
    merged_lora = {}

    for alpha in (0.25, 0.5, 0.75):
        Pl, nb = denser_merge(Pl0, Pt, alpha)
        merged_lora[alpha] = Pl
        Sl = proto_scores(Ql, Pl, cand)
        for wl in (0.5, 1.0, 1.5, 2.0):
            for wf in (1.5, 2.0, 2.5):
                acc = top1(base + wf * Sf + wl * Sl, gold)
                row = {
                    'mode': 'lora_denser', 'alpha': alpha, 'wf': wf, 'wl': wl,
                    'proxy': acc, 'delta': acc - ref, 'nboth': nb,
                }
                rows.append(row)
                if acc > best:
                    best, best_cfg = acc, row
        print(f'alpha={alpha} best_so_far={best:.4f} {best_cfg}', flush=True)

    # also denser both with independent alphas
    for af in (0.25, 0.5):
        for al in (0.25, 0.5, 0.75):
            Pf2, _ = denser_merge(Pf0, Pt, af)
            Pl2, nb = denser_merge(Pl0, Pt, al)
            Sf2 = proto_scores(Qf, Pf2, cand)
            Sl2 = proto_scores(Ql, Pl2, cand)
            for wf, wl in ((2.0, 1.0), (2.0, 1.5), (1.5, 1.0), (2.5, 1.0)):
                acc = top1(base + wf * Sf2 + wl * Sl2, gold)
                row = {
                    'mode': 'dual_denser', 'af': af, 'al': al, 'wf': wf, 'wl': wl,
                    'proxy': acc, 'delta': acc - ref, 'nboth_lora': nb,
                }
                rows.append(row)
                if acc > best:
                    best, best_cfg = acc, row

    # class-disjoint CV on best cfg
    y = gold.cpu()
    cls = y.unique()
    perm = cls[torch.randperm(len(cls), generator=torch.Generator().manual_seed(0))]
    n_tune = max(1, int(0.5 * len(perm)))
    val_mask = torch.isin(y, perm[n_tune:])
    ref_val = top1(base[val_mask] + 2.0 * Sf[val_mask] + 1.0 * Sl0[val_mask], y[val_mask])

    cv = None
    if best_cfg.get('mode') in ('lora_denser', 'dual_denser'):
        if best_cfg['mode'] == 'lora_denser':
            Pl = merged_lora[best_cfg['alpha']]
            Sl = proto_scores(Ql, Pl, cand)
            Sf_u, Sl_u = Sf, Sl
            wf, wl = best_cfg['wf'], best_cfg['wl']
        else:
            Pf2, _ = denser_merge(Pf0, Pt, best_cfg['af'])
            Pl2, _ = denser_merge(Pl0, Pt, best_cfg['al'])
            Sf_u = proto_scores(Qf, Pf2, cand)
            Sl_u = proto_scores(Ql, Pl2, cand)
            wf, wl = best_cfg['wf'], best_cfg['wl']
        tuned_val = top1(base[val_mask] + wf * Sf_u[val_mask] + wl * Sl_u[val_mask], y[val_mask])
        cv = {'ref_val': ref_val, 'tuned_val': tuned_val, 'delta': tuned_val - ref_val}

    # save best lora merged if useful
    if best_cfg.get('mode') == 'lora_denser':
        Pl = merged_lora[best_cfg['alpha']]
        torch.save(
            {
                'protos': Pl, 'alpha': best_cfg['alpha'],
                'sources': ['inat_protos_bioclip2_lora_v2', 'tol_protos_bioclip2'],
            },
            os.path.join(OUT, f"inat_tol_merged_b2lora_a{best_cfg['alpha']:g}.pt"),
        )

    delta = best - ref
    out = {
        'ref': ref,
        'best': best,
        'best_cfg': best_cfg,
        'delta': delta,
        'cv': cv,
        'clears_kill_03': delta >= 0.3,
        'clears_kill_1': delta >= 1.0,
        'proj_real_039': 50.77246600308426 + 0.039 * delta,
        'proj_real_36': 50.77246600308426 + 0.36 * delta,
        'sweep_top': sorted(rows, key=lambda r: -r['proxy'])[:20],
    }
    op = os.path.join(OUT, 'v46_lora_tol_denser_proxy.json')
    json.dump(out, open(op, 'w'), indent=2)
    print(json.dumps({k: out[k] for k in out if k != 'sweep_top'}, indent=2), flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
