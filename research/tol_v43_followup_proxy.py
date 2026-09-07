"""Follow-ups after raw ToL gap-fill death: denser merge, genus backoff, H-proto on v43.

  conda activate onet && python research/tol_v43_followup_proxy.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

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
H_PROTO = os.path.join(OUT, 'inat_protos_h_full.pt')
H_Q = os.path.join(OUT, 'emb_train_h.pt')
H_Q_ALT = os.path.join(OUT, 'emb_train.pt')  # L/H shared sometimes


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
    return dbnorm(S), has


def denser_merge(inat, tol, alpha=0.5):
    """Average iNat+ToL where both exist; keep single-source otherwise."""
    ina = F.normalize(inat.float(), dim=-1)
    to = F.normalize(tol.float(), dim=-1)
    hi = ina.norm(dim=-1) > 0.5
    ht = to.norm(dim=-1) > 0.5
    out = torch.zeros_like(ina)
    both = hi & ht
    only_i = hi & ~ht
    only_t = ht & ~hi
    out[only_i] = ina[only_i]
    out[only_t] = to[only_t]
    out[both] = F.normalize((1 - alpha) * ina[both] + alpha * to[both], dim=-1)
    return out, int(both.sum()), int((out.norm(dim=-1) > 0.5).sum())


def genus_backoff(species_protos, classes):
    """Fill missing species with mean of covered congeners."""
    P = F.normalize(species_protos.float(), dim=-1).clone()
    by_genus = defaultdict(list)
    for i, c in enumerate(classes):
        g = c.split()[0] if c else ''
        if g:
            by_genus[g].append(i)
    nfill = 0
    for g, idxs in by_genus.items():
        covered = [i for i in idxs if P[i].norm() > 0.5]
        if not covered:
            continue
        mean = F.normalize(P[covered].mean(0), dim=0)
        for i in idxs:
            if P[i].norm() <= 0.5:
                P[i] = mean
                nfill += 1
    return P, nfill


def main():
    D = FishData()
    cand = D.cand
    classes = D.classes
    S0, gold, qH = stack_base(D, cand, 1.0)
    bank = torch.load(BANK, weights_only=False)['bank']
    Smp = score_maxpool(qH, bank, cand, topm=4)
    base = S0 + IMG_W * Smp

    fi, ff, _ = load_emb(FROZEN_Q)
    Qf, _ = queries(D, fi, ff)
    Pf = torch.load(FROZEN_PROTO, weights_only=False)['protos']
    Sf, _ = proto_scores(Qf, Pf, cand)
    li, lf, _ = load_emb(LORA_Q)
    Ql, _ = queries(D, li, lf)
    Pl = torch.load(LORA_PROTO, weights_only=False)['protos']
    Sl, _ = proto_scores(Ql, Pl, cand)
    ref = top1(base + WF * Sf + WL * Sl, gold)
    print(f'REF={ref:.4f}', flush=True)

    Pt = torch.load(TOL_PROTO, weights_only=False)['protos']
    rows = []
    best, best_cfg = ref, {'mode': 'ref'}

    # denser merge alphas
    for alpha in (0.25, 0.5, 0.75):
        Pm, nboth, ncov = denser_merge(Pf, Pt, alpha=alpha)
        Sm, _ = proto_scores(Qf, Pm, cand)
        for wf in (0.5, 1.0, 1.5, 2.0, 2.5):
            acc = top1(base + wf * Sm + WL * Sl, gold)
            row = {
                'mode': 'denser_merge', 'alpha': alpha, 'wf': wf,
                'proxy': acc, 'delta': acc - ref, 'nboth': nboth, 'ncov': ncov,
            }
            rows.append(row)
            print(row, flush=True)
            if acc > best:
                best, best_cfg = acc, row

    # genus backoff on frozen / merged
    Pg, nfill = genus_backoff(Pf, classes)
    Sg, _ = proto_scores(Qf, Pg, cand)
    for wf in (0.5, 1.0, 1.5, 2.0, 2.5):
        acc = top1(base + wf * Sg + WL * Sl, gold)
        row = {
            'mode': 'genus_backoff_inat', 'wf': wf, 'nfill': nfill,
            'proxy': acc, 'delta': acc - ref,
        }
        rows.append(row)
        print(row, flush=True)
        if acc > best:
            best, best_cfg = acc, row

    Pm, _, _ = denser_merge(Pf, Pt, alpha=0.5)
    Pmg, nfill2 = genus_backoff(Pm, classes)
    Smg, _ = proto_scores(Qf, Pmg, cand)
    for wf in (0.5, 1.0, 1.5, 2.0):
        acc = top1(base + wf * Smg + WL * Sl, gold)
        row = {
            'mode': 'denser+genus', 'wf': wf, 'nfill': nfill2,
            'proxy': acc, 'delta': acc - ref,
        }
        rows.append(row)
        print(row, flush=True)
        if acc > best:
            best, best_cfg = acc, row

    # frozen-H iNat proto leg on v43 (orthogonal dim)
    hq = H_Q if os.path.exists(H_Q) else H_Q_ALT
    if os.path.exists(H_PROTO) and os.path.exists(hq):
        hi, hf, _ = load_emb(hq)
        Qh, _ = queries(D, hi, hf)
        Ph = torch.load(H_PROTO, weights_only=False)['protos']
        # dim check
        if Qh.shape[1] == Ph.shape[1]:
            Sh, _ = proto_scores(Qh, Ph, cand)
            for wh in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
                acc = top1(base + WF * Sf + WL * Sl + wh * Sh, gold)
                row = {'mode': 'h_add', 'wh': wh, 'proxy': acc, 'delta': acc - ref}
                rows.append(row)
                print(row, flush=True)
                if acc > best:
                    best, best_cfg = acc, row
        else:
            print(f'H dim mismatch Q={Qh.shape[1]} P={Ph.shape[1]}', flush=True)
    else:
        print(f'skip H: proto={os.path.exists(H_PROTO)} q={os.path.exists(hq)}', flush=True)

    delta = best - ref
    out = {
        'ref': ref,
        'best': best,
        'best_cfg': best_cfg,
        'delta': delta,
        'clears_kill_03': delta >= 0.3,
        'proj_real_036': 50.75564278704613 + 0.36 * delta,
        'sweep_top': sorted(rows, key=lambda r: -r['proxy'])[:20],
    }
    op = os.path.join(OUT, 'tol_v43_followup_proxy.json')
    json.dump(out, open(op, 'w'), indent=2)
    print(json.dumps({k: out[k] for k in out if k != 'sweep_top'}, indent=2), flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
