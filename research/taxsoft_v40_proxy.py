"""Soft taxonomic logit bias on v40 maxpool stack (uniform full-space argmax).

Unlike dead hard 2-stage / family-as-label (HANDOFF §4), adds a small uniform
bonus to all candidate classes sharing a family with an *anchor* from the same
forward pass (text stack or maxpool leg). No folder oracle, no candidate shrink.
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
from inat_maxpool_proxy import score_maxpool, v36_stack, top1  # noqa: E402


def family_masks(cand, classes, tax):
    cand_list = cand.tolist()
    fam_of = []
    for gi in cand_list:
        rec = tax.get(classes[gi]) or {}
        fam_of.append(rec.get('family') or '')
    fams = sorted(set(f for f in fam_of if f))
    fam_idx = {f: i for i, f in enumerate(fams)}
    M = torch.zeros(len(cand), len(fams), dtype=torch.bool)
    for j, f in enumerate(fam_of):
        if f in fam_idx:
            M[j, fam_idx[f]] = True
    return M, fam_of


def apply_bias(S, M, fam_id_per_row, beta):
    if beta == 0:
        return S
    B = torch.zeros_like(S)
    for i in range(S.shape[0]):
        fi = fam_id_per_row[i]
        if fi >= 0:
            B[i] = M[:, fi].float() * beta
    return S + B


def fam_ids_from_scores(S, M):
    """Per query: family index of argmax class, or -1."""
    pred = S.argmax(1)
    out = torch.full((S.shape[0],), -1, dtype=torch.long)
    for i in range(S.shape[0]):
        j = int(pred[i])
        cols = M[j].nonzero(as_tuple=True)[0]
        if cols.numel():
            out[i] = int(cols[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default=os.path.join(OUT, 'inat_photo_bank_ctftshift.pt'))
    ap.add_argument('--w-mp', type=float, default=4.0)
    ap.add_argument('--topm', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(OUT, 'taxsoft_v40_proxy_results.json'))
    a = ap.parse_args()

    D = FishData()
    cand = D.cand
    tax = json.load(open(os.path.join(OUT, 'taxonomy_full.json')))
    M, _ = family_masks(cand, D.classes, tax)

    bank = torch.load(a.bank, weights_only=False)['bank']
    S0, gold, qH = v36_stack(D, cand)
    Smp = score_maxpool(qH, bank, cand, topm=a.topm)
    S = S0 + a.w_mp * Smp
    ref = top1(S, gold)
    print(f'v40 ref S0+{a.w_mp}*maxpool_top{a.topm}: {ref:.2f}', flush=True)

    anchors = {
        'text_s0': fam_ids_from_scores(S0, M),
        'maxpool': fam_ids_from_scores(Smp, M),
        'full_pre': fam_ids_from_scores(S, M),
    }

    best = ref
    best_cfg = ('none', 0.0)
    grid = {}
    for aname, fids in anchors.items():
        for beta in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
            Sb = apply_bias(S, M, fids, beta)
            acc = top1(Sb, gold)
            key = f'{aname}_b{beta:g}'
            grid[key] = acc
            if acc > best:
                best, best_cfg = acc, (aname, beta)
            if beta in (0.1, 0.2, 0.5, 1.0):
                print(f'  {aname} beta={beta:g}: {acc:.2f} ({acc-ref:+.2f})', flush=True)

    # class-disjoint CV on inat-covered pseudo classes
    d = torch.load(os.path.join(OUT, 'inat_protos_ctftshift_full.pt'), weights_only=False)
    P = F.normalize(d['protos'].float(), dim=-1)[cand]
    has = P.norm(dim=-1) > 0.5
    rng = random.Random(0)
    cov_classes = [c for c in D.pseudo if has[D.cand_pos[D.ci[c]]]]
    rng.shuffle(cov_classes)
    half = len(cov_classes) // 2
    tune_cls, val_cls = set(cov_classes[:half]), set(cov_classes[half:])
    cand_list = cand.tolist()
    q_cls = [D.classes[cand_list[g]] for g in gold.tolist()]
    tune_m = torch.tensor([c in tune_cls for c in q_cls])
    val_m = torch.tensor([c in val_cls for c in q_cls])

    aname, beta = best_cfg
    fids = anchors.get(aname, anchors['text_s0'])
    if aname == 'none':
        aname, beta, fids = 'text_s0', 0.0, anchors['text_s0']

    best_tune_b, best_tune = 0.0, top1(S[tune_m], gold[tune_m])
    for beta in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0):
        for an, fid in anchors.items():
            acc = top1(apply_bias(S, M, fid, beta)[tune_m], gold[tune_m])
            if acc > best_tune:
                best_tune, best_tune_b, aname = acc, beta, an
    fids = anchors[aname]
    val_ref = top1(S[val_m], gold[val_m])
    val_tuned = top1(apply_bias(S, M, fids, best_tune_b)[val_m], gold[val_m])

    kill = 0.3
    clears = (best - ref) >= kill and (val_tuned - val_ref) >= 0.1
    res = {
        'ref_v40_maxpool': ref,
        'best': best,
        'best_anchor': best_cfg[0],
        'best_beta': best_cfg[1],
        'delta_vs_ref': best - ref,
        'clears_kill_03': clears,
        'grid': grid,
        'cv': {
            'anchor': aname,
            'beta': best_tune_b,
            'val_ref': val_ref,
            'val_tuned': val_tuned,
            'val_delta': val_tuned - val_ref,
        },
    }
    json.dump(res, open(a.out, 'w'), indent=1)
    print(f'BEST {best:.2f} ({best-ref:+.2f}) anchor={best_cfg[0]} beta={best_cfg[1]} clears_kill={clears}', flush=True)
    print(f'CV val {val_ref:.2f}->{val_tuned:.2f} ({val_tuned-val_ref:+.2f})', flush=True)
    print(f'wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
