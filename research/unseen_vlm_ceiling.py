"""VLM rerank ceiling on the pseudo-unseen proxy (plan step 5).

Does NOT run a VLM. Measures the in-shortlist oracle: if a perfect reranker
always picked gold whenever gold is in the top-K of the current text scores,
how much would top-1 rise? If that ceiling is < ~+2pt, kill VLM without GPU.

Uses frozen-H @ taxon (same comparator as unseen_text) and optionally the
deployed ctftbig@taxon score if emb_train_ctftbig.pt exists.

  python research/unseen_vlm_ceiling.py
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, OUT  # noqa: E402


def top1(S, gold):
    return (S.argmax(1) == gold).float().mean().item() * 100


def oracle_at_k(S, gold, ks):
    """Oracle accuracy if gold is taken whenever it appears in top-k."""
    out = {}
    # ranks: lower is better
    # S: [N, C]
    order = S.argsort(dim=1, descending=True)
    for k in ks:
        hit = (order[:, :k] == gold.unsqueeze(1)).any(dim=1).float()
        out[k] = hit.mean().item() * 100
    return out


def main():
    D = FishData()
    files, gold = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            files.append(fn)
            gold.append(D.cand_pos[D.ci[c]])
    gold = torch.tensor(gold)

    d = torch.load(os.path.join(OUT, 'emb_train_h.pt'), weights_only=False)
    idx = {fn: i for i, fn in enumerate(d['files'])}
    feats = F.normalize(d['feats'].float(), dim=-1)
    keep = [i for i, fn in enumerate(files) if fn in idx]
    Qh = torch.stack([feats[idx[files[i]]] for i in keep])
    gold_h = gold[keep]
    Tt = D.TtH[D.cand]
    Sh = dbnorm(Qh @ Tt.t())
    base_h = top1(Sh, gold_h)
    ks = [5, 10, 20, 50, 100]
    or_h = oracle_at_k(Sh, gold_h, ks)
    print(f'frozen-H @ taxon top-1: {base_h:.2f}  n={len(keep)}', flush=True)
    for k in ks:
        print(f'  oracle@top{k:3d}: {or_h[k]:.2f}  (ceiling delta {or_h[k]-base_h:+.2f})', flush=True)

    results = {
        'frozen_h_taxon': {'top1': base_h, 'oracle': {str(k): or_h[k] for k in ks},
                           'n': len(keep)},
    }

    # deployed-ish: ctftbig @ taxon if available
    ctft_path = os.path.join(OUT, 'emb_train_ctftbig.pt')
    if os.path.exists(ctft_path):
        d = torch.load(ctft_path, weights_only=False)
        idx = {fn: i for i, fn in enumerate(d['files'])}
        feats = F.normalize(d['feats'].float(), dim=-1)
        keep2 = [i for i, fn in enumerate(files) if fn in idx]
        Qc = torch.stack([feats[idx[files[i]]] for i in keep2])
        gold_c = gold[keep2]
        Sc = dbnorm(Qc @ Tt.t())
        # add 0.5 L@name if possible
        qL, gL = [], []
        for i in keep2:
            fn = files[i]
            if fn in D.LtI:
                qL.append(D.LtF[D.LtI[fn]])
                gL.append(int(gold[i]))
        if qL:
            qL = torch.stack(qL)
            gL = torch.tensor(gL)
            # align: keep2 indices that are in LtI — rebuild
            keep3, gold3 = [], []
            for i in keep2:
                fn = files[i]
                if fn in D.LtI:
                    keep3.append(fn)
                    gold3.append(int(gold[i]))
            Qc2 = torch.stack([feats[idx[fn]] for fn in keep3])
            qL = torch.stack([D.LtF[D.LtI[fn]] for fn in keep3])
            gold3 = torch.tensor(gold3)
            Sdep = dbnorm(Qc2 @ Tt.t()) + 0.5 * dbnorm(qL @ D.TnL[D.cand].t())
            base_d = top1(Sdep, gold3)
            or_d = oracle_at_k(Sdep, gold3, ks)
            print(f'\ndeployed-ish ctft@taxon+0.5L top-1: {base_d:.2f}  n={len(keep3)}', flush=True)
            for k in ks:
                print(f'  oracle@top{k:3d}: {or_d[k]:.2f}  (ceiling delta {or_d[k]-base_d:+.2f})',
                      flush=True)
            results['deployed_ctft_taxon_L'] = {
                'top1': base_d, 'oracle': {str(k): or_d[k] for k in ks}, 'n': len(keep3),
            }
    else:
        print('emb_train_ctftbig.pt missing — skip deployed leg', flush=True)

    # kill if even top-20 oracle on best leg gives < +2pt
    leg = results.get('deployed_ctft_taxon_L') or results['frozen_h_taxon']
    base = leg['top1']
    ceil20 = leg['oracle']['20']
    delta20 = ceil20 - base
    promote = delta20 >= 2.0
    results['verdict'] = {
        'oracle20_delta': delta20,
        'promote_vlm_rerank': promote,
        'reason': (
            f'top-20 oracle ceiling +{delta20:.2f}pt >= 2.0 — VLM rerank worth trying'
            if promote else
            f'top-20 oracle ceiling +{delta20:.2f}pt < 2.0 — kill VLM without GPU burn'
        ),
    }
    print('\nVERDICT:', json.dumps(results['verdict'], indent=1), flush=True)
    p = os.path.join(OUT, 'unseen_vlm_ceiling_results.json')
    json.dump(results, open(p, 'w'), indent=1)
    print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
