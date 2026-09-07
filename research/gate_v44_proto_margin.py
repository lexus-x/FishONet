"""Gate-signal lift diagnostic on eval folders (NOT used for submission routing).

Adds image-proto margin / coverage features to combined gate; reports AUC only.
Uses splits/*.pkl for labels — diagnostic only, never for builder routing.

  conda activate onet && python research/gate_v44_proto_margin.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, dbnorm, load_emb  # noqa: E402

DROOT = os.path.join(os.path.dirname(OUT), 'data', 'dl')


def auc_binary(scores, y_pos):
    s = scores.detach().float().cpu()
    y = y_pos.detach().float().cpu()
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    if len(pos) * len(neg) > 5e7:
        pos = pos[torch.randperm(len(pos))[:20000]]
        neg = neg[torch.randperm(len(neg))[:20000]]
    total = 0.0
    n = 0
    for i in range(0, len(pos), 2000):
        p = pos[i:i + 2000]
        cmp = (p.unsqueeze(1) > neg.unsqueeze(0)).float()
        eq = (p.unsqueeze(1) == neg.unsqueeze(0)).float() * 0.5
        total += (cmp + eq).sum().item()
        n += p.numel() * neg.numel()
    return total / max(n, 1)


def z1(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def main():
    tf = list(pickle.load(open(f'{DROOT}/splits/test.pkl', 'rb')))
    uf = list(pickle.load(open(f'{DROOT}/splits/unseen.pkl', 'rb')))
    # filenames may be paths or basenames
    def bn(x):
        return os.path.basename(x) if isinstance(x, str) else x
    tf, uf = [bn(x) for x in tf], [bn(x) for x in uf]
    all_files = tf + uf
    y_seen = torch.tensor([1] * len(tf) + [0] * len(uf))

    # load ctftshift queries + text for deployed gate pieces
    def loadq(tag):
        te = torch.load(f'{OUT}/emb_test_{tag}.pt', weights_only=False)
        ue = torch.load(f'{OUT}/emb_unseen_{tag}.pt', weights_only=False)
        ti = {fn: i for i, fn in enumerate(te['files'])}
        ui = {fn: i for i, fn in enumerate(ue['files'])}
        q = []
        for fn in all_files:
            if fn in ti:
                q.append(te['feats'][ti[fn]])
            else:
                q.append(ue['feats'][ui[fn]])
        return F.normalize(torch.stack(q).float(), dim=-1)

    Qc = loadq('ctftshift')
    lab = json.load(open(f'{DROOT}/label_train.json'))
    classes = list(pickle.load(open(f'{DROOT}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    seen = sorted(set(lab.values()) & set(classes))
    kept = torch.tensor([ci[c] for c in seen])
    other = torch.tensor([i for i in range(len(classes)) if classes[i] not in set(seen)])

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1)
    img_seenmax = (Qc @ TtH[kept].t()).max(1).values  # weak proxy; real gate uses seen_block
    # Better: use saved prediction margins if we rebuild — approximate with text margin
    text_margin = (Qc @ TtH[kept].t()).max(1).values - (Qc @ TtH[other].t()).max(1).values
    # Use train image prototypes for img_seenmax properly
    tr = torch.load(f'{OUT}/emb_train_ctftshift.pt', weights_only=False)
    tri = {fn: i for i, fn in enumerate(tr['files'])}
    feats = F.normalize(tr['feats'].float(), dim=-1)
    from collections import defaultdict
    by = defaultdict(list)
    for fn, sp in lab.items():
        if fn in tri and sp in ci:
            by[sp].append(feats[tri[fn]])
    P = torch.zeros(len(seen), feats.shape[1])
    for i, c in enumerate(seen):
        P[i] = F.normalize(torch.stack(by[c]).mean(0), dim=0)
    img_seenmax = (Qc @ P.t()).max(1).values

    combined = z1(img_seenmax) + 2.0 * z1(text_margin)
    base_auc = auc_binary(combined, y_seen)
    print(f'base combined AUC={base_auc:.4f}', flush=True)

    # B2 denser proto margin on frozen queries
    Qb = loadq('bioclip2')
    Pm = F.normalize(torch.load(f'{OUT}/inat_tol_merged_b2_a05.pt', weights_only=False)['protos'].float(), dim=-1)
    Po = Pm[other]
    has = Po.norm(dim=-1) > 0.5
    Po = Po.clone()
    Po[~has] = 0
    sims = Qb @ Po.t()
    sims[:, ~has] = -1e4
    top2 = sims.topk(2, dim=1).values
    proto_margin = top2[:, 0] - top2[:, 1]
    proto_max = top2[:, 0]
    cov = (sims.max(1).values > -1e3).float()

    rows = []
    best = base_auc
    best_cfg = {'mode': 'base'}
    for a in (0.0, 0.5, 1.0, 1.5, 2.0):
        for b in (0.0, 0.5, 1.0, 1.5):
            for c in (0.0, 0.5, 1.0):
                if a == 0 and b == 0 and c == 0:
                    continue
                s = combined + a * z1(proto_margin) + b * z1(proto_max) + c * z1(cov)
                auc = auc_binary(s, y_seen)
                row = {'a_pm': a, 'b_pmax': b, 'c_cov': c, 'auc': auc, 'delta': auc - base_auc}
                rows.append(row)
                if auc > best:
                    best, best_cfg = auc, row
    out = {
        'base_auc': base_auc,
        'best_auc': best,
        'best_cfg': best_cfg,
        'delta_auc': best - base_auc,
        'clears_001': (best - base_auc) >= 0.001,
        'sweep_top': sorted(rows, key=lambda r: -r['auc'])[:15],
    }
    op = os.path.join(OUT, 'gate_v44_proto_margin.json')
    json.dump(out, open(op, 'w'), indent=2)
    print(json.dumps({k: out[k] for k in out if k != 'sweep_top'}, indent=2), flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
