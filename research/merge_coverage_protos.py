"""Merge iNat full protos with GBIF/Wikimedia coverage protos (weighted mean, then L2)."""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'research'))
from common import FishData, OUT  # noqa: E402


def load_pack(path):
    d = torch.load(path, weights_only=False)
    P = d['protos'].float()
    cnt = d.get('cnt')
    if cnt is None:
        cnt = (P.norm(dim=-1) > 0.5).float()
    return P, cnt.float(), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=os.path.join(OUT, 'inat_protos_ctftshift_full.pt'))
    ap.add_argument('--extra', default=os.path.join(OUT, 'coverage_protos_ctftshift.pt'))
    ap.add_argument('--out', default=os.path.join(OUT, 'protos_ctftshift_merged.pt'))
    ap.add_argument('--prefer', choices=['union', 'extra_if_base_empty'], default='union')
    a = ap.parse_args()

    D = FishData()
    C = len(D.classes)
    Pb, cb, meta_b = load_pack(a.base)
    Pe, ce, meta_e = load_pack(a.extra)
    assert Pb.shape[0] == C and Pe.shape[0] == C

    acc = torch.zeros(C, Pb.shape[1])
    cnt = torch.zeros(C)
    for i in range(C):
        w0, w1 = float(cb[i]), float(ce[i])
        if w0 <= 0 and w1 <= 0:
            continue
        if a.prefer == 'extra_if_base_empty' and w0 > 0:
            v = F.normalize(Pb[i], dim=-1)
            acc[i] = v * w0
            cnt[i] = w0
        elif w0 > 0 and w1 > 0:
            v = (F.normalize(Pb[i], dim=-1) * w0 + F.normalize(Pe[i], dim=-1) * w1) / (w0 + w1)
            acc[i] = v * (w0 + w1)
            cnt[i] = w0 + w1
        elif w1 > 0:
            acc[i] = F.normalize(Pe[i], dim=-1) * w1
            cnt[i] = w1
        else:
            acc[i] = F.normalize(Pb[i], dim=-1) * w0
            cnt[i] = w0

    protos = torch.zeros_like(acc)
    for i in range(C):
        if cnt[i] > 0:
            protos[i] = F.normalize(acc[i], dim=-1)

    covered = [D.classes[i] for i in range(C) if cnt[i] > 0]
    out_d = {
        'protos': protos,
        'cnt': cnt,
        'covered': covered,
        'enc': meta_b.get('enc', 'ctft'),
        'ckpt': meta_b.get('ckpt'),
        'squash_tta': meta_b.get('squash_tta'),
        'model': meta_b.get('model'),
        'classes': D.classes,
        'merged_from': [a.base, a.extra],
    }
    torch.save(out_d, a.out)
    n = int((cnt > 0).sum())
    n_new = int(((cb <= 0) & (ce > 0)).sum())
    print(f'wrote {a.out}: covered {n}/{C} (+{n_new} from coverage only)', flush=True)


if __name__ == '__main__':
    main()
