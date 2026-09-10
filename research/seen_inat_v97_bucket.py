"""v97: does the iNat seen-evidence leg (v95, globally DEAD) help specifically for
DATA-STARVED classes (n_train<=2, which is 58.3% of all 4,636 seen classes)?

v95 measured a global monotonic decline. But for a class with 1-2 training images,
proto==cmax (identical, per HANDOFF v83 notes) -- the base signal there is a single
noisy exemplar. iNat may still be worse than even 1 real photo (domain gap), or it
may only look bad in the GLOBAL average because it actively hurts the well-trained
majority while helping the starved minority. This buckets holdout accuracy by
n_train to tell those two stories apart before declaring the whole direction closed.

  conda activate onet && python research/seen_inat_v97_bucket.py
"""
from __future__ import annotations

import json
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, MEMBERS, dev, holdout_split, load_train_embs, zc
from seen_inat_v95 import BANKS, inat_leg, member_scores

GRID_WI = [0.0, 0.25, 0.5, 1.0]
BASE_W = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
BUCKETS = [(0, 2), (3, 5), (6, 1_000_000)]


def main():
    torch.set_num_threads(8)
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby, val_seen = sp['trby'], [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    ci = {c: i for i, c in enumerate(classes)}
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    n_train = torch.tensor([len(trby[c]) for c in seen], dtype=torch.float32)
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zbase, Zleg = {}, {}
    for t, _, _ in MEMBERS:
        if t not in BASE_W:
            continue
        idx, feats, _ = train[t]
        P = torch.zeros(S, feats.shape[1])
        cnt = torch.zeros(S)
        TFl, TLl = [], []
        for c in seen:
            for fn in trby[c]:
                if fn not in idx:
                    continue
                f = feats[idx[fn]]
                P[s2i[c]] += f
                cnt[s2i[c]] += 1
                TFl.append(f)
                TLl.append(s2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
        TF = torch.stack(TFl).to(dev)
        TL = torch.tensor(TLl).to(dev)
        qF = torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev)
        base = member_scores(qF, P, TF, TL, Tseen, S)
        Zbase[t] = zc(base)

        bank = torch.load(BANKS[t], weights_only=False)['bank']
        IFl, ILl = [], []
        IP = torch.zeros(S, feats.shape[1])
        covered = torch.zeros(S, dtype=torch.bool)
        for gci, mat in bank.items():
            c = classes[gci]
            if c not in s2i:
                continue
            si = s2i[c]
            m = F.normalize(mat.float(), dim=-1)
            IP[si] = F.normalize(m.mean(0), dim=0)
            covered[si] = True
            IFl.append(m)
            ILl.append(torch.full((m.shape[0],), si, dtype=torch.long))
        IF = torch.cat(IFl).to(dev)
        IL = torch.cat(ILl).to(dev)
        IP = IP.to(dev)
        leg = inat_leg(qF, IP, IF, IL, covered.to(dev), S)
        Zleg[t] = zc(leg)
        print(f'  {t} base+leg built (bank covered {int(covered.sum())}/{S})', flush=True)
        del TF, TL, qF, base, leg, IF, IL
        torch.cuda.empty_cache()

    n_train_gold = n_train[yv]  # n_train of the TRUE class for each val row

    print('\n=== accuracy by n_train bucket, wi grid ===', flush=True)
    for lo, hi in BUCKETS:
        mask = (n_train_gold >= lo) & (n_train_gold <= hi)
        n_rows = int(mask.sum())
        print(f'\nbucket n_train in [{lo},{hi}]: {n_rows} val rows '
              f'({100 * n_rows / len(yv):.1f}% of val)', flush=True)
        yv_b = yv[mask]
        for wi in GRID_WI:
            f = sum(BASE_W[t] * (Zbase[t] + wi * Zleg[t]) for t in BASE_W)
            acc = 100 * (f[mask].argmax(1).cpu() == yv_b).float().mean().item()
            print(f'  wi={wi:<5} -> {acc:6.2f}', flush=True)


if __name__ == '__main__':
    main()
