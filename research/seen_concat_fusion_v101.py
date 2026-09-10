"""v101: feature-level (representation) fusion instead of score-level fusion.

Every deployed SEEN scorer (v56/v77/v95/v98...) fuses per-leg SCORES: compute a
cosine-similarity score per leg (ctftshift/ftshift/fullft336shift), z-score each,
then weighted-sum the scores. This tries fusing the REPRESENTATIONS instead: L2-
normalize each leg's embedding, concatenate into one joint vector per image (train
and query), build ONE prototype per class in the concatenated space (mean of
concatenated train vectors, renormalized), then a single cosine-similarity argmax.
No score fusion, no cmax term, no taxon leg -- just: does concatenating BEFORE
computing similarity beat combining similarities AFTER?

Uses the same holdout split as every other seen_*_v9x script (rarest-20%-classes
pseudo-novel split, val_seen rows only, class-disjoint). Kill bar: +1.0pt over the
repo-best 91.674% closed-set holdout accuracy.

  conda activate onet && python research/seen_concat_fusion_v101.py
"""
from __future__ import annotations

import json
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, dev, holdout_split, load_train_embs

LEGS = ['ctftshift', 'ftshift', 'fullft336shift']
BASELINE = 91.674
KILL_LIFT = 1.0
BUCKETS = [(0, 2), (3, 5), (6, 1_000_000)]


def main():
    torch.set_num_threads(8)
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}

    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby, val_seen = sp['trby'], [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    n_train = torch.tensor([len(trby[c]) for c in seen], dtype=torch.float32)
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    # each train[t] feats are already L2-normalized per-leg (load() in learned_gate_v77).
    # concat train vectors (per-image), then per-class prototype = mean, renormalized.
    idxs = {t: train[t][0] for t in LEGS}
    feats = {t: train[t][1] for t in LEGS}
    dim_total = sum(feats[t].shape[1] for t in LEGS)
    print(f'legs {LEGS} | concat dim {dim_total}', flush=True)

    def concat_vec(fn):
        return torch.cat([feats[t][idxs[t][fn]] for t in LEGS], dim=0)

    P = torch.zeros(S, dim_total)
    cnt = torch.zeros(S)
    for c in seen:
        for fn in trby[c]:
            if not all(fn in idxs[t] for t in LEGS):
                continue
            P[s2i[c]] += concat_vec(fn)
            cnt[s2i[c]] += 1
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)

    Q = torch.stack([concat_vec(fn) for fn in val_seen])
    Q = F.normalize(Q, dim=-1).to(dev)

    sim = Q @ P.t()
    pred = sim.argmax(1).cpu()
    acc = 100 * (pred == yv).float().mean().item()
    lift = acc - BASELINE
    verdict = ('CLEARS — concat-space prototype fusion beats score fusion' if lift >= KILL_LIFT
               else 'DEAD — representation-level concat fusion does not beat score fusion by >=1.0pt')

    print(f'\nconcat-fusion prototype cosine argmax: {acc:.3f}% '
          f'(baseline {BASELINE:.3f}%, lift {lift:+.3f}pt, kill >= +{KILL_LIFT})', flush=True)
    print(f'VERDICT: {verdict}', flush=True)

    print('\n=== accuracy by n_train bucket ===', flush=True)
    n_train_gold = n_train[yv]
    bucket_lines = []
    for lo, hi in BUCKETS:
        mask = (n_train_gold >= lo) & (n_train_gold <= hi)
        n_rows = int(mask.sum())
        bacc = 100 * (pred[mask] == yv[mask]).float().mean().item() if n_rows else float('nan')
        line = f'  n_train in [{lo},{hi}]: {n_rows:5d} rows ({100 * n_rows / len(yv):.1f}%) -> {bacc:.2f}%'
        print(line, flush=True)
        bucket_lines.append(line)

    json.dump({'acc': acc, 'baseline': BASELINE, 'lift': lift, 'verdict': verdict,
               'dim_total': dim_total, 'legs': LEGS},
              open(f'{OUT}/seen_concat_fusion_v101.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/seen_concat_fusion_v101.json', flush=True)


if __name__ == '__main__':
    main()
