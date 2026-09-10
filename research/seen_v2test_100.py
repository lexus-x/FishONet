"""Test whether the fullft336_v2 (8-epoch extended fine-tune) embeddings lift seen
holdout closed-set accuracy over the deployed 3-leg fusion / the v98 reweight winner.
fullft336_v2 is already wired into MEMBERS (weight 0, unused) in learned_gate_v77.py --
this just scores it solo and in combos. No training, cached embeddings only.

  conda activate onet && python research/seen_v2test_100.py
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, MEMBERS, dev, holdout_split, load_train_embs, zc
from seen_inat_v95 import member_scores

LEGS = ['ctftshift', 'ftshift', 'fullft336shift', 'fullft336_v2']
DEPLOYED = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5, 'fullft336_v2': 0.0}
V98_WINNER = {'ctftshift': 3.0, 'ftshift': 0.0, 'fullft336shift': 0.5, 'fullft336_v2': 0.0}
KILL_LIFT = 1.0


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
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zbase = {}
    for t in LEGS:
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
        acc = 100 * (base.argmax(1).cpu() == yv).float().mean().item()
        print(f'  {t:16s} solo val acc {acc:6.2f}', flush=True)
        del TF, TL, qF, base
        torch.cuda.empty_cache()

    def fused_acc(w):
        f = sum(w[t] * Zbase[t] for t in LEGS)
        return 100 * (f.argmax(1).cpu() == yv).float().mean().item()

    dep_acc = fused_acc(DEPLOYED)
    v98_acc = fused_acc(V98_WINNER)
    print(f'\ndeployed (no v2) {dep_acc:.2f} | v98 winner (no v2) {v98_acc:.2f}', flush=True)

    print('\n=== combos including fullft336_v2 ===', flush=True)
    grid = {
        'ctftshift': [1.0, 2.0, 3.0],
        'ftshift': [0.0, 1.0, 2.5],
        'fullft336shift': [0.0, 0.5, 2.5],
        'fullft336_v2': [0.5, 1.0, 2.0, 3.0],
    }
    rows = {}
    names = list(grid.keys())
    for combo in itertools.product(*grid.values()):
        w = dict(zip(names, combo))
        rows[str(w)] = fused_acc(w)
    best_key = max(rows, key=rows.get)
    top8 = sorted(rows.items(), key=lambda kv: -kv[1])[:8]
    for k, v in top8:
        print(f'  {k} -> {v:.2f}', flush=True)
    best = rows[best_key]
    lift_vs_v98 = best - v98_acc
    print(f'\nbest-with-v2 {best:.2f} vs v98winner(no v2) {v98_acc:.2f}  lift {lift_vs_v98:+.2f}pt '
          f'(kill >= +{KILL_LIFT})', flush=True)
    verdict = ('CLEARS' if lift_vs_v98 >= KILL_LIFT else 'DEAD')
    print(f'VERDICT: {verdict} -- fullft336_v2 solo acc vs old fullft336shift is the key signal above', flush=True)
    json.dump({'rows': rows, 'deployed_no_v2': dep_acc, 'v98_winner_no_v2': v98_acc,
                'best': best_key, 'best_acc': best, 'lift_vs_v98': lift_vs_v98, 'verdict': verdict},
               open(f'{OUT}/seen_v2test_100.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_v2test_100.json', flush=True)


if __name__ == '__main__':
    main()
