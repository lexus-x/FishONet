"""v98 verify: is the seen-fusion reweighting gain (1.0/2.5/2.5 -> ctftshift-dominant,
+1.34pt on the full val_seen pool) real signal or search overfitting on one split?

HANDOFF explicitly burned this exact failure mode once already (unseen text-stack
fusion-weight re-tune: "without CV it 'gains' +1.3; with a class-disjoint tune/val
split, held-out val = -0.26 -- pure overfit"). Same magnitude, same mechanism
(re-weighting an existing fusion), so this MUST be checked before trusting it.

Method: split the 11,866-row val_seen pool in half by a fixed seed (disjoint from
the grid search, which used the full pool -- so this is a fresh generalization
check, not a re-run of the same numbers). Report deployed vs ctftshift-solo vs the
v98 winner on EACH half independently. If the ranking and magnitude hold on both
halves, it's a genuine, non-overfit fusion-weight bug (the current deployed
1.0/2.5/2.5 was tuned for the ctftbig-era ensemble and never re-validated after
ctftshift replaced ctftbig as the strongest member).

  conda activate onet && python research/seen_reweight_v98_verify.py
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
from seen_inat_v95 import member_scores

DEPLOYED = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
CANDIDATES = {
    'deployed_1.0_2.5_2.5': DEPLOYED,
    'ctftshift_solo_1.0_0_0': {'ctftshift': 1.0, 'ftshift': 0.0, 'fullft336shift': 0.0},
    'v98_winner_3.0_0_0.5': {'ctftshift': 3.0, 'ftshift': 0.0, 'fullft336shift': 0.5},
    'moderate_2.0_1.0_1.0': {'ctftshift': 2.0, 'ftshift': 1.0, 'fullft336shift': 1.0},
}


def main():
    torch.set_num_threads(8)
    torch.manual_seed(0)
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
    yv_full = torch.tensor([s2i[lab[f]] for f in val_seen])
    n = len(val_seen)
    print(f'val_seen rows {n} | seen classes {S}', flush=True)

    perm = torch.randperm(n)
    half_a = perm[:n // 2]
    half_b = perm[n // 2:]

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zbase = {}
    for t, _, _ in MEMBERS:
        if t not in DEPLOYED:
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
        del TF, TL, qF, base
        torch.cuda.empty_cache()

    def acc_on(mask_idx, w):
        f = sum(w[t] * Zbase[t] for t in DEPLOYED)
        sub = f[mask_idx]
        yv_sub = yv_full[mask_idx]
        return 100 * (sub.argmax(1).cpu() == yv_sub).float().mean().item()

    print('\n=== half-split generalization check (fresh random split, NOT the grid-search pool) ===',
          flush=True)
    print(f'{"config":<28}{"full":>8}{"half A":>8}{"half B":>8}', flush=True)
    rows = {}
    for name, w in CANDIDATES.items():
        full = acc_on(torch.arange(n), w)
        a = acc_on(half_a, w)
        b = acc_on(half_b, w)
        rows[name] = {'full': full, 'half_a': a, 'half_b': b}
        print(f'{name:<28}{full:>8.2f}{a:>8.2f}{b:>8.2f}', flush=True)

    dep = rows['deployed_1.0_2.5_2.5']
    win = rows['v98_winner_3.0_0_0.5']
    lift_a = win['half_a'] - dep['half_a']
    lift_b = win['half_b'] - dep['half_b']
    print(f'\nlift vs deployed: half A {lift_a:+.2f}  half B {lift_b:+.2f}  '
          f'(full-pool lift was +{win["full"] - dep["full"]:.2f})', flush=True)
    consistent = lift_a > 0.5 and lift_b > 0.5
    print(f'VERDICT: {"CONSISTENT across independent halves -- real signal" if consistent else "INCONSISTENT -- likely search/holdout overfit, do NOT ship"}',
          flush=True)
    json.dump(rows, open(f'{OUT}/seen_reweight_v98_verify.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_reweight_v98_verify.json', flush=True)


if __name__ == '__main__':
    main()
