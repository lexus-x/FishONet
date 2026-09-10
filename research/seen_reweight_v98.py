"""v98: re-tune the SEEN fusion's member weights (ctftshift/ftshift/fullft336shift).

Current weights 1.0/2.5/2.5 date from the "concentrated ensemble" comparison at v27
(2026-07-23, an unlearned 10-model vs 3-model proxy-score pick) and have been carried
unchanged through every subsequent architecture change (learned gate v77/v79, seen
re-ranker v83) without ever being re-tuned against the CURRENT stack. Per-candidate
re-ranking and per-image gating are measured closed/saturated, but this specific
hyperparameter -- how the three members combine into the base fusion the re-ranker's
top-K is drawn from -- was never itself revisited. Pre-registered before any number
was seen: grid below, kill bar val closed-set accuracy lift >= +1.0pt over 1.0/2.5/2.5
(same bar as v93/v95, single held-out split, no CV -- time-boxed check, not a final
claim). No new embeddings; reuses the same base member scores as v95/v96.

  conda activate onet && python research/seen_reweight_v98.py
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

GRID = {
    'ctftshift': [0.5, 1.0, 1.5, 2.0, 3.0],
    'ftshift': [0.0, 0.5, 1.0, 1.5, 2.5],
    'fullft336shift': [0.0, 0.5, 1.0, 1.5, 2.5],
}
DEPLOYED = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
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
        acc = 100 * (base.argmax(1).cpu() == yv).float().mean().item()
        print(f'  {t:16s} solo val acc {acc:6.2f}', flush=True)
        del TF, TL, qF, base
        torch.cuda.empty_cache()

    def fused_acc(w):
        f = sum(w[t] * Zbase[t] for t in DEPLOYED)
        return 100 * (f.argmax(1).cpu() == yv).float().mean().item()

    base_acc = fused_acc(DEPLOYED)
    print(f'\ndeployed 1.0/2.5/2.5 val acc: {base_acc:.2f}', flush=True)

    print('\n=== pre-registered weight grid (val closed-set accuracy) ===', flush=True)
    rows = {}
    names = list(GRID.keys())
    for combo in itertools.product(*GRID.values()):
        w = dict(zip(names, combo))
        acc = fused_acc(w)
        rows[str(w)] = acc

    best_key = max(rows, key=rows.get)
    lift = rows[best_key] - base_acc
    top5 = sorted(rows.items(), key=lambda kv: -kv[1])[:5]
    for k, v in top5:
        print(f'  {k} -> {v:.2f}', flush=True)
    verdict = (f'CLEARS — {best_key}' if lift >= KILL_LIFT
               else 'DEAD — no reweighting lifts holdout closed-set acc >= +1.0')
    print(f'\nbase {base_acc:.2f}  best {best_key} {rows[best_key]:.2f}  lift {lift:+.2f}pt '
          f'(kill >= +{KILL_LIFT})', flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'rows': rows, 'base': base_acc, 'best': best_key, 'lift': lift, 'verdict': verdict},
              open(f'{OUT}/seen_reweight_v98.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_reweight_v98.json', flush=True)


if __name__ == '__main__':
    main()
