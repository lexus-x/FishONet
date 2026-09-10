"""v93 Lever B: seen-scorer leg expansion — add TaxaBind + BioCLIP-frozen legs
to the deployed 3-member seen fusion (ctftshift 1.0 / ftshift 2.5 / fullft336shift
2.5). TaxaBind and BioCLIP have never been tested on BASE seen scoring (only as
re-ranker/gate features). Goal: raise closed-set accuracy (a_cond), which multiplies
TPR in overall = 0.5635*(TPR*a_cond) + 0.4365*(TNR*b_cond).

Selection is HOLDOUT-ONLY (train-derived split via holdout_split: rarest-20%
classes = pseudo-novel, per-class last-20% images = val_seen). Eval class labels
do not exist locally, so eval confirmation happens via real Codabench slots.

Ship design (v93): the improved fused scores replace the seen ARGMAX only;
gate routing keeps the ORIGINAL seen_block so gate features/calibration are
untouched — a_cond is isolated from TPR risk.

Pre-registered before any number was seen:
  grid w_tb x w_b2 in {0, 0.5, 1.0}^2 (9 combos, (0,0) = deployed base)
  recipe for new legs mirrors deployed members:
    taxabind: proto + 2*cmax + 4*(text taxabind-space)  [has a text head]
    b2frozen: proto + 2*cmax                            [no text head on disk]
  per-member zc standardization over the scored batch (deployed convention)
  kill bar: best-by-val combo lifts val closed-set accuracy >= +1.0pt over base

  conda activate onet && python research/seen_acond_v93.py
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import (
    D, FEATS, MEMBERS, TRAIN, dev, holdout_split, load, load_train_embs, zc,
)

LAM = 4.0
GRID_TB = [0.0, 0.5, 1.0]
GRID_B2 = [0.0, 0.5, 1.0]
BASE_W = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
KILL_LIFT = 1.0
CACHE = f'{OUT}/seen_acond_v93_cache.pt'


def build_protos(trby, seen, s2i, emb_by_member):
    """Protos + per-image train feats from TRAIN-fold rows only (val excluded)."""
    S = len(seen)
    out = {}
    for t, (idx, feats, files) in emb_by_member.items():
        P = torch.zeros(S, feats.shape[1])
        cnt = torch.zeros(S)
        TF, TL = [], []
        for c in seen:
            for fn in trby[c]:
                if fn not in idx:
                    continue
                f = feats[idx[fn]]
                P[s2i[c]] += f
                cnt[s2i[c]] += 1
                TF.append(f)
                TL.append(s2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
        out[t] = (P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev))
    return out


def member_scores(qF, P, TF, TL, Tseen, S, hspace):
    qF = qF.to(dev)
    out = torch.empty(qF.shape[0], S, device=dev)
    for i in range(0, qF.shape[0], 2000):
        e = qF[i:i + 2000]
        sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = e @ P.t() + 2.0 * cmax
        if hspace and Tseen is not None:
            sc = sc + LAM * (e @ Tseen.t())
        out[i:i + 2000] = sc
    return out


def main():
    torch.set_num_threads(8)
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(f'{D}/label_train.json'))

    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby, val_seen = sp['trby'], [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    # member embeddings: 3 deployed + taxabind + b2frozen
    emb = {t: train[t] for t, _, _ in MEMBERS}
    emb['taxabind'] = tb_train
    emb['b2frozen'] = b2f_train
    PR = build_protos(trby, seen, s2i, emb)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load(f'{OUT}/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Tseen_of = {'taxabind': Ttb[kept_idx]}
    HS = dict(MEMBERS) if False else {t: hs for t, hs, _ in MEMBERS}
    HS['taxabind'] = True   # has a text head
    HS['b2frozen'] = False  # no text head on disk (pre-registered)

    # val query features
    def val_q(t):
        idx, feats, _ = emb[t]
        return torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev)

    if __import__('os').path.exists(CACHE):
        d = torch.load(CACHE, weights_only=False)
        Zm, yv = d['Zm'], d['yv']
        print(f'loaded cached val member scores {list(Zm.keys())}', flush=True)
    else:
        Zm = {}
        for t in ['ctftshift', 'ftshift', 'fullft336shift', 'taxabind', 'b2frozen']:
            P_, TF_, TL_ = PR[t]
            sc = member_scores(val_q(t), P_, TF_, TL_, Tseen_of.get(t), S, HS[t])
            Zm[t] = zc(sc)
            acc1 = (sc.argmax(1).cpu() == torch.tensor([s2i[lab[f]] for f in val_seen])).float().mean().item()
            print(f'  {t:16s} solo val acc {100*acc1:6.2f}', flush=True)
        yv = torch.tensor([s2i[lab[f]] for f in val_seen])
        torch.save({'Zm': {k: v.cpu() for k, v in Zm.items()}, 'yv': yv}, CACHE)

    def fused_acc(w_tb, w_b2):
        f = (BASE_W['ctftshift'] * Zm['ctftshift'].to(dev) + BASE_W['ftshift'] * Zm['ftshift'].to(dev)
             + BASE_W['fullft336shift'] * Zm['fullft336shift'].to(dev)
             + w_tb * Zm['taxabind'].to(dev) + w_b2 * Zm['b2frozen'].to(dev))
        return 100 * (f.argmax(1).cpu() == yv).float().mean().item()

    print('\n=== pre-registered grid (val closed-set accuracy, holdout-only) ===', flush=True)
    rows = {}
    base_acc = None
    for w_tb in GRID_TB:
        for w_b2 in GRID_B2:
            a = fused_acc(w_tb, w_b2)
            rows[f'tb{w_tb}_b2{w_b2}'] = a
            if w_tb == 0.0 and w_b2 == 0.0:
                base_acc = a
            print(f'  w_tb={w_tb}  w_b2={w_b2}  ->  {a:6.2f}', flush=True)

    best = max(rows, key=rows.get)
    lift = rows[best] - base_acc
    verdict = ('CLEARS — build v93 seen argmax with ' + best if lift >= KILL_LIFT
               else 'DEAD — no leg addition lifts holdout closed-set accuracy >= +1.0')
    print(f'\nbase {base_acc:.2f}  best {best} {rows[best]:.2f}  lift {lift:+.2f}pt  '
          f'(kill >= +{KILL_LIFT})', flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'rows': rows, 'base': base_acc, 'best': best, 'lift': lift,
               'verdict': verdict, 'kill_lift': KILL_LIFT,
               'recipe': {'taxabind': 'proto+2cmax+4taxon(taxabind space)',
                          'b2frozen': 'proto+2cmax'}},
              open(f'{OUT}/seen_acond_v93.json', 'w'), indent=2)
    print(f"wrote {OUT}/seen_acond_v93.json", flush=True)


if __name__ == '__main__':
    main()
