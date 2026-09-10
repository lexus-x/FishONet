"""v96: does Sinkhorn balanced-assignment help the SEEN route?

Sinkhorn (class-balance correction) has only ever been applied to the UNSEEN
route (HANDOFF: +0.09pt real originally; NEGATIVE once the unseen route got a
calibrated re-ranker, because "Sinkhorn's original gain came from correcting a
MISCALIBRATED fusion by forcing class balance; a calibrated model doesn't need
it"). The seen route's re-ranker (rerank_seen_v83) is ALSO a calibrated
logistic model, and the seen "test" folder is natural photography (almost
certainly non-uniform species frequency), unlike unseen's more uniform pseudo
assignment -- both facts predict Sinkhorn will be negative here too. This is a
5-minute holdout check to confirm/kill that prediction before spending any
more time on it. No new embeddings; reuses seen_inat_v95's base fusion (wi=0,
i.e. deployed recipe unchanged).

  conda activate onet && python research/seen_sinkhorn_v96.py
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
from seen_inat_v95 import BASE_W, LAM, member_scores

TAUS = [0.0, 1.0, 1.5, 1.8, 2.5, 3.5, 5.0]
SINK_ITER = 50


def sinkhorn(logits, tau, n_iter=SINK_ITER):
    P = torch.softmax(logits / tau, dim=1)
    n, C = P.shape
    target_col = n / C
    for _ in range(n_iter):
        P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-9)
        P = P * (target_col / P.sum(dim=0, keepdim=True).clamp(min=1e-9))
    return P


def main():
    torch.set_num_threads(8)
    lab = json.load(open(f'{D}/label_train.json'))
    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby, val_seen = sp['trby'], [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    ci = {c: i for i, c in enumerate(list(pickle.load(open(f'{D}/all_classes.pkl', 'rb'))))}
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    f0 = torch.zeros(len(val_seen), S, device=dev)
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
        f0 += BASE_W[t] * zc(base)
        print(f'  {t} base val acc {100 * (base.argmax(1).cpu() == yv).float().mean().item():.2f}', flush=True)
        del TF, TL, qF, base
        torch.cuda.empty_cache()

    base_acc = 100 * (f0.argmax(1).cpu() == yv).float().mean().item()
    print(f'\nbase (no Sinkhorn) fused val acc: {base_acc:.2f}', flush=True)

    print('\n=== Sinkhorn tau sweep (val closed-set accuracy) ===', flush=True)
    rows = {'base': base_acc}
    for tau in TAUS:
        if tau == 0.0:
            continue
        P = sinkhorn(f0, tau)
        acc = 100 * (P.argmax(1).cpu() == yv).float().mean().item()
        rows[f'tau{tau}'] = acc
        print(f'  tau={tau:<5} -> {acc:6.2f}  (delta {acc - base_acc:+.2f})', flush=True)

    best = max(rows, key=rows.get)
    lift = rows[best] - base_acc
    verdict = ('CLEARS' if lift >= 1.0 else 'DEAD — Sinkhorn does not lift seen-route holdout acc >= +1.0')
    print(f'\nbest {best} {rows[best]:.2f}  lift {lift:+.2f}pt  VERDICT: {verdict}', flush=True)
    json.dump({'rows': rows, 'base': base_acc, 'best': best, 'lift': lift, 'verdict': verdict},
              open(f'{OUT}/seen_sinkhorn_v96.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_sinkhorn_v96.json', flush=True)


if __name__ == '__main__':
    main()
