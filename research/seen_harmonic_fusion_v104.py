"""v104: replace the ADDITIVE (sum) cross-leg combination with an agreement-rewarding,
worst-leg-punishing OPERATOR -- min / soft-min / sigmoid-harmonic-mean -- over the
same 3 deployed legs (ctftshift/ftshift/fullft336shift), same weights, same z-scored
per-leg scores as v98's deployed fusion. Distinct axis from v98 (weight reweighting)
and v99 (ordinal rank fusion, both DEAD): this keeps the weights and the cardinal
magnitudes, only changes how the 3 numbers per (row, class) are folded into one.

    x_t(i,c)   = DEPLOYED[t] * zc(base_t)(i,c)          # v98's per-leg weighted z-score
    sum        = x_ctft + x_ft + x_full                  # deployed (AND-in-logit-space already,
                                                           # since these are additive log-scores --
                                                           # but empirically worth checking non-additive
                                                           # alternatives that explicitly penalize the
                                                           # single worst leg rather than let a strong
                                                           # leg's outlier compensate)
    min        = min_t x_t                                # hard: worst leg sets the score
    softmin_tau= -tau * logsumexp_t(-x_t/tau)              # tau->0 => min, tau->inf => mean (per-row
                                                           # constant shift only, argmax-invariant)
    harmonic_T : P_t = sigmoid(x_t/T) in (0,1);  H = (sum_t w_t) / (sum_t w_t/P_t)
                                                           # any P_t -> 0 (leg strongly against c)
                                                           # craters H -> 0 regardless of the other legs

Baselines on THIS same leak-free holdout (v98/v103's harness/split, single held-out
class-disjoint split, no CV -- time-boxed check, not a final claim):
  deployed 3-leg linear (additive) fusion : 89.87%
  ctftshift solo                          : 91.20%
  v83 seen re-ranker (current best-in-repo): 91.674%

Pre-registered before any number is seen:
  grid: op in {min, softmin(tau), harmonic(T)}
        tau in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        T   in [0.5, 1.0, 2.0, 4.0]
  kill bar: best op lifts val closed-set accuracy >= +1.0pt over 91.674% i.e. >= 92.674%.

  conda activate onet && python research/seen_harmonic_fusion_v104.py
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
LEGS = list(DEPLOYED.keys())
V83_RERANK = 91.674
KILL_LIFT = 1.0
TAU_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
T_GRID = [0.5, 1.0, 2.0, 4.0]


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
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen]).to(dev)
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    X = {}  # weighted z-scored legs: DEPLOYED[t] * zc(base_t)
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
        X[t] = DEPLOYED[t] * zc(base)
        acc = 100 * (base.argmax(1) == yv).float().mean().item()
        print(f'  {t:16s} solo val acc {acc:6.2f}', flush=True)
        del TF, TL, qF, base
        torch.cuda.empty_cache()

    Xs = torch.stack([X[t] for t in LEGS], dim=0)  # [3, N, S]

    def acc_of(score):
        return 100 * (score.argmax(1) == yv).float().mean().item()

    rows = {}
    rows['deployed_sum'] = acc_of(Xs.sum(0))
    rows['min'] = acc_of(Xs.min(0).values)

    for tau in TAU_GRID:
        soft = -tau * torch.logsumexp(-Xs / tau, dim=0)
        rows[f'softmin_tau{tau}'] = acc_of(soft)

    wsum = sum(DEPLOYED.values())
    for T in T_GRID:
        Pt = torch.sigmoid(Xs / T).clamp(min=1e-4)
        denom = sum(DEPLOYED[t] / Pt[i] for i, t in enumerate(LEGS))
        H = wsum / denom
        rows[f'harmonic_T{T}'] = acc_of(H)

    print('\n=== v104 combination-operator grid (val closed-set accuracy) ===', flush=True)
    for k, v in sorted(rows.items(), key=lambda kv: -kv[1]):
        print(f'  {k:22s} -> {v:6.2f}', flush=True)

    best = max(rows, key=rows.get)
    best_acc = rows[best]
    lift = best_acc - V83_RERANK
    verdict = (f'CLEARS -- {best} beats v83 rerank by {lift:+.2f}pt' if lift >= KILL_LIFT
               else f'DEAD -- best op ({best} {best_acc:.2f}) does not beat v83 rerank '
                    f'({V83_RERANK}) by >= +{KILL_LIFT}pt')
    print(f'\ndeployed_sum {rows["deployed_sum"]:.2f}  ctftshift-solo-ref 91.20  '
          f'v83-rerank-ref {V83_RERANK}  best {best} {best_acc:.2f}  lift-vs-v83 {lift:+.2f}',
          flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'rows': rows, 'v83_rerank': V83_RERANK, 'best': best, 'best_acc': best_acc,
               'lift_vs_v83': lift, 'verdict': verdict},
              open(f'{OUT}/seen_harmonic_fusion_v104.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_harmonic_fusion_v104.json', flush=True)


if __name__ == '__main__':
    main()
