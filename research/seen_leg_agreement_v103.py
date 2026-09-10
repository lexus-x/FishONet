"""v103: inter-leg PREDICTION AGREEMENT as a per-candidate bonus on the seen fusion score.

Genuinely new mechanism (round-2 idea (c), not a reweight of v98/v95's continuous
scores): every prior seen-fusion change (v93/v95/v98/v99) only rescaled or added
CONTINUOUS per-member score vectors -- reweighting can only stretch/shrink a leg's
whole score vector, it can never single out one specific candidate class. This adds
a DISCRETE signal that a linear reweight structurally cannot represent: whether the
three deployed legs' OWN independent top-1/top-2 picks land on the same class. When
two or three legs' argmax agree on class c, that is evidence for c that is orthogonal
to how large any one leg's raw dot-product happens to be -- it catches the case where
the z-score sum's argmax is pulled off the majority pick by one leg's outlier-large
score for a DIFFERENT class (a failure mode no amount of global reweighting can fix,
since reweighting scales entire vectors, not single entries).

Score:
    fused_base(i,c) = sum_t DEPLOYED[t] * zc(base_t)(i,c)      # v98's deployed fusion
    bonus(i,c)       = one of three discrete cross-leg vote counts (below)
    final(i,c)       = fused_base(i,c) + ALPHA * bonus(i,c)
    pred(i)          = argmax_c final(i,c)

Bonus variants (pre-registered, all "model-free" -- no learned weights, just counting
which classes the legs' OWN argmax/top-2 already point to):
  agree1         : +1 to class c for every leg whose top-1 == c            (range 0-3)
  agree_top2soft : +1 for a leg's top-1 == c, +0.5 for a leg's #2 == c     (softer, catches near-misses)
  unanimous      : +1 to class c ONLY when all 3 legs' top-1 unanimously == c (else 0)

Baselines to clear (all measured on THIS same leak-free holdout, research/seen_reweight_v98.py's
harness/split, single held-out class-disjoint split, no CV -- time-boxed check, not a final claim):
  deployed 3-leg linear fusion (ALPHA=0, i.e. v98's DEPLOYED weights) : 89.87%
  ctftshift solo                                                      : 91.20%
  v83 seen re-ranker (current best-in-repo)                           : 91.674%

Pre-registered before any number is seen:
  grid: bonus in {agree1, agree_top2soft, unanimous} x ALPHA in
        [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
  kill bar: best (bonus, ALPHA) lifts val closed-set accuracy >= +1.0pt over 91.674%
            i.e. >= 92.674% holdout accuracy.

Why more likely to transfer to real than v98's reweight (which shipped NEGATIVE, v99
53.60% vs v83 53.70%): per HANDOFF.md's twice-measured finding, "new information in
the gate compounds; re-weighting does not" -- v77's new features gave +1.767 overall-pt
(routing +1.082, conditionals +0.686) while v79's reweighting of those SAME features
only netted +0.042 (a_cond fell, giving ~90% of the routing gain back). Cross-leg vote
agreement is information the fusion has never used (only continuous score MAGNITUDE
has ever been read); if it lifts holdout accuracy, unlike a reweight it should carry a
non-degenerate fraction of that lift into a_cond / seen-head real accuracy too, because
it changes *which* class wins for specific hard rows, not just how confidently.

  conda activate onet && python research/seen_leg_agreement_v103.py
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
V83_RERANK = 91.674          # current best-in-repo, seen re-ranker (ships in real v83)
KILL_LIFT = 1.0              # over V83_RERANK
ALPHA_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def agree1_bonus(top1, N, S):
    """+1 per leg whose top-1 pick == c."""
    b = torch.zeros(N, S, device=dev)
    idx = torch.arange(N, device=dev)
    for t in LEGS:
        b[idx, top1[t]] += 1.0
    return b


def agree_top2soft_bonus(top2idx, N, S):
    """+1 for a leg's top-1 == c, +0.5 for a leg's #2 == c."""
    b = torch.zeros(N, S, device=dev)
    idx = torch.arange(N, device=dev)
    for t in LEGS:
        ids = top2idx[t]
        b[idx, ids[:, 0]] += 1.0
        b[idx, ids[:, 1]] += 0.5
    return b


def unanimous_bonus(top1, N, S):
    """+1 to c ONLY when all 3 legs' top-1 unanimously agree on c."""
    b = torch.zeros(N, S, device=dev)
    idx = torch.arange(N, device=dev)
    t0, t1, t2 = LEGS
    all_agree = (top1[t0] == top1[t1]) & (top1[t1] == top1[t2])
    b[idx[all_agree], top1[t0][all_agree]] = 1.0
    return b


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
    N = len(val_seen)
    print(f'val_seen rows {N} | seen classes {S}', flush=True)

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

    fused_base = sum(DEPLOYED[t] * Zbase[t] for t in LEGS)
    base_acc = 100 * (fused_base.argmax(1).cpu() == yv).float().mean().item()
    print(f'\ndeployed 3-leg fusion (ALPHA=0) val acc: {base_acc:.2f}  '
          f'(sanity-check vs v98\'s 89.87)', flush=True)

    top1 = {t: Zbase[t].argmax(1) for t in LEGS}
    top2idx = {t: Zbase[t].topk(2, dim=1).indices for t in LEGS}
    bonuses = {
        'agree1': agree1_bonus(top1, N, S),
        'agree_top2soft': agree_top2soft_bonus(top2idx, N, S),
        'unanimous': unanimous_bonus(top1, N, S),
    }
    n_unanimous = int((bonuses['unanimous'].sum(1) > 0).sum())
    print(f'rows where all 3 legs unanimously agree: {n_unanimous}/{N} '
          f'({100 * n_unanimous / N:.1f}%)', flush=True)

    print('\n=== pre-registered bonus x ALPHA grid (val closed-set accuracy) ===', flush=True)
    rows = {'alpha0_base': base_acc}
    for bname, B in bonuses.items():
        for alpha in ALPHA_GRID:
            final = fused_base + alpha * B
            acc = 100 * (final.argmax(1).cpu() == yv).float().mean().item()
            rows[f'{bname}_a{alpha}'] = acc

    best_key = max(rows, key=rows.get)
    lift_v83 = rows[best_key] - V83_RERANK
    lift_base = rows[best_key] - base_acc
    top8 = sorted(rows.items(), key=lambda kv: -kv[1])[:8]
    for k, v in top8:
        print(f'  {k:24s} -> {v:6.2f}', flush=True)
    verdict = (f'CLEARS — {best_key} beats v83 re-ranker by {lift_v83:+.2f}pt'
               if lift_v83 >= KILL_LIFT else
               'DEAD — no leg-agreement bonus lifts holdout acc >= +1.0pt over v83 (91.674)')
    print(f'\nbase(ALPHA=0) {base_acc:.2f}  v83-reranker {V83_RERANK:.3f}  '
          f'best {best_key} {rows[best_key]:.2f}  '
          f'lift-vs-fusion {lift_base:+.2f}  lift-vs-v83 {lift_v83:+.2f} (kill >= +{KILL_LIFT})',
          flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'rows': rows, 'base_acc': base_acc, 'v83_rerank': V83_RERANK,
               'best': best_key, 'lift_vs_v83': lift_v83, 'lift_vs_fusion': lift_base,
               'n_unanimous_frac': n_unanimous / N, 'verdict': verdict},
              open(f'{OUT}/seen_leg_agreement_v103.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_leg_agreement_v103.json', flush=True)


if __name__ == '__main__':
    main()
