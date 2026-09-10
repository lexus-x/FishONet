"""v101 Idea (d): James-Stein prototype shrinkage as a 4th ADDITIVE fusion term.

New paradigm (not another reweighting of proto+cmax+taxon): for each seen class c,
build a SECOND prototype that blends the class's own (noisy, few-shot for the tail)
image-mean prototype P_c with the class's taxon-text anchor T_c, weighted by how much
training evidence backs P_c:

    lambda_c   = n_c / (n_c + tau)                      # James-Stein shrinkage weight
    P_shrunk_c = normalize(lambda_c * P_c + (1-lambda_c) * T_c)
    shrink_score(q, c) = q . P_shrunk_c

This is mechanically different from the existing fusion even though it reuses P and T:
the deployed formula ADDS two separately-normalized SCORES (q.P + LAM*q.T); shrinkage
blends the two REPRESENTATIONS before a single cosine, which is a nonlinear function of
(P, T) that no linear combination of the existing per-leg scores can reproduce. For
n_c -> 0 it degenerates smoothly to the taxon-only classifier; for n_c >> tau it recovers
the deployed prototype. Targets exactly the diagnosed failure mode (v97: n_train in [0,2]
= 5.7% of val rows but only 59.91% acc) rather than uniformly reweighting saturated legs.

Two-stage test, both required to ship:
  1. SOLO: does shrink_score (fused across the 3 deployed members at BASE_W) beat the
     plain proto+cmax+taxon fusion on its own, and does the gain concentrate in the
     n_train<=2 / n_train in [3,5] buckets (per-bucket breakdown, v97-style)?
  2. ADDITIVE (the assigned test): fused = sum_t BASE_W[t] * (Zbase[t] + w4 * Zshrink[t]),
     sweep tau in {1,2,4,8,16,32,64} x w4 in {0,0.25,0.5,1.0,1.5,2.5,4.0} (pre-registered,
     42 combos, cheap -- reuses the same cached member scores, no new embeddings/GPU
     training). Kill bar: best additive combo >= 92.674% holdout (91.674 + 1.0pt, the
     v83 re-ranker's holdout number -- see research/seen_reweight_v98.py harness).

Why this is more likely to transfer than v98's dead linear reweight: v98 reweighted the
SAME three already-saturated legs against each other (no new information -> a_cond/b_cond
argument from HANDOFF v77 predicts near-zero transfer, confirmed real-negative). This adds
a genuinely new per-class signal (evidence-weighted blend with the taxon anchor) that the
fusion has never had access to in representation space -- same "new information compounds,
reweighting doesn't" principle that made v77's gate real-positive. If the additive test
fails but SOLO's bucket breakdown still shows tail-specific lift, that is evidence for
going back to (b) a trained per-class shrinkage-strength (not a single global tau) rather
than abandoning the direction.

  conda activate onet && python research/seen_shrink_v101_james_stein_proto_taxon_additive.py
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

BASE_W = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
TAU_GRID = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
W4_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.5, 4.0]
KILL_LIFT = 1.0          # over 91.674 -> 92.674 absolute
V83_RERANKER_HOLDOUT = 91.674
BUCKETS = [(0, 2), (3, 5), (6, 1_000_000)]


def shrink_scores(qF, P, cnt, Tanchor, tau):
    """cos(q, normalize(lambda*P + (1-lambda)*T)), lambda = n/(n+tau)."""
    lam = (cnt / (cnt + tau)).unsqueeze(1).to(dev)
    Pshrunk = F.normalize(lam * P + (1 - lam) * Tanchor, dim=-1)
    return qF @ Pshrunk.t()


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
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    n_train = torch.tensor([len(trby[c]) for c in seen], dtype=torch.float32)
    n_train_gold = n_train[yv]
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zbase, Zshrink = {}, {t: {} for t in BASE_W}
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
        Pn = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
        TF = torch.stack(TFl).to(dev)
        TL = torch.tensor(TLl).to(dev)
        qF = torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev)

        base = member_scores(qF, Pn, TF, TL, Tseen, S)
        Zbase[t] = zc(base)
        acc = 100 * (base.argmax(1).cpu() == yv).float().mean().item()
        print(f'  {t:16s} deployed-base solo val acc {acc:6.2f}', flush=True)

        for tau in TAU_GRID:
            sc = shrink_scores(qF, Pn, cnt, Tseen, tau)
            Zshrink[t][tau] = zc(sc)
        del TF, TL, qF, base
        torch.cuda.empty_cache()

    def base_fused_acc():
        f = sum(BASE_W[t] * Zbase[t] for t in BASE_W)
        return 100 * (f.argmax(1).cpu() == yv).float().mean().item()

    def shrink_fused(tau):
        return sum(BASE_W[t] * Zshrink[t][tau] for t in BASE_W)

    def additive_acc(tau, w4):
        f = sum(BASE_W[t] * (Zbase[t] + w4 * Zshrink[t][tau]) for t in BASE_W)
        return 100 * (f.argmax(1).cpu() == yv).float().mean().item()

    base_acc = base_fused_acc()
    print(f'\ndeployed 3-leg fusion (proto+2*cmax+4*taxon) holdout acc: {base_acc:.2f}', flush=True)
    print(f'target bar: >= {V83_RERANKER_HOLDOUT + KILL_LIFT:.3f} '
          f'({V83_RERANKER_HOLDOUT} v83 re-ranker holdout + {KILL_LIFT} kill lift)', flush=True)

    print('\n=== STAGE 1: solo shrink-fusion accuracy by tau ===', flush=True)
    solo_rows = {}
    for tau in TAU_GRID:
        sf = shrink_fused(tau)
        acc = 100 * (sf.argmax(1).cpu() == yv).float().mean().item()
        solo_rows[tau] = acc
        print(f'  tau={tau:<6} solo shrink acc {acc:6.2f}  (vs base {acc - base_acc:+.2f})', flush=True)
    best_solo_tau = max(solo_rows, key=solo_rows.get)

    print(f'\n=== per-bucket breakdown at best solo tau={best_solo_tau} (base vs shrink-solo) ===', flush=True)
    sf_best = shrink_fused(best_solo_tau)
    bf = sum(BASE_W[t] * Zbase[t] for t in BASE_W)
    for lo, hi in BUCKETS:
        mask = (n_train_gold >= lo) & (n_train_gold <= hi)
        n_rows = int(mask.sum())
        ab = 100 * (bf[mask].argmax(1).cpu() == yv[mask]).float().mean().item()
        asf = 100 * (sf_best[mask].argmax(1).cpu() == yv[mask]).float().mean().item()
        print(f'  n_train in [{lo},{hi}]: {n_rows} rows ({100 * n_rows / len(yv):.1f}%) '
              f'| base {ab:6.2f} | shrink-solo {asf:6.2f} | delta {asf - ab:+.2f}', flush=True)

    print('\n=== STAGE 2 (assigned): additive ensemble, tau x w4 grid ===', flush=True)
    rows = {}
    for tau in TAU_GRID:
        for w4 in W4_GRID:
            acc = additive_acc(tau, w4)
            rows[f'tau{tau}_w4{w4}'] = acc
    best_key = max(rows, key=rows.get)
    top5 = sorted(rows.items(), key=lambda kv: -kv[1])[:5]
    for k, v in top5:
        print(f'  {k} -> {v:.2f}', flush=True)
    lift_vs_base = rows[best_key] - base_acc
    lift_vs_v83 = rows[best_key] - V83_RERANKER_HOLDOUT
    verdict = (f'CLEARS — {best_key} beats target' if lift_vs_v83 >= KILL_LIFT
               else 'DEAD — additive shrinkage does not clear +1.0pt over 91.674')
    print(f'\nbase(this harness) {base_acc:.2f}  best {best_key} -> {rows[best_key]:.2f} '
          f'(lift over this-harness base {lift_vs_base:+.2f}, over v83-holdout-target {lift_vs_v83:+.2f})',
          flush=True)
    print(f'VERDICT: {verdict}', flush=True)

    # per-bucket breakdown at the best ADDITIVE combo too (what the idea's kill bar targets)
    best_tau = float(best_key.split('_w4')[0].replace('tau', ''))
    best_w4 = float(best_key.split('_w4')[1])
    af_best = sum(BASE_W[t] * (Zbase[t] + best_w4 * Zshrink[t][best_tau]) for t in BASE_W)
    print(f'\n=== per-bucket breakdown at best ADDITIVE combo {best_key} (base vs additive) ===', flush=True)
    bucket_report = []
    for lo, hi in BUCKETS:
        mask = (n_train_gold >= lo) & (n_train_gold <= hi)
        n_rows = int(mask.sum())
        ab = 100 * (bf[mask].argmax(1).cpu() == yv[mask]).float().mean().item()
        aaf = 100 * (af_best[mask].argmax(1).cpu() == yv[mask]).float().mean().item()
        print(f'  n_train in [{lo},{hi}]: {n_rows} rows ({100 * n_rows / len(yv):.1f}%) '
              f'| base {ab:6.2f} | additive-best {aaf:6.2f} | delta {aaf - ab:+.2f}', flush=True)
        bucket_report.append({'lo': lo, 'hi': hi, 'n_rows': n_rows, 'base_acc': ab,
                               'additive_acc': aaf, 'delta': aaf - ab})

    json.dump({'base_acc': base_acc, 'solo_rows': solo_rows, 'additive_rows': rows,
               'best_additive': best_key, 'best_additive_acc': rows[best_key],
               'bucket_report_at_best_additive': bucket_report,
               'v83_reranker_holdout': V83_RERANKER_HOLDOUT, 'kill_lift': KILL_LIFT,
               'verdict': verdict},
              open(f'{OUT}/seen_shrink_v101.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_shrink_v101.json', flush=True)


if __name__ == '__main__':
    main()
