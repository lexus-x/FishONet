"""v101: James-Stein / empirical-Bayes prototype shrinkage toward the taxon-text
anchor, weighted by inverse training-image count.

MECHANISM (genuinely new, not a reweighting of the 3-leg fusion): every class
prototype P_c currently = plain mean of its training-image embeddings, treated
identically regardless of how many (noisy) images back it. seen_inat_v97_bucket.py
showed accuracy craters for classes with few images (n_train<=2: 59.91%,
n_train in [3,5]: 74.39%, n_train>=6: 93.29%) -- the noisy-mean-prototype problem
is real and UNADDRESSED (every prior lever reweighted the same fixed geometric
formula, never touched what the prototype vector itself is).

Fix: shrink each class's raw mean prototype toward its taxon-text embedding
(already loaded as Tseen for the deployed LAM*taxon additive term -- reused
here as the natural, already-in-repo, training-image-count-INDEPENDENT anchor)
in proportion to how few images back it:

    lambda_c   = n_train_c / (n_train_c + tau)          # James-Stein shrinkage weight
    P_shrink_c = normalize(lambda_c * P_c + (1 - lambda_c) * Tseen_c)

tau=0 -> lambda=1 for all classes -> P_shrink == P (exact sanity check, must
reproduce the deployed baseline). Large tau -> heavy shrinkage even for
well-supported classes. Grid sweeps tau to find where shrinkage helps the
starved tail without eroding the 6+ bucket.

ISOLATION: only the P.t() term is swapped for P_shrink.t(); cmax (built from
the SAME raw per-image TF/TL, unshrunk -- it is already a different, non-mean
aggregation and is not the noisy signal) and the LAM*taxon additive term
(unchanged formula) are left exactly as in the deployed 3-leg fusion. This
isolates "does modifying the prototype's geometry help" from "does merely
adding more taxon signal help" (the latter is already tested to death via
LAM's fixed weight in every prior seen_inat_v9x script).

WHY THIS TIME MIGHT TRANSFER (unlike v98's uniform leg reweighting, which was
real-NEGATIVE despite +1.34pt holdout): v98 moved weight between three legs
that were already saturated/agreeing almost everywhere -- it only re-sorts
marginal cases within an already-optimized ordering (repo's own diagnosis:
"re-weighting the same features only slides marginal images across a fixed
ordering, so a_cond falls"). Shrinkage is different in kind: it targets a
SPECIFIC, DIAGNOSED, currently-unaddressed failure mode (n_train<=5 classes,
13.7% of val rows, where the prototype itself is admitted-noisy) with a
mechanism that only *activates* for those classes (lambda_c ~ 1 for n>=~30,
i.e. the 86.3% "n_train>=6" bucket where the current 93.29% pipeline is
already strong is left ~untouched by construction). A win here is new
INFORMATION about which classes have unreliable prototypes, not a slide along
an existing axis -- closer in kind to v77 (new evidence -> new ranking) than
to v98/v99 (reweight -> same ranking, different cutoff).

SCOPE NOTE: the mandated global kill bar is >=92.674% holdout (i.e. beating
the v83 re-ranker's 91.674%, which already includes per-candidate re-ranking
on TOP of the base 3-leg fusion). This script tests the ingredient swap
in isolation against the base 3-leg fusion it feeds (recomputed in-script,
should equal ~89.87%) -- that is the correct, apples-to-apples comparison for
THIS mechanism. Internal pass bar: best tau's overall accuracy lift over the
in-script deployed baseline >= +1.0pt, AND the 6+ bucket does not drop by more
than 0.3pt (no regression on the 86.3%-of-rows majority). A pass is NECESSARY
but not SUFFICIENT for the global bar -- the immediate follow-up (not run
here, to stay under the 4-minute/no-new-training budget) is swapping P for
P_shrink inside rerank_seen_v83.py's proto feature and re-measuring the full
re-ranked pipeline against 92.674%.

  conda activate onet && python research/seen_js_shrink_v101.py
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

BASE_W = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
LAM = 4.0
TAU_GRID = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 60.0, 999.0]
KILL_LIFT = 1.0
BUCKETS = [(0, 2), (3, 5), (6, 1_000_000)]


def shrink_scores(qF, Pshrink, TF, TL, Tseen, S):
    """Same as seen_inat_v95.member_scores but takes a pre-shrunk prototype."""
    out = torch.empty(qF.shape[0], S, device=dev)
    for i in range(0, qF.shape[0], 2000):
        e = qF[i:i + 2000]
        sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        out[i:i + 2000] = e @ Pshrink.t() + 2.0 * cmax + LAM * (e @ Tseen.t())
    return out


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
    n_train = torch.tensor([len(trby[c]) for c in seen], dtype=torch.float32).to(dev)
    n_train_gold = n_train.cpu()[yv]  # n_train of the TRUE class per val row, for bucketing
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    # per-member: raw prototype P, per-image (TF,TL) for cmax, query feats qF -- identical
    # construction to seen_inat_v95.member_scores / v98, so tau=0 must reproduce baseline.
    Zbase, Zshrink_by_tau = {}, {tau: {} for tau in TAU_GRID}
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

        base = shrink_scores(qF, P, TF, TL, Tseen, S)  # tau=0 path computed once, reused below
        Zbase[t] = zc(base)
        acc = 100 * (base.argmax(1).cpu() == yv).float().mean().item()
        print(f'  {t:16s} solo base val acc {acc:6.2f}', flush=True)

        for tau in TAU_GRID:
            if tau == 0.0:
                Zshrink_by_tau[tau][t] = Zbase[t]
                continue
            lam = (n_train / (n_train + tau)).unsqueeze(1)          # [S,1]
            Pshrink = F.normalize(lam * P + (1 - lam) * Tseen, dim=-1)
            sc = shrink_scores(qF, Pshrink, TF, TL, Tseen, S)
            Zshrink_by_tau[tau][t] = zc(sc)
            del sc
        del TF, TL, qF, base
        torch.cuda.empty_cache()

    def fused(zdict):
        return sum(BASE_W[t] * zdict[t] for t in BASE_W)

    print('\n=== overall + bucket accuracy by tau (holdout, val_seen) ===', flush=True)
    base_acc = 100 * (fused(Zbase).argmax(1).cpu() == yv).float().mean().item()
    print(f'sanity: recomputed deployed baseline (tau=0 path) = {base_acc:.2f} '
          f'(expect ~89.87)', flush=True)

    rows = {}
    for tau in TAU_GRID:
        pred = fused(Zshrink_by_tau[tau]).argmax(1).cpu()
        overall = 100 * (pred == yv).float().mean().item()
        buckets = {}
        for lo, hi in BUCKETS:
            m = (n_train_gold >= lo) & (n_train_gold <= hi)
            buckets[f'{lo}-{hi}'] = 100 * (pred[m] == yv[m]).float().mean().item()
        rows[tau] = {'overall': overall, **buckets}
        print(f'  tau={tau:<6} overall={overall:6.2f}  '
              f'[0,2]={buckets["0-2"]:6.2f}  [3,5]={buckets["3-5"]:6.2f}  '
              f'[6,+]={buckets["6-1000000"]:6.2f}', flush=True)

    best_tau = max(rows, key=lambda k: rows[k]['overall'])
    lift = rows[best_tau]['overall'] - base_acc
    strong_delta = rows[best_tau]['6-1000000'] - rows[0.0]['6-1000000']
    weak_delta = rows[best_tau]['0-2'] - rows[0.0]['0-2']
    verdict = (f'CLEARS internal bar — tau={best_tau} lift {lift:+.2f}pt, '
               f'6+ bucket delta {strong_delta:+.2f}pt, 0-2 bucket delta {weak_delta:+.2f}pt'
               if lift >= KILL_LIFT and strong_delta >= -0.3
               else 'DEAD — shrinkage does not lift holdout >= +1.0pt without hurting 6+ bucket')
    print(f'\nbase {base_acc:.2f}  best tau={best_tau} {rows[best_tau]["overall"]:.2f}  '
          f'lift {lift:+.2f}pt (kill >= +{KILL_LIFT}, 6+ bucket regression must be > -0.3pt)',
          flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    print(f'\nfor reference, distance to global bar: best {rows[best_tau]["overall"]:.2f} '
          f'vs v83 re-ranker 91.674 vs global kill 92.674 '
          f'(this script does not include the re-ranker -- see SCOPE NOTE in docstring)',
          flush=True)
    json.dump({'rows': rows, 'base': base_acc, 'best_tau': best_tau, 'lift': lift,
               'strong_bucket_delta': strong_delta, 'weak_bucket_delta': weak_delta,
               'verdict': verdict, 'kill_lift': KILL_LIFT},
              open(f'{OUT}/seen_js_shrink_v101.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_js_shrink_v101.json', flush=True)


if __name__ == '__main__':
    main()
