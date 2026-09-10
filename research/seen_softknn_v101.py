"""v101: soft distance-weighted top-k pooling -- a THIRD aggregation strategy
strictly between the two the deployed fusion already uses.

MECHANISM (genuinely new, not a reweighting of the 3-leg fusion): the deployed
per-leg score is  proto.q + 2*cmax + 4*taxon.q.  `proto` averages ALL of a
class's training images with UNIFORM weight (k=infinity, hard-blind to which
images are close to the query). `cmax` uses the SINGLE nearest training image
(k=1, throws away every other image no matter how relevant). Nothing in the
current pipeline uses "the k=3-8 nearest images, weighted by how close each
one is" -- a standard soft-kNN retrieval pooling that sits strictly between
those two extremes and has never been tried here.

    softknn_c(q) = sum_{i in top-k(q, class c)} softmax(sim(q,i)/tau) * sim(q,i)

As tau->0, k=1 this collapses to exactly cmax (sanity check, built into the
grid below: k=1 must reproduce the cmax-based baseline bit-for-bit regardless
of tau). As k grows and tau grows it drifts toward a similarity-weighted mean
over more neighbours -- closer to proto's territory but still distance-aware
and still bounded to the actual nearest evidence, unlike proto's flat average.
For the 58.3% of classes with n_train<=2 (seen_inat_v97_bucket.py), k>=3
naturally uses ALL available images (nothing to discard) so softknn degrades
gracefully toward exactly what proto/cmax already do there -- the mechanism
only *activates new information* for classes with >=3 images, i.e. it targets
the well-supported bucket's UNTAPPED middle ground (using image 2 and 3
instead of throwing them away) rather than trying to fix the un-fixable
1-image tail.

IMPLEMENTATION: computing per-class top-k without a Python loop over ~4,636
classes uses an iterative masked scatter_reduce(reduce='amax'): find the
per-(query,class) max via scatter_reduce (same primitive the deployed cmax
already uses), record it, mask out every training image that hit that max,
repeat k_max times. This costs k_max passes of the same scatter_reduce cmax
already pays once -- cheap, GPU-vectorized, no per-class loop. The k_max=8
value stack is computed ONCE per leg; every (k,tau) grid cell below just
slices and re-softmaxes that cached stack (no recomputation), so the grid is
nearly free after the one-time k_max pass.

TWO TESTS (both requested for this idea, both isolate softknn from the other
two known-dead levers -- leg reweighting and re-ranking):
  Track 1 -- DROP-IN REPLACEMENT (standalone new aggregation): swap only the
    2*cmax term for 2*softknn(k,tau) in the deployed per-leg formula; proto
    and the LAM*taxon term untouched. Isolates "does this aggregation beat
    single-nearest-neighbor" from every other axis.
  Track 2 -- ADDITIVE ENSEMBLE (idea d): add the best softknn leg as a 4th
    zc()-normalized term on top of the UNMODIFIED deployed base (cmax intact),
    grid over an additive weight wi, same grid shape as seen_inat_v95.py's
    proven-working wi sweep. Different paradigms make different mistakes;
    additive combination is the pattern that already compounded once (v77).

WHY THIS TIME MIGHT TRANSFER BETTER THAN v98 (uniform leg reweighting, which
was real-NEGATIVE despite +1.34pt holdout): v98 only re-sorted marginal cases
within an already-saturated 3-way ordering built from the SAME three fixed
aggregation heuristics -- no new information entered the score, so a_cond
fell and gave the holdout gain back on real. softknn is a different
computation over the SAME raw training images (not a different mixture of
the same three already-agreeing legs) -- it can rank a query against class c
using evidence (images 2 and 3) that proto blurs into the mean and cmax
discards entirely. A win here is a genuinely different geometric read of the
existing evidence, closer in kind to v77 (new evidence -> new ranking) than
to v98/v99 (same evidence, different mixture weight -> same ranking).

SCOPE NOTE: the mandated global kill bar is >=92.674% holdout (beating the
v83 re-ranker's 91.674%, which already layers per-candidate re-ranking on TOP
of the base 3-leg fusion). This script tests the softknn ingredient against
the base 3-leg fusion it feeds (recomputed in-script, should equal ~89.87%
for Track 1's tau=0-equivalent / k=1 sanity row) -- the correct apples-to-
apples comparison for THIS mechanism, same convention as seen_js_shrink_v101.
Internal pass bar: best (k,tau) overall lift over the in-script deployed
baseline >= +1.0pt, AND the 6+ bucket (86.3% of rows, currently 93.29%) does
not drop by more than 0.3pt. A pass is NECESSARY but not SUFFICIENT for the
global bar -- the follow-up (not run here, out of the no-new-training/4-min
budget) is swapping cmax for softknn inside rerank_seen_v83.py's cmax-derived
features (or adding softknn_max/softknn_margin as two NEW re-ranker features)
and re-measuring the full re-ranked pipeline against 92.674%.

  conda activate onet && python research/seen_softknn_v101.py
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
K_MAX = 8
K_GRID = [1, 3, 5, 8]                      # k=1 must reproduce cmax exactly (sanity row)
TAU_GRID = [0.02, 0.05, 0.1, 0.2, 0.4]      # cosine-sim scale; smaller = closer to hard max
WI_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.5]   # Track 2 additive weight, same shape as v95
KILL_LIFT = 1.0
BUCKETS = [(0, 2), (3, 5), (6, 1_000_000)]
QCHUNK = 400   # query chunk for the k_max-pass masked-scatter stack (memory-bounded)


def softknn_stack(qF, TF, TL, S, k_max):
    """(batch, S, k_max) per-class top-k_max raw cosine values, most-similar first.
    Iterative masked scatter_reduce(amax): same primitive the deployed cmax uses,
    called k_max times, masking out each round's per-class argmax before the next.
    Classes with fewer than k_max training images pad with -1e9 in the remaining
    slots (softmax later gives those slots ~0 weight -- underflows to exactly 0.0
    in float32 well before the -1e9/tau range, so no NaN/inf risk)."""
    out = torch.empty(qF.shape[0], S, k_max, device=dev)
    for i in range(0, qF.shape[0], QCHUNK):
        e = qF[i:i + QCHUNK]
        sim = e @ TF.t()                                  # (b, N_images)
        idx_full = TL.unsqueeze(0).expand(e.shape[0], -1)  # view, no extra alloc
        for kk in range(k_max):
            cmax = torch.full((e.shape[0], S), -1e9, device=dev)
            cmax.scatter_reduce_(1, idx_full, sim, reduce='amax')
            out[i:i + QCHUNK, :, kk] = cmax
            hit = sim >= cmax.gather(1, idx_full) - 1e-6
            sim = sim.masked_fill(hit, -1e9)
    return out


def softknn_score(V, k, tau):
    """V: (batch, S, k_max) cached stack -> (batch, S) pooled score for this (k,tau)."""
    Vk = V[:, :, :k]
    w = F.softmax(Vk / tau, dim=-1)
    return (w * Vk).sum(-1)


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
    n_train = torch.tensor([len(trby[c]) for c in seen], dtype=torch.float32)
    n_train_gold = n_train[yv]
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zbase, Vstack = {}, {}
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

        cmax = torch.full((qF.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(qF.shape[0], -1), qF @ TF.t(), reduce='amax')
        base = qF @ P.t() + 2.0 * cmax + LAM * (qF @ Tseen.t())
        Zbase[t] = zc(base)
        acc = 100 * (base.argmax(1).cpu() == yv).float().mean().item()
        print(f'  {t:16s} solo base(cmax) val acc {acc:6.2f}', flush=True)

        Vstack[t] = softknn_stack(qF, TF, TL, S, K_MAX).cpu()   # (batch,S,K_MAX), off GPU to save mem
        del TF, TL, qF, base, cmax
        torch.cuda.empty_cache()
        print(f'  {t:16s} softknn stack cached {tuple(Vstack[t].shape)}', flush=True)

    def fused(zdict):
        return sum(BASE_W[t] * zdict[t] for t in BASE_W)

    base_acc = 100 * (fused(Zbase).argmax(1).cpu() == yv).float().mean().item()
    print(f'\nsanity: recomputed deployed baseline = {base_acc:.2f} (expect ~89.87)', flush=True)

    # ---------------- Track 1: drop-in replacement (2*cmax -> 2*softknn(k,tau)) ----------
    print('\n=== Track 1: replace cmax with softknn(k,tau), overall + bucket acc ===', flush=True)
    # proto/taxon terms are cheap (no k-pass) -- recompute once per leg, then combine with the
    # already-cached softknn stack per (k,tau) grid cell below (no second expensive stack pass).
    Proto_t, Taxon_t = {}, {}
    for t, _, _ in MEMBERS:
        if t not in BASE_W:
            continue
        idx, feats, _ = train[t]
        P = torch.zeros(S, feats.shape[1])
        cnt = torch.zeros(S)
        for c in seen:
            for fn in trby[c]:
                if fn not in idx:
                    continue
                f = feats[idx[fn]]
                P[s2i[c]] += f
                cnt[s2i[c]] += 1
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
        qF = torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev)
        Proto_t[t] = (qF @ P.t()).cpu()
        Taxon_t[t] = (qF @ Tseen.t()).cpu()
        del qF
        torch.cuda.empty_cache()

    rows1 = {}
    for k in K_GRID:
        for tau in TAU_GRID:
            if k == 1 and tau != TAU_GRID[0]:
                continue
            Zrep = {}
            for t in BASE_W:
                knn = softknn_score(Vstack[t].to(dev), k, tau).cpu()
                rep = Proto_t[t] + 2.0 * knn + LAM * Taxon_t[t]
                Zrep[t] = zc(rep)
            pred = fused(Zrep).argmax(1)
            overall = 100 * (pred == yv).float().mean().item()
            buckets = {}
            for lo, hi in BUCKETS:
                m = (n_train_gold >= lo) & (n_train_gold <= hi)
                buckets[f'{lo}-{hi}'] = 100 * (pred[m] == yv[m]).float().mean().item()
            key = f'k{k}_tau{tau}'
            rows1[key] = {'overall': overall, **buckets}
            print(f'  {key:14s} overall={overall:6.2f}  [0,2]={buckets["0-2"]:6.2f}  '
                  f'[3,5]={buckets["3-5"]:6.2f}  [6,+]={buckets["6-1000000"]:6.2f}', flush=True)

    best1 = max(rows1, key=lambda kk: rows1[kk]['overall'])
    lift1 = rows1[best1]['overall'] - base_acc
    strong1 = rows1[best1]['6-1000000'] - rows1[f'k1_tau{TAU_GRID[0]}']['6-1000000']
    print(f'\nTrack 1 best {best1} overall={rows1[best1]["overall"]:.2f} lift={lift1:+.2f}pt '
          f'6+ bucket delta={strong1:+.2f}pt', flush=True)

    # ---------------- Track 2: additive ensemble on top of UNMODIFIED deployed base -------
    print('\n=== Track 2: deployed base + wi * softknn leg (best k,tau from Track 1) ===', flush=True)
    bk, btau = best1.split('_')
    bk = int(bk[1:]); btau = float(btau[3:])
    Zknn = {t: zc(softknn_score(Vstack[t].to(dev), bk, btau)) for t in BASE_W}
    rows2 = {}
    for wi in WI_GRID:
        f = sum(BASE_W[t] * (Zbase[t] + wi * Zknn[t]) for t in BASE_W)
        rows2[f'wi{wi}'] = 100 * (f.argmax(1).cpu() == yv).float().mean().item()
        print(f'  wi={wi:<5} -> {rows2[f"wi{wi}"]:6.2f}', flush=True)
    best2 = max(rows2, key=rows2.get)
    lift2 = rows2[best2] - base_acc

    best_overall = max(lift1, lift2)
    winner = 'Track1(replace)' if lift1 >= lift2 else 'Track2(additive)'
    verdict = (f'CLEARS internal bar — {winner} lift {best_overall:+.2f}pt (Track1 best={best1} '
               f'{rows1[best1]["overall"]:.2f}, Track2 best={best2} {rows2[best2]:.2f})'
               if best_overall >= KILL_LIFT and strong1 >= -0.3
               else 'DEAD — softknn pooling does not lift holdout >= +1.0pt without hurting 6+ bucket')
    print(f'\nbase {base_acc:.2f}  Track1 best {rows1[best1]["overall"]:.2f} ({lift1:+.2f}pt)  '
          f'Track2 best {rows2[best2]:.2f} ({lift2:+.2f}pt)  (kill >= +{KILL_LIFT}, '
          f'6+ bucket regression must be > -0.3pt)', flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    print(f'\nfor reference, distance to global bar: best {max(rows1[best1]["overall"], rows2[best2]):.2f} '
          f'vs v83 re-ranker 91.674 vs global kill 92.674 '
          f'(this script does not include the re-ranker -- see SCOPE NOTE in docstring)', flush=True)

    json.dump({'base': base_acc, 'track1': rows1, 'track1_best': best1, 'track1_lift': lift1,
               'track2': rows2, 'track2_best': best2, 'track2_lift': lift2,
               'strong_bucket_delta_track1': strong1, 'verdict': verdict, 'kill_lift': KILL_LIFT},
              open(f'{OUT}/seen_softknn_v101.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_softknn_v101.json', flush=True)


if __name__ == '__main__':
    main()
