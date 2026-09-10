"""v102: bootstrap-resampled prototype bagging for SEEN closed-set classification.

MECHANISM (genuinely new -- variance reduction on the estimator itself, not a
reweighting or a shrinkage of it): every class prototype P_c currently = a
single plain mean of its n_train training-image embeddings. For low-n classes
(n<=5, 13.7% of val rows per seen_inat_v97_bucket.py: n<=2 -> 59.91% acc,
n in [3,5] -> 74.39% acc, n>=6 -> 93.29% acc) that single mean is a
high-variance point estimate of the true class center. Classic bagging: draw
N_BOOT bootstrap resamples (with replacement, size n_c) of that class's OWN
training images, build one mean-prototype per resample, score the query
against EACH bootstrap prototype, and average the resulting SCORES (not the
prototypes) across resamples. Because cosine similarity is nonlinear,
mean-of-scores != score-of-mean-prototype (Jensen gap) -- bagging is not a
no-op even though every bootstrap draws from the exact same n_c vectors that
already feed the plain mean.

This is a DIFFERENT mechanism from James-Stein shrinkage (blends toward a
DIFFERENT external anchor, the taxon-text embedding) and from every other
lever in this repo (fusion reweighting, re-ranking, added evidence legs):
pure noise-averaging of the SAME few-shot estimator, no new anchor, no new
evidence source, no new information entering the pipeline at all -- so if it
works, the "new information compounds" transfer story does NOT apply and the
honest prior is closer to v98's reweighting (small/negative real transfer).
The counter-argument for testing it anyway: unlike v98 (which only re-sorted
marginal images within an already-saturated global ordering), this changes
the ORDERING WITHIN the diagnosed-broken low-n bucket specifically, where the
single-point-estimate prototype is admitted-noisy -- closer in spirit to
v101's shrinkage (isolated to the same tail) than to v98's global reweight.

ISOLATION: only the proto (P) term is bagged. cmax (already built from the
raw per-image TF/TL, a different non-mean aggregation) and the LAM*taxon
additive term are left EXACTLY as in the deployed 3-leg fusion. Classes with
n_train > BAG_MAX use the exact plain single-prototype term (byte-identical
to baseline -- no bagging, no RNG touches them) so any accuracy delta is
provably confined to the targeted low-n tail, not fusion noise elsewhere.

Kill bar (pre-registered before any number is seen): fused (ctftshift 1.0 /
ftshift 2.5 / fullft336shift 2.5) argmax closed-set accuracy on val_seen must
beat 91.674 (current best-in-repo, the v83 seen re-ranker) by >= +1.0pt,
i.e. >= 92.674%. Also reports the n_train-bucketed breakdown (same buckets as
seen_inat_v97_bucket.py) to show whether any gain is concentrated in the
0-2 / 3-5 image buckets as the mechanism predicts, a ctftshift-solo number
directly comparable to the 91.20% solo baseline, and an N_BOOT stability
sweep (bagging that only "helps" at one arbitrary N_BOOT and not nearby
values is bootstrap noise in the eval itself, not signal).

  conda activate onet && python research/proto_bagging_v102.py
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, MEMBERS, dev, holdout_split, load_train_embs, zc

LAM = 4.0
DEPLOYED_W = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
BAG_MAX = 5              # only bag classes with n_train <= this (the diagnosed low-n tail)
N_BOOT_GRID = [5, 10, 20]
DEFAULT_N_BOOT = 10
KILL_LIFT = 1.0
BEST_IN_REPO = 91.674
BUCKETS = [(0, 2), (3, 5), (6, 1_000_000)]
SEED = 0


def build_leg(t, train, seen, s2i, trby, val_seen):
    """class_feats[c]: (n_c,D) raw train-fold feats for class c (own images only,
    same rows that already back the deployed plain-mean prototype). TF/TL/P/qF
    are the exact deployed-formula pieces (member_scores in seen_inat_v95.py)."""
    idx, feats, _ = train[t]
    class_feats, TFl, TLl = [], [], []
    for c in seen:
        fl = [feats[idx[fn]] for fn in trby[c] if fn in idx]
        f = torch.stack(fl).to(dev)
        class_feats.append(f)
        TFl += fl
        TLl += [s2i[c]] * len(fl)
    TF = torch.stack(TFl).to(dev)
    TL = torch.tensor(TLl).to(dev)
    P = torch.stack([F.normalize(f.mean(0), dim=0) for f in class_feats]).to(dev)
    qF = torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev)
    return class_feats, TF, TL, P, qF


def bagged_proto_term(qF, class_feats, n_boot, rng):
    """(Nq,S): for n_c<=BAG_MAX, mean over n_boot bootstrap-resampled
    mean-prototype scores; for n_c>BAG_MAX, the single plain-mean prototype
    score (untouched, matches baseline exactly)."""
    S = len(class_feats)
    out = torch.zeros(qF.shape[0], S, device=dev)
    for c, f in enumerate(class_feats):
        n = f.shape[0]
        if n > BAG_MAX:
            P = F.normalize(f.mean(0, keepdim=True), dim=-1)
            out[:, c] = (qF @ P.t()).squeeze(1)
            continue
        idx = torch.from_numpy(rng.integers(0, n, size=(n_boot, n))).to(dev)  # (n_boot, n)
        samp = f[idx]                                    # (n_boot, n, D)
        Pb = F.normalize(samp.mean(1), dim=-1)            # (n_boot, D)
        out[:, c] = (qF @ Pb.t()).mean(1)                 # (Nq,)
    return out


def cmax_taxon(qF, TF, TL, Tseen, S):
    """The two unbagged terms: 2*cmax + LAM*taxon, chunked to bound memory."""
    out = torch.empty(qF.shape[0], S, device=dev)
    for i in range(0, qF.shape[0], 2000):
        e = qF[i:i + 2000]
        sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        out[i:i + 2000] = 2.0 * cmax + LAM * (e @ Tseen.t())
    return out


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
    n_bag_cls = int((n_train <= BAG_MAX).sum())
    print(f'val_seen rows {len(val_seen)} | seen classes {S} | '
          f'classes with n_train<=BAG_MAX({BAG_MAX}): {n_bag_cls}/{S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    legs = {}
    fixed_term = {}
    for t, _, _ in MEMBERS:
        if t not in DEPLOYED_W:
            continue
        class_feats, TF, TL, P, qF = build_leg(t, train, seen, s2i, trby, val_seen)
        legs[t] = (class_feats, qF, P)
        fixed_term[t] = cmax_taxon(qF, TF, TL, Tseen, S)

        plain = (qF @ P.t()) + fixed_term[t]
        acc_plain = 100 * (plain.argmax(1).cpu() == yv).float().mean().item()
        print(f'  {t:16s} plain solo (deployed formula) acc {acc_plain:6.2f}', flush=True)
        del TF, TL

    def fused(score_by_leg):
        f = sum(DEPLOYED_W[t] * zc(score_by_leg[t]) for t in DEPLOYED_W)
        return f, 100 * (f.argmax(1).cpu() == yv).float().mean().item()

    plain_scores = {t: legs[t][1] @ legs[t][2].t() + fixed_term[t] for t in DEPLOYED_W}
    f_plain, acc_plain_fused = fused(plain_scores)
    print(f'\ndeployed plain fusion (sanity check, expect ~89.87): {acc_plain_fused:.2f}', flush=True)

    print(f'\n=== N_BOOT sweep (fused accuracy, BAG_MAX={BAG_MAX}) ===', flush=True)
    rng = np.random.default_rng(SEED)
    sweep = {}
    bag_scores_default = None
    for n_boot in N_BOOT_GRID:
        bag_scores = {}
        for t in DEPLOYED_W:
            class_feats, qF, _ = legs[t]
            bag_scores[t] = bagged_proto_term(qF, class_feats, n_boot, rng) + fixed_term[t]
        f_bag, acc_bag_fused = fused(bag_scores)
        sweep[n_boot] = acc_bag_fused
        print(f'  N_BOOT={n_boot:<3} -> fused acc {acc_bag_fused:6.2f}  '
              f'(vs plain {acc_bag_fused - acc_plain_fused:+.2f})', flush=True)
        if n_boot == DEFAULT_N_BOOT:
            bag_scores_default, f_bag_default, acc_bag_default = bag_scores, f_bag, acc_bag_fused

    print('\n=== bucketed accuracy: plain vs bagged(N_BOOT=%d) -- targets the diagnosed low-n tail ===' % DEFAULT_N_BOOT, flush=True)
    for lo, hi in BUCKETS:
        mask = (n_train_gold >= lo) & (n_train_gold <= hi)
        n_rows = int(mask.sum())
        ap = 100 * (f_plain[mask].argmax(1).cpu() == yv[mask]).float().mean().item()
        ab = 100 * (f_bag_default[mask].argmax(1).cpu() == yv[mask]).float().mean().item()
        print(f'  n_train in [{lo},{hi}] ({n_rows} rows, {100*n_rows/len(yv):.1f}%): '
              f'plain {ap:6.2f} -> bagged {ab:6.2f}  ({ab-ap:+.2f})', flush=True)

    lift = acc_bag_default - BEST_IN_REPO
    verdict = (f'CLEARS +{lift:.2f}pt over {BEST_IN_REPO}' if lift >= KILL_LIFT
               else f'DEAD -- {lift:+.2f}pt vs +{KILL_LIFT} bar')
    print(f'\nfused bagged(N_BOOT={DEFAULT_N_BOOT}) {acc_bag_default:.2f} vs best-in-repo {BEST_IN_REPO} '
          f'-> lift {lift:+.2f}pt (kill >= +{KILL_LIFT})', flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'acc_plain_fused': acc_plain_fused, 'sweep': sweep,
               'acc_bag_default': acc_bag_default, 'best_in_repo': BEST_IN_REPO,
               'lift': lift, 'verdict': verdict, 'bag_max': BAG_MAX,
               'n_bag_classes': n_bag_cls, 'default_n_boot': DEFAULT_N_BOOT},
              open(f'{OUT}/proto_bagging_v102.json', 'w'), indent=2)
    print(f'wrote {OUT}/proto_bagging_v102.json', flush=True)


if __name__ == '__main__':
    main()
