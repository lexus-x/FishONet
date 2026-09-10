"""v104 Idea (e): PER-CLASS score calibration via leave-one-out self-similarity
(class-conditional z-scoring of match evidence) -- untouched mechanism.

THE GAP: every score in this pipeline (proto.q, cmax, taxon.q) sits on the SAME raw
cosine scale for every class, and the only normalization that ever runs -- zc() in
learned_gate_v77.py -- standardizes by ONE scalar mean/std computed over the WHOLE
score matrix (all queries x all classes at once). No per-CLASS recalibration exists
anywhere. A raw cosine of 0.55 to the prototype of a visually uniform species (every
training photo near-identical, self-similarity ~0.85) is a WEAK match; the same 0.55
to a polymorphic species' prototype (juvenile/adult/sex-dimorphic, self-similarity
~0.35) is a STRONG one. The fixed formula cannot tell these apart -- it has no notion
of "what does a genuine match to THIS class actually look like."

Distinct in mechanism from everything else in flight:
  - NOT whitening (seen_whiten_v102): that is a GLOBAL linear transform of the
    embedding DIMENSIONS, identical for every class, applied before any dot product.
    This never touches an embedding -- it recalibrates the SCALAR score axis, per
    class, after the dot product, using that class's own training-image self-
    consistency.
  - NOT James-Stein shrinkage (seen_js_shrink_v101 / seen_shrink_v101): those move
    the PROTOTYPE LOCATION toward a taxon-text anchor. This never moves a prototype
    -- it only rescales/recenters the score each class produces, using a statistic
    (intra-class cosine dispersion) shrinkage never computes.
  - NOT a per-image MoE weight (seen_moe_fusion_v100): that reweights the three LEGS
    per QUERY (row-wise). This recalibrates per CLASS, identically for every query
    (column-wise) -- an orthogonal axis.
  - NOT a trained probe/re-ranker (seen_linear_probe_v101, rerank_seen_v83): fully
    closed-form, no fit, no gradient, no candidate-axis learning -> no exposure to
    the v81 leak class (HANDOFF 2026-08-29: per-candidate learning on the pseudo-
    novel pool overstates holdout up to 66x; this has no learned candidate-axis
    parameter at all, so that failure mode cannot apply).

MECHANISM (per leg t in {ctftshift, ftshift, fullft336shift}, per seen class c):
  1. Leave-one-out self-similarity: for every training image i in trby[c],
         P_c^{-i} = normalize((sum_c - f_i) / (n_c - 1))
         s_i      = cos(f_i, P_c^{-i})
     "how well does a genuine same-class image match its own honest (leak-free,
     that image excluded) prototype." n_c<2 classes have no LOO estimate.
  2. self_mu_c = mean_i s_i;  raw self_var_c = var_i s_i  (n_c>=2 only).
     Variance is SHRUNK toward the global pooled LOO variance with a James-Stein-
     style weight on n_c -- reusing the shrinkage FORM, applied to a different
     target (a calibration statistic, not a prototype) than v101/v100:
         w_c   = n_c / (n_c + TAU_VAR)
         var_c = w_c * self_var_c + (1 - w_c) * global_var      # n_c<2 -> ~global_var
         sd_c  = sqrt(var_c).clamp(min=EPS)
     Classes with n_c<2 get self_mu_c = global_mu (no other option).
  3. Calibrated proto term replaces q.P_c inside the member score; cmax and the
     taxon term are left untouched so any effect is attributable to this one swap:
         MODE='raw'    : proto_cal = q.P_c                        (== baseline, sanity check)
         MODE='center'  : proto_cal = q.P_c - self_mu_c
         MODE='zscore'  : proto_cal = (q.P_c - self_mu_c) / sd_c
     sc_cal = ALPHA * proto_cal + 2.0*cmax + LAM*taxon   (2.0 / LAM=4.0 unchanged
     from the deployed formula; ALPHA is grid-searched because zscore units are
     O(1..5), not O(0.3..0.9) like the raw cosine they replace).

Pre-registered before any number is seen: MODE in {raw, center, zscore} x
ALPHA in {0.5, 1, 2, 4, 8}, evaluated on (a) ctftshift solo (must match ~91.20 at
MODE=raw/ALPHA=1, this task's quoted baseline) and (b) the deployed 3-leg fusion
(1.0/2.5/2.5, must match ~89.87 at MODE=raw/ALPHA=1) -- both are exact-formula
identities, not tolerances, so any mismatch is a bug, not noise.
KILL BAR: best (mode, alpha) on the 3-leg fusion lifts val_seen holdout argmax
accuracy >= +1.0pt over 91.674% (the v83 re-ranker, current repo best)
-> i.e. >= 92.674%.

Why more likely to transfer than v98's reweight: v98 only slides a FIXED per-query
ranking across a static linear combination -- HANDOFF's "re-weighting does not
compound" finding, because the relative ORDER of classes for a given query never
changes, only how far apart the moved threshold sits. This changes which classes
look "easy" vs "hard" to match IN THE FIRST PLACE (a tight-cluster class's high raw
score gets discounted, a diffuse class's mediocre raw score gets credited) -- new
per-class information entering the ranking itself, which is exactly the axis v77
found compounds through both routing AND conditional accuracy, not merely a slide.

  conda activate onet && python research/seen_selfcal_v104.py
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

LAM = 4.0
DEPLOYED = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
MODES = ['raw', 'center', 'zscore']
ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]
TAU_VAR = 5.0
EPS = 1e-3
KILL_LIFT_OVER_V83 = 1.0
V83_RERANK = 91.674
BEST_CTFT_SOLO = 91.20
DEPLOYED_3LEG = 89.87


def loo_self_sim(TFc):
    """TFc: (n, d) unit-normalized embeddings of one class's own training images.
    Returns per-image leave-one-out cosine to that image's own (honest) prototype,
    or None if n < 2 (no LOO estimate possible)."""
    n = TFc.shape[0]
    if n < 2:
        return None
    s = TFc.sum(0, keepdim=True)
    loo = F.normalize((s - TFc) / (n - 1), dim=-1)
    return (loo * TFc).sum(-1)


def calib_stats(seen, s2i, trby, idx, feats, S):
    """Per leg: self_mu (S,), self_sd (S,) with James-Stein-shrunk variance, n_arr (S,)."""
    self_mu = torch.zeros(S)
    self_var = torch.zeros(S)
    n_arr = torch.zeros(S)
    have = torch.zeros(S, dtype=torch.bool)
    all_s = []
    for c in seen:
        fns = [fn for fn in trby[c] if fn in idx]
        n_arr[s2i[c]] = len(fns)
        if len(fns) < 2:
            continue
        TFc = torch.stack([feats[idx[fn]] for fn in fns])
        s = loo_self_sim(TFc)
        self_mu[s2i[c]] = s.mean()
        self_var[s2i[c]] = s.var(unbiased=False)
        have[s2i[c]] = True
        all_s.append(s)
    global_s = torch.cat(all_s)
    global_mu, global_var = global_s.mean(), global_s.var(unbiased=False)
    self_mu[~have] = global_mu
    w = n_arr / (n_arr + TAU_VAR)
    var_shrunk = w * self_var + (1 - w) * global_var
    self_sd = var_shrunk.clamp(min=EPS ** 2).sqrt()
    return self_mu.to(dev), self_sd.to(dev)


def member_scores_cal(qF, P, TF, TL, Tseen, S, self_mu, self_sd, mode, alpha):
    """Deployed formula with q.P_c optionally per-class calibrated. mode='raw',
    alpha=1.0 must reproduce seen_inat_v95.member_scores bit-for-bit."""
    out = torch.empty(qF.shape[0], S, device=dev)
    for i in range(0, qF.shape[0], 2000):
        e = qF[i:i + 2000]
        sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        proto_raw = e @ P.t()
        if mode == 'raw':
            proto_term = proto_raw
        elif mode == 'center':
            proto_term = proto_raw - self_mu.unsqueeze(0)
        elif mode == 'zscore':
            proto_term = (proto_raw - self_mu.unsqueeze(0)) / self_sd.unsqueeze(0)
        else:
            raise ValueError(mode)
        out[i:i + 2000] = alpha * proto_term + 2.0 * cmax + LAM * (e @ Tseen.t())
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
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(),
                       dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zraw, Zcal = {}, {}   # Zcal[t][(mode, alpha)] -> zc(score)
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

        self_mu, self_sd = calib_stats(seen, s2i, trby, idx, feats, S)

        base = member_scores_cal(qF, P, TF, TL, Tseen, S, self_mu, self_sd, 'raw', 1.0)
        Zraw[t] = zc(base)
        acc = 100 * (base.argmax(1).cpu() == yv).float().mean().item()
        print(f'  {t:16s} raw/alpha=1 solo val acc {acc:6.2f} (sanity check vs deployed)', flush=True)

        for mode, alpha in itertools.product(MODES, ALPHAS):
            if mode == 'raw' and alpha != 1.0:
                continue   # raw ignores alpha's units by construction; skip redundant grid cells
            sc = member_scores_cal(qF, P, TF, TL, Tseen, S, self_mu, self_sd, mode, alpha)
            Zcal[(t, mode, alpha)] = zc(sc)
        del TF, TL, qF, base
        torch.cuda.empty_cache()

    def fused_acc(pick):
        """pick: dict leg -> (mode, alpha); combine with DEPLOYED weights."""
        f = sum(DEPLOYED[t] * Zcal[(t,) + pick[t]] for t in DEPLOYED)
        return 100 * (f.argmax(1).cpu() == yv).float().mean().item()

    sanity = fused_acc({t: ('raw', 1.0) for t in DEPLOYED})
    print(f'\nsanity: 3-leg fusion at raw/alpha=1 = {sanity:.2f} (must == deployed {DEPLOYED_3LEG})',
          flush=True)

    print('\n=== solo ctftshift, mode x alpha grid ===', flush=True)
    solo_rows = {}
    for mode, alpha in itertools.product(MODES, ALPHAS):
        if mode == 'raw' and alpha != 1.0:
            continue
        acc = 100 * (Zcal[('ctftshift', mode, alpha)].argmax(1).cpu() == yv).float().mean().item()
        solo_rows[f'{mode}/a={alpha}'] = acc
        print(f'  {mode:8s} alpha={alpha:<5} -> {acc:6.2f}', flush=True)

    print('\n=== 3-leg fusion, SAME (mode,alpha) applied to all three legs ===', flush=True)
    fusion_rows = {}
    for mode, alpha in itertools.product(MODES, ALPHAS):
        if mode == 'raw' and alpha != 1.0:
            continue
        pick = {t: (mode, alpha) for t in DEPLOYED}
        acc = fused_acc(pick)
        fusion_rows[f'{mode}/a={alpha}'] = acc
        print(f'  {mode:8s} alpha={alpha:<5} -> {acc:6.2f}', flush=True)

    best_key = max(fusion_rows, key=fusion_rows.get)
    best_acc = fusion_rows[best_key]
    lift_v83 = best_acc - V83_RERANK
    lift_deployed = best_acc - DEPLOYED_3LEG
    verdict = (f'CLEARS — {best_key} beats v83 rerank by {lift_v83:+.2f}pt'
               if lift_v83 >= KILL_LIFT_OVER_V83
               else 'DEAD — per-class self-similarity calibration does not lift holdout >= +1.0 over v83')
    print(f'\nbest 3-leg: {best_key} = {best_acc:.2f}  '
          f'vs v83-rerank {V83_RERANK:.3f} ({lift_v83:+.2f})  '
          f'vs deployed {DEPLOYED_3LEG:.2f} ({lift_deployed:+.2f})  '
          f'(kill >= +{KILL_LIFT_OVER_V83} over v83)', flush=True)
    print(f'VERDICT: {verdict}', flush=True)

    json.dump({'sanity_3leg_raw': sanity, 'solo_ctftshift': solo_rows, 'fusion': fusion_rows,
               'best': best_key, 'best_acc': best_acc, 'lift_over_v83': lift_v83,
               'lift_over_deployed': lift_deployed, 'verdict': verdict},
              open(f'{OUT}/seen_selfcal_v104.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_selfcal_v104.json', flush=True)


if __name__ == '__main__':
    main()
