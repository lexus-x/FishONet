"""v100: per-image adaptive (mixture-of-experts) fusion weight for the 3 SEEN legs.

Every prior lever on this fusion (v98 grid reweight, v95 iNat leg, v83 re-ranker) kept
ONE global weight vector (or one global model) applied identically to every query image.
This is different in mechanism: a tiny per-image GATE reads each leg's OWN confidence on
THIS query (its top1-top2 margin and top1 score, already computed as a side-effect of
scoring -- no new embeddings, no new data) and outputs a per-image softmax weight over the
3 legs, so two images can end up trusting the legs in different proportions. That is
structurally impossible for a fixed w in sum_t w_t * zc(base_t).

Why this is more likely to transfer than v98's reweight (HANDOFF 2026-08-29, v77 finding):
"NEW INFORMATION in the gate compounds; RE-WEIGHTING does not." v98 slides the SAME fixed
ordering of images across a static combination -- it cannot fix a case where leg A is more
reliable on image X but leg B is more reliable on image Y, because both get the same w_t
regardless of X vs Y. The per-image gate below reads image-specific state at decision time
(this is the same category of feature -- margin, top1 score -- that made the v77 seen/novel
gate work, just applied one level down inside the seen fusion itself instead of only at the
seen-vs-novel routing decision).

Mechanism, concretely:
  1. Zbase[t] = zc(member_scores(...)) for t in {ctftshift, ftshift, fullft336shift} --
     IDENTICAL base formula to v98/v95 (proto + 2*cmax + 4*taxon), so this is a pure
     re-scoring-time change, no new evidence source.
  2. Per leg, per query: margin_t = top1_t - top2_t, max_t = top1_t (2 feats/leg, 6 total),
     z-scored. These are already implicitly computed by any consumer of Zbase (v77's own
     gate uses the identical sb_margin/sb_max shape of feature) -- genuinely free.
  3. A single nn.Linear(6, 3) + softmax "gate" maps those 6 features -> a per-image weight
     triple (g_ctft(q), g_ft(q), g_full(q)) that sums to 1. fused(q) = sum_t g_t(q)*Zbase[t][q].
  4. Trained end-to-end by cross-entropy on the TRUE fused argmax over the full S-class
     space (not a proxy), with class-disjoint 5-fold CV (same fold convention as
     learned_gate_v77.py) so the reported number is out-of-fold, never in-sample.
  5. Diagnostic: report std of g_t across rows. If it collapses to ~0 (constant weight per
     leg for every image), the gate found nothing to condition on and this mechanism failed
     to engage -- distinct from "engaged but didn't help enough to clear the bar."

Kill bar: OOF closed-set accuracy >= 92.674% (+1.0pt over the current best-in-repo SEEN
re-ranker, v83's 91.674%). Baselines reproduced in this script for a sanity cross-check:
deployed global fusion 89.87% (must match v98/v83's recall@1), ctftshift solo 91.20%.

  conda activate onet && python research/seen_moe_fusion_v100_d_per_image_moe_fusion_gate.py
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, MEMBERS, dev, holdout_split, load_train_embs, zc
from seen_inat_v95 import member_scores

LEGS = ['ctftshift', 'ftshift', 'fullft336shift']
DEPLOYED = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}   # baseline only
KILL_LIFT = 1.0
BASELINE_V83 = 91.674          # current best-in-repo SEEN accuracy (candidate re-ranker)
NFOLD = 5
EPOCHS = 400
LR = 0.05
WD = 1e-3


def build_leg_scores():
    """Same base-score construction as v98/v95: proto + 2*cmax + 4*taxon per leg,
    z-scored (zc) so all 3 legs sit on a comparable scale before any weighting."""
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby = sp['trby']
    val_seen = [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    n_train = torch.tensor([len(trby[c]) for c in seen], dtype=torch.float32)
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zbase = {}
    for t in LEGS:
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
    return Zbase, yv, S, n_train


def leg_feats(Zt):
    """margin (top1-top2) and top1, per leg -- the only per-image signal the gate sees."""
    top2 = Zt.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1], top2[:, 0]


class Gate(nn.Module):
    def __init__(self, n_feat, n_legs):
        super().__init__()
        self.lin = nn.Linear(n_feat, n_legs)
        self.log_tau = nn.Parameter(torch.zeros(1))

    def forward(self, G):
        g = F.softmax(self.lin(G), dim=1)
        return g, self.log_tau.exp()


def fuse(g, Bstack):
    """g: (N, L) softmax weights, Bstack: (L, N, S) per-leg score rows -> (N, S)."""
    return torch.einsum('nl,lns->ns', g, Bstack)


def train_fold(Gtr, Btr, ytr, Gva, Bva, seed):
    torch.manual_seed(seed)
    model = Gate(Gtr.shape[1], Btr.shape[0]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    ce = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        opt.zero_grad()
        g, tau = model(Gtr)
        loss = ce(tau * fuse(g, Btr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        g_va, tau = model(Gva)
        fused_va = tau * fuse(g_va, Bva)
    return fused_va, g_va.cpu()


def main():
    torch.set_num_threads(8)
    try:
        Zbase, yv, S, n_train = build_leg_scores()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise SystemExit('OOM building leg scores -- shared GPU is likely busy; retry later '
                          'or rerun with CUDA_VISIBLE_DEVICES="" to force CPU (slower).')

    N = yv.shape[0]
    fused_deploy = sum(DEPLOYED[t] * Zbase[t] for t in LEGS)
    base_acc = 100 * (fused_deploy.argmax(1).cpu() == yv).float().mean().item()
    print(f'\ndeployed global 1.0/2.5/2.5 val acc: {base_acc:.3f}  (sanity check vs v98: 89.87)',
          flush=True)

    feats = []
    for t in LEGS:
        m, mx = leg_feats(Zbase[t])
        feats += [m, mx]
    G = torch.stack(feats, dim=1)                      # (N, 6)
    G = (G - G.mean(0, keepdim=True)) / (G.std(0, keepdim=True) + 1e-6)
    Bstack = torch.stack([Zbase[t] for t in LEGS], dim=0)   # (3, N, S)

    # class-disjoint 5-fold CV, same convention as learned_gate_v77.py
    uniq = sorted(set(yv.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in yv.tolist()])

    oof_fused = torch.zeros(N, S, device=dev)
    oof_g = torch.zeros(N, len(LEGS))
    for f in range(NFOLD):
        tr = torch.tensor(folds != f)
        va = torch.tensor(folds == f)
        fused_va, g_va = train_fold(G[tr], Bstack[:, tr, :], yv[tr].to(dev),
                                     G[va], Bstack[:, va, :], seed=1000 + f)
        oof_fused[va] = fused_va
        oof_g[va] = g_va
        print(f'  fold {f}: train={int(tr.sum())} val={int(va.sum())}', flush=True)

    oof_acc = 100 * (oof_fused.argmax(1).cpu() == yv).float().mean().item()
    lift = oof_acc - BASELINE_V83
    print('\n=== per-image adaptive gate, OOF closed-set accuracy ===', flush=True)
    print(f'deployed global fusion : {base_acc:.3f}', flush=True)
    print(f'v83 re-ranker (bar)    : {BASELINE_V83:.3f}', flush=True)
    print(f'MoE per-image gate OOF : {oof_acc:.3f}  (lift vs v83 bar {lift:+.3f}, '
          f'kill >= +{KILL_LIFT})', flush=True)

    for i, t in enumerate(LEGS):
        col = oof_g[:, i]
        print(f'  g_{t:16s} mean {col.mean():.3f}  std {col.std():.4f}  '
              f'(std~0 => gate collapsed to a constant, i.e. found nothing to condition on)',
              flush=True)

    verdict = ('CLEARS' if lift >= KILL_LIFT else 'DEAD -- per-image gate does not lift '
               f'OOF accuracy >= +{KILL_LIFT} over the v83 bar')
    print(f'\nVERDICT: {verdict}', flush=True)

    # bucket breakdown by n_train per class (same buckets as seen_inat_v97_bucket.py)
    BUCKETS = [(0, 2), (3, 5), (6, 1_000_000)]
    row_ntrain = n_train[yv]   # (N,) n_train of the TRUE class for each val row
    bucket_rows = {}
    pred_deploy = fused_deploy.argmax(1).cpu()
    pred_oof = oof_fused.argmax(1).cpu()
    print('\n=== bucket breakdown (n_train per true class) ===', flush=True)
    print(f'{"bucket":10s} {"n":>6s} {"deployed%":>10s} {"MoE_OOF%":>10s} {"delta":>8s}', flush=True)
    for lo, hi in BUCKETS:
        m = (row_ntrain >= lo) & (row_ntrain <= hi)
        if m.sum() == 0:
            continue
        acc_d = 100 * (pred_deploy[m] == yv[m]).float().mean().item()
        acc_o = 100 * (pred_oof[m] == yv[m]).float().mean().item()
        key = f'{lo}-{hi if hi < 1000 else "+"}'
        bucket_rows[key] = {'n': int(m.sum()), 'deployed_acc': acc_d, 'moe_oof_acc': acc_o,
                             'delta': acc_o - acc_d}
        print(f'{key:10s} {int(m.sum()):6d} {acc_d:10.2f} {acc_o:10.2f} {acc_o-acc_d:+8.2f}',
              flush=True)
    json.dump({'base_acc': base_acc, 'v83_bar': BASELINE_V83, 'oof_acc': oof_acc,
               'lift': lift, 'verdict': verdict,
               'gate_weight_std': {t: float(oof_g[:, i].std()) for i, t in enumerate(LEGS)},
               'buckets': bucket_rows},
              open(f'{OUT}/seen_moe_fusion_v100.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/seen_moe_fusion_v100.json', flush=True)


if __name__ == '__main__':
    main()
