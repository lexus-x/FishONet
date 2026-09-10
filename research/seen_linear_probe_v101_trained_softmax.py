"""v101: trained linear probe (multinomial softmax) on frozen ctftshift embeddings,
as a genuinely different SCORING MECHANISM from the fixed proto+2*cmax+4*taxon heuristic
-- not another reweighting of it.

Why this targets the known failure mode (research/seen_inat_v97_bucket.py) instead of
uniformly reweighting already-saturated legs: the current classifier gives every class's
prototype the SAME geometric treatment (mean of its images) regardless of whether that
mean is backed by 1 image or 100. A trained nn.Linear(dim, S) decision boundary can use
the DISCRIMINATIVE structure between classes (which the fixed mean+cosine heuristic
cannot see) -- but with only 1-2 images for the weakest ~14% of classes, an unregularized
softmax over S~4636 classes will overfit those rows outright. So:
  1. init each class row of W from that class's own prototype (mean embedding) * SCALE0
     -- epoch-0 behaviour ~= the existing prototype classifier, not a cold start.
  2. per-class L2 anchors each row BACK toward its prototype init, weighted by
     tau / (n_train_c + tau) -- a James-Stein-style shrinkage penalty (Idea (a)'s formula,
     applied here as a regularizer on a TRAINED classifier instead of as a direct blend).
     Classes with 1-2 images stay pinned near the prototype (little/no risk of overfit);
     classes with 50+ images are free to move to wherever the data actually separates them.
  3. tested standalone (vs ctftshift solo 91.20) AND as a 4th zc()-normalized ADDITIVE leg
     fused into the deployed 3-leg base (vs 89.87 deployed, vs 91.674 v83 rerank -- Idea (d)).

Protocol matches research/seen_inat_v95.py / seen_reweight_v98.py exactly: SAME
holdout_split() from learned_gate_v77.py (no new split), single held-out val_seen,
small PRE-REGISTERED grid over (lambda, wp), printed before any number is cherry-picked.
This is a time-boxed triage check, not a final claim -- if it clears kill, re-verify with
the class-disjoint 5-fold CV harness (learned_gate_v77.main()'s NFOLD pattern) before it
is ever real-tested, exactly as v77/v83 did before shipping.

Why more likely to transfer than v98 (linear reweight, real-negative): v98 only slid
marginal images across a FIXED ordering of the same 3 legs -- it changes no per-class
information, so a_cond falls and gives ~90% of any holdout gain back on real (documented
gate-vs-reweight asymmetry, HANDOFF 2026-08-29). This probe adds genuinely NEW per-class
information (a discriminatively-trained boundary, not a cosine-to-mean heuristic) that the
fixed fusion has never had access to -- the same category of change (new information, not
reweighting) that is the only thing measured to compound on real so far (v77's gate).

Pass/fail bar: fused argmax accuracy on val_seen >= 92.674% (+1.0pt over 91.674, the
current best-in-repo real-shipped seen re-ranker's holdout number).

  conda activate onet && python research/seen_linear_probe_v101_trained_softmax.py
"""
from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import MEMBERS, dev, holdout_split, load_train_embs, zc
from seen_inat_v95 import member_scores

BASE_W = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
PROBE_LEG = 'ctftshift'          # strongest solo leg (91.20 holdout) -- the probe's input space
SCALE0 = 16.0                    # init logit scale for unit-norm cosine features
TAU = 5.0                        # shrinkage half-point (images) in the James-Stein coefficient
LAMBDA_GRID = [0.0, 10.0, 30.0, 100.0, 300.0]   # per-class-L2 strength, pre-registered
WP_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.5, 4.0]  # fusion weight for the probe leg, pre-registered
EPOCHS = 80
EVAL_EVERY = 10
BATCH = 4096
LR = 3e-3
WD_UNIFORM = 1e-4                # tiny uniform weight_decay in the optimizer, harmless safety net
V83_RERANK_HOLDOUT = 91.674      # current best-in-repo holdout (5-fold CV, rerank_seen_v83.py)
KILL = 1.0


def _selftest():
    """Pure-math sanity check on the shrinkage coefficient (no GPU needed)."""
    def coef(n, tau=TAU):
        return tau / (n + tau)
    assert abs(coef(0) - 1.0) < 1e-9, 'zero-image class must be fully anchored'
    assert coef(1000) < 0.01, 'well-supported class must be nearly unregularized'
    ns = [0, 1, 2, 5, 20, 1000]
    cs = [coef(n) for n in ns]
    assert all(cs[i] >= cs[i + 1] for i in range(len(cs) - 1)), 'coef must be monotonically decreasing in n'
    print('  _selftest ok: shrinkage coef monotonic, coef(0)=1, coef(1000)<0.01', flush=True)


def build_member(t, kept, s2i, trby, train, Tseen, S):
    """Deployed base score (proto + 2*cmax + 4*taxon) for member t -- identical loop to
    seen_inat_v95.py's main(), factored out so the probe leg (t=PROBE_LEG) can also hand
    back its raw per-image (features, labels, prototype, count) for training."""
    idx, feats, _ = train[t]
    P = torch.zeros(S, feats.shape[1])
    cnt = torch.zeros(S)
    TFl, TLl = [], []
    for c in kept:
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
    TL = torch.tensor(TLl, dtype=torch.long, device=dev)
    return P, TF, TL, cnt.to(dev)


def train_probe(Xtr, ytr, W_init, n_train_t, lam, Xval, yval, tau=TAU,
                 epochs=EPOCHS, batch=BATCH, lr=LR):
    """Multinomial softmax over S classes, per-class L2 anchored to W_init (the class
    prototype), scaled by tau/(n_train_c+tau). Tracks the best-val-accuracy checkpoint
    across epochs (train briefly, don't just take the last epoch)."""
    S, dim = W_init.shape
    W = torch.nn.Parameter(W_init.clone())
    log_prior = torch.log((n_train_t + 1.0) / (n_train_t.sum() + S))
    b = torch.nn.Parameter(log_prior.clone())
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=WD_UNIFORM)
    coef = (tau / (n_train_t + tau)).unsqueeze(1)   # [S,1], no grad needed (constant)
    n = Xtr.shape[0]
    best_acc, best_logits = -1.0, None
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            bidx = perm[i:i + batch]
            xb, yb = Xtr[bidx], ytr[bidx]
            logits = xb @ W.t() + b
            loss = F.cross_entropy(logits, yb)
            if lam > 0:
                loss = loss + lam * (coef * (W - W_init) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % EVAL_EVERY == 0 or ep == epochs:
            with torch.no_grad():
                vlogits = Xval @ W.t() + b
            vacc = 100 * (vlogits.argmax(1) == yval).float().mean().item()
            if vacc > best_acc:
                best_acc, best_logits = vacc, vlogits.clone()
    return best_acc, best_logits


def bucket_breakdown(logits, yval, cnt, files_cls_count, edges=((0, 2), (3, 5), (6, 10 ** 9))):
    """Accuracy split by n_train bucket for the true class of each val row -- the
    documented 59.9/74.4/93.3 failure mode this probe targets (seen_inat_v97_bucket.py)."""
    pred = logits.argmax(1)
    correct = (pred == yval)
    out = {}
    for lo, hi in edges:
        mask = (files_cls_count >= lo) & (files_cls_count <= hi)
        n = int(mask.sum().item())
        acc = 100 * correct[mask].float().mean().item() if n else float('nan')
        out[f'{lo}-{hi if hi < 10**8 else "inf"}'] = (acc, n)
    return out


def main():
    torch.set_num_threads(8)
    print('=== v101 self-test ===', flush=True)
    _selftest()

    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    kept, trby = sorted(sp['kept']), sp['trby']
    s2i = {c: i for i, c in enumerate(kept)}
    S = len(kept)
    kept_idx = torch.tensor([sp['ci'][c] for c in kept], device=dev)
    files = sp['val_files'][:sp['n_seen']]
    val_cls = sp['cls'][:sp['n_seen']]
    yval = torch.tensor([s2i[c] for c in val_cls], device=dev)
    print(f'val_seen rows {len(files)} | kept (seen) classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(),
                       dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zbase, probe_ctx = {}, None
    for t, _, _ in MEMBERS:
        if t not in BASE_W:
            continue
        P, TF, TL, cnt = build_member(t, kept, s2i, trby, train, Tseen, S)
        Qval = torch.stack([train[t][1][train[t][0][fn]] for fn in files]).to(dev)
        base = member_scores(Qval, P, TF, TL, Tseen, S)
        Zbase[t] = zc(base)
        acc = 100 * (base.argmax(1) == yval).float().mean().item()
        print(f'  {t:16s} solo val acc {acc:6.2f}', flush=True)
        if t == PROBE_LEG:
            probe_ctx = (P, TF, TL, cnt, Qval)
        else:
            del TF, TL
        del base
        torch.cuda.empty_cache()

    deployed_acc = 100 * (sum(BASE_W[t] * Zbase[t] for t in BASE_W).argmax(1) == yval).float().mean().item()
    print(f'\ndeployed 3-leg fusion val acc: {deployed_acc:.2f}  (repo baseline 89.87)', flush=True)

    P, Xtr, ytr, cnt, Xval = probe_ctx
    W_init = P * SCALE0
    print(f'\n=== pre-registered lambda grid, trained {PROBE_LEG} probe (tau={TAU}) ===', flush=True)
    probe_results = {}
    for lam in LAMBDA_GRID:
        acc, logits = train_probe(Xtr, ytr, W_init, cnt, lam, Xval, yval)
        probe_results[lam] = (acc, logits)
        print(f'  lambda={lam:<6.0f} solo probe val acc {acc:6.2f}', flush=True)

    best_lam = max(probe_results, key=lambda k: probe_results[k][0])
    best_solo_acc, best_logits = probe_results[best_lam]
    ctft_solo = 100 * (member_scores(Xval, P, Xtr, ytr, Tseen, S).argmax(1) == yval).float().mean().item()
    print(f'\nbest solo probe: lambda={best_lam} acc={best_solo_acc:.2f}  '
          f'(vs ctftshift prototype-leg solo {ctft_solo:.2f})', flush=True)

    # bucket breakdown (0-2 / 3-5 / 6+ n_train images per class) for the true class
    # of each val row -- targets the documented 59.9/74.4/93.3 gap directly.
    val_cnt = cnt[yval]
    print('\n=== bucket breakdown (n_train images per true class) ===', flush=True)
    base_fused = sum(BASE_W[t] * Zbase[t] for t in BASE_W)
    ctft_fixed_logits = member_scores(Xval, P, Xtr, ytr, Tseen, S)
    for name, logits in [('ctftshift solo (fixed)', ctft_fixed_logits),
                          (f'trained probe (lambda={best_lam})', best_logits),
                          ('deployed 3-leg fusion', base_fused)]:
        bb = bucket_breakdown(logits, yval, cnt, val_cnt)
        parts = ', '.join(f'{k}: {acc:5.2f} (n={n})' for k, (acc, n) in bb.items())
        print(f'  {name:32s} {parts}', flush=True)

    Zprobe = zc(best_logits)
    print(f'\n=== pre-registered wp grid, probe fused into deployed 3-leg base ===', flush=True)
    fuse_results = {}
    for wp in WP_GRID:
        fused = sum(BASE_W[t] * Zbase[t] for t in BASE_W) + wp * Zprobe
        acc = 100 * (fused.argmax(1) == yval).float().mean().item()
        fuse_results[wp] = acc
        print(f'  wp={wp:<5} fused val acc {acc:6.2f}', flush=True)

    best_wp = max(fuse_results, key=fuse_results.get)
    best_fused_acc = fuse_results[best_wp]
    lift_vs_deployed = best_fused_acc - deployed_acc
    lift_vs_v83 = best_fused_acc - V83_RERANK_HOLDOUT
    verdict = (f'CLEARS -- lambda={best_lam} wp={best_wp} beats v83 rerank holdout by '
               f'{lift_vs_v83:+.3f}pt (kill +{KILL:.1f}); escalate to 5-fold CV before real-testing'
               if lift_vs_v83 >= KILL else
               f'FAILS -- best fused {best_fused_acc:.3f} vs bar {V83_RERANK_HOLDOUT + KILL:.3f} '
               f'(lift over v83 rerank {lift_vs_v83:+.3f}pt, kill +{KILL:.1f})')
    print(f'\nbest fused: lambda={best_lam} wp={best_wp} acc={best_fused_acc:.3f}  '
          f'vs deployed {lift_vs_deployed:+.3f}  vs v83-rerank-holdout {lift_vs_v83:+.3f}', flush=True)
    print(f'VERDICT: {verdict}', flush=True)

    json.dump({
        'deployed_acc': deployed_acc, 'ctft_solo_acc': ctft_solo,
        'probe_lambda_grid': {str(k): v[0] for k, v in probe_results.items()},
        'best_lambda': best_lam, 'best_solo_probe_acc': best_solo_acc,
        'fuse_wp_grid': fuse_results, 'best_wp': best_wp, 'best_fused_acc': best_fused_acc,
        'v83_rerank_holdout': V83_RERANK_HOLDOUT, 'lift_vs_v83': lift_vs_v83, 'verdict': verdict,
    }, open(f'{OUT}/seen_linear_probe_v101.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/seen_linear_probe_v101.json', flush=True)


if __name__ == '__main__':
    main()
