"""Freeze the gate's batch-dependent pieces into per-image-safe constants.

Problem (flagged 2026-09-01): build_v109_genus_gamble.py computes
  Z = standardize(X_eval, uniform_weight)          # wz1 mean/std over the LIVE eval batch
  thr = topk(combined, 0.60 * N).values.min()       # rank cutoff over the LIVE eval batch
Both make one image's prediction depend on the other 35,664 images in the batch -- not
"fed one by one". Also a train/test mismatch bug: the gate was TRAINED on features
standardized against the HOLDOUT's own stats (learned_gate_v77.py main(), line ~333), but at
eval time standardize() is recomputed fresh on the eval batch instead of reusing those stats.

Fix: freeze wz1's (mu, sd) per gate feature from the holdout population (same one used to
train the deployed model), and freeze the SEEN_FRAC=0.60 operating threshold as an ABSOLUTE
score value calibrated once on that same holdout. Both become plain constants -> at eval,
combined[i] = model.predict_proba(frozen_standardize(X_eval[i])) and
route_seen[i] = combined[i] >= FROZEN_THR depend only on image i.

  conda activate onet && python research/compute_frozen_gate_constants.py
"""
import json
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learned_gate_v77 import FEATS, SEEN_FRAC, COEF_TPR, COEF_TNR, deploy_weights  # noqa: E402

OUT = 'outputs'
GATE_PKL = os.environ.get('GATE_PKL', f'{OUT}/learned_gate_v79.pkl')


def wz1_fit(v, w):
    mu = (w * v).sum()
    sd = torch.sqrt((w * (v - mu) ** 2).sum()).clamp(min=1e-6)
    return mu.item(), sd.item()


def standardize_frozen(X, mu_sd):
    return torch.stack([(X[:, j] - mu) / sd for j, (mu, sd) in enumerate(mu_sd)], dim=1)


def operating_point_abs(score, y, w):
    """Like learned_gate_v77.operating_point but also returns the ABSOLUTE score threshold."""
    o = torch.argsort(score, descending=True)
    cw = torch.cumsum(w[o], 0)
    k = int(torch.searchsorted(cw, torch.tensor(SEEN_FRAC * cw[-1].item())).item())
    thr = score[o[k]].item()
    route_seen = score >= thr
    ys = torch.tensor(y)
    tpr = (w * route_seen * (ys == 1)).sum().item() / (w * (ys == 1)).sum().item()
    tnr = (w * ~route_seen * (ys == 0)).sum().item() / (w * (ys == 0)).sum().item()
    proj = 100 * (COEF_TPR * tpr + COEF_TNR * tnr)
    return thr, tpr, tnr, proj


def main():
    d = torch.load(f'{OUT}/gate_feats_holdout_v77.pt', weights_only=False)
    X, y = d['X'], d['y']
    assert d['feats'] == FEATS
    gate = pickle.load(open(GATE_PKL, 'rb'))
    print(f'gate: {GATE_PKL} kind={gate["kind"]}')

    w = deploy_weights(y)
    mu_sd = [wz1_fit(X[:, j], w) for j in range(X.shape[1])]

    # (a) reference: live-batch standardize, exactly what build_v109 currently does (recomputed
    #     fresh from whatever population is passed in -- here the holdout itself, for comparison)
    from learned_gate_v77 import standardize as standardize_live
    Z_live = standardize_live(X, w)
    combined_live = torch.tensor(gate['model'].predict_proba(Z_live.numpy())[:, 1], dtype=torch.float32)
    thr_live, tpr_live, tnr_live, proj_live = operating_point_abs(combined_live, y, w)

    # (b) frozen: same holdout population, but standardize_frozen uses saved constants (bit
    #     identical to (a) here since the constants WERE fit on this exact holdout -- this just
    #     proves the freeze step introduces no drift by itself)
    Z_frozen = standardize_frozen(X, mu_sd)
    combined_frozen = torch.tensor(gate['model'].predict_proba(Z_frozen.numpy())[:, 1], dtype=torch.float32)
    thr_frozen, tpr_frozen, tnr_frozen, proj_frozen = operating_point_abs(combined_frozen, y, w)

    print(f'\nholdout check (live vs frozen standardize, same population -- should match closely):')
    print(f'  live:   thr={thr_live:.6f} TPR={100*tpr_live:.2f} TNR={100*tnr_live:.2f} proj={proj_live:.3f}')
    print(f'  frozen: thr={thr_frozen:.6f} TPR={100*tpr_frozen:.2f} TNR={100*tnr_frozen:.2f} proj={proj_frozen:.3f}')
    diff = (combined_live - combined_frozen).abs().max().item()
    print(f'  max |combined_live - combined_frozen| = {diff:.8f}')

    payload = {
        'mu_sd': mu_sd,
        'feats': FEATS,
        'thr': thr_frozen,
        'seen_frac': SEEN_FRAC,
        'holdout_tpr': tpr_frozen,
        'holdout_tnr': tnr_frozen,
        'holdout_proj': proj_frozen,
        'gate_pkl': GATE_PKL,
        'note': 'mu_sd and thr are frozen from the training-time holdout population only; '
                'no eval-batch statistic is used anywhere in this file.',
    }
    json.dump(payload, open(f'{OUT}/frozen_gate_v109.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/frozen_gate_v109.json  (thr={thr_frozen:.6f})')


if __name__ == '__main__':
    main()
