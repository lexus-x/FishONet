"""v93 Lever A: CORAL / quantile alignment of eval gate features to holdout
marginals — a transductive mechanism distinct from the dead v91 self-training
(v91 moved pseudo-LABELS; this moves the feature DISTRIBUTION space).

The gate's eval AUC (0.9300) sits ~5pt under its holdout AUC (0.981). If that
gap is first-order distribution shift in the 11 gate features, re-mapping each
eval marginal/covariance onto the holdout's should recover part of it.

Reference pipeline is bit-for-bit v91's (gate_feats_eval_v91.pt cache +
gate_feats_holdout_v90_norm.pt holdout, LogisticRegression C=100, deploy
weights). Harness validity gate: reference AUC must reproduce 0.9300 +/- 0.002
or everything aborts.

PRE-REGISTERED (fixed before any eval number was seen):
  ship variant  = coral_l05  (shrunk CORAL, lambda=0.5) — the conservative one
  kill bar      = ship-variant dAUC >= +0.010 vs reference
  coral_l10 / quantile are reported for information only and are NOT shippable
  (no eval-folder selection of variants; f-sweep output is diagnostic for the
  real-slot sweep, per user authorization logged in HANDOFF).

COMPLIANCE: eval-folder labels are used for MEASUREMENT only; no folder
identity enters any fit; the shipped artifact is a fixed feature transform +
the holdout-trained gate.

  conda activate onet && python research/gate_coral_v93.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import FEATS, W_SEEN, W_UNS, auc, deploy_weights, standardize

C = 100.0
SEEN_FRAC = 0.60
A_COND, B_COND = 0.87018, 0.29533
FEATS11 = [f for f in FEATS if f != 'b2l_max']
LAM_SHIP = 0.5          # pre-registered ship variant
KILL_DAUC = 0.010
REF_AUC_EXPECT = 0.9300


def coral(Ze: torch.Tensor, Zh: torch.Tensor, lam: float) -> torch.Tensor:
    """Whiten eval with its covariance, re-color with (lam * holdout cov +
    (1-lam) I), shift mean to holdout mean. lam=1: full moment match."""
    mu_e, mu_h = Ze.mean(0), Zh.mean(0)
    Ce = torch.cov(Ze.t()) + 1e-4 * torch.eye(Ze.shape[1])
    Ch = torch.cov(Zh.t()) + 1e-4 * torch.eye(Zh.shape[1])
    we_, Ve_ = torch.linalg.eigh(Ce)
    wh_, Vh_ = torch.linalg.eigh(Ch)
    Ce_mhalf = Ve_ @ torch.diag(we_.clamp(min=0).rsqrt()) @ Ve_.t()
    Ch_half = Vh_ @ torch.diag((lam * wh_.clamp(min=0)).sqrt() + (1 - lam)) @ Vh_.t()
    return (Ze - mu_e) @ Ce_mhalf @ Ch_half + mu_h


def quantile_map(Ze: torch.Tensor, Zh: torch.Tensor) -> torch.Tensor:
    """Per-feature empirical-quantile mapping of eval marginals onto holdout."""
    out = torch.empty_like(Ze)
    n_h = Zh.shape[0]
    Zh_s = torch.sort(Zh, dim=0).values
    for j in range(Ze.shape[1]):
        ranks = torch.argsort(torch.argsort(Ze[:, j], stable=True), stable=True).to(torch.float64)
        pos = (ranks / Ze.shape[0]).clamp(0, 1)
        idx = (pos * (n_h - 1)).round().long()
        out[:, j] = Zh_s[idx, j].float()
    return out


def op_point(score, y, frac=SEEN_FRAC):
    o = torch.argsort(score, descending=True)
    k = int(round(frac * len(score)))
    route = torch.zeros(len(score), dtype=torch.bool)
    route[o[:k]] = True
    ys = torch.tensor(y)
    tpr = (route & (ys == 1)).sum().item() / max((ys == 1).sum().item(), 1)
    tnr = (~route & (ys == 0)).sum().item() / max((ys == 0).sum().item(), 1)
    proj = 100 * (W_SEEN * A_COND * tpr + W_UNS * B_COND * tnr)
    return tpr, tnr, proj


def f_sweep(score, y):
    """Diagnostic only: projected overall across the f-grid (for the real-slot
    sweep plan). Not a selection mechanism for anything trained here."""
    out = {}
    for f in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        tpr, tnr, proj = op_point(score, y, frac=f)
        out[f] = round(proj, 3)
    return out


def main():
    torch.set_num_threads(8)
    de = torch.load(f'{OUT}/gate_feats_eval_v91.pt', weights_only=False)
    Xe, ye = de['X'], de['y']
    dh = torch.load(f'{OUT}/gate_feats_holdout_v90_norm.pt', weights_only=False)
    keep = [FEATS.index(f) for f in FEATS11]
    Xh, yh = dh['X'][:, keep], dh['y']
    print(f'eval {tuple(Xe.shape)}  holdout {tuple(Xh.shape)}', flush=True)

    wh = deploy_weights(yh)
    Zh = standardize(Xh, wh)
    we = torch.full((Xe.shape[0],), 1.0 / Xe.shape[0])
    Ze = standardize(Xe, we)

    ref = LogisticRegression(max_iter=2000, C=C)
    ref.fit(Zh.numpy(), yh, sample_weight=wh.numpy())

    res = {}

    def record(name, Z_aligned):
        s = torch.tensor(ref.predict_proba(Z_aligned.numpy())[:, 1], dtype=torch.float32)
        a = auc(s, ye, we)
        tpr, tnr, proj = op_point(s, ye)
        res[name] = {'auc': a, 'tpr': tpr, 'tnr': tnr, 'proj': proj,
                     'f_sweep': f_sweep(s, ye)}
        print(f'{name:12s} AUC {a:.4f}  TPR {100*tpr:6.2f}  TNR {100*tnr:6.2f}  '
              f'proj {proj:7.3f}  sweep {res[name]["f_sweep"]}', flush=True)
        return s

    print('=== harness validity ===', flush=True)
    record('reference', Ze)
    dauc_ref = abs(res['reference']['auc'] - REF_AUC_EXPECT)
    assert dauc_ref <= 0.002, f'harness drift {dauc_ref:.4f} — ABORT'
    print(f'harness OK (|dAUC| vs v91 {dauc_ref:.4f})', flush=True)

    print('\n=== pre-registered alignment variants ===', flush=True)
    record('coral_l10', coral(Ze, Zh, 1.0))
    record('coral_l05', coral(Ze, Zh, LAM_SHIP))
    record('quantile', quantile_map(Ze, Zh))

    dauc = res['coral_l05']['auc'] - res['reference']['auc']
    dproj = res['coral_l05']['proj'] - res['reference']['proj']
    verdict = ('CLEARS — build v93 with coral_l05 gate' if dauc >= KILL_DAUC
               else 'DEAD — distribution alignment does not move eval routing')
    print(f'\nship variant coral_l05: dAUC {dauc:+.4f}  dproj {dproj:+.3f}pt  '
          f'(kill: dAUC < +{KILL_DAUC})', flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'results': res, 'ship_variant': 'coral_l05', 'dauc': dauc,
               'dproj': dproj, 'verdict': verdict, 'feats': FEATS11,
               'kill_bar': KILL_DAUC},
              open(f'{OUT}/gate_coral_v93.json', 'w'), indent=2)
    print(f'wrote {OUT}/gate_coral_v93.json', flush=True)


if __name__ == '__main__':
    main()
