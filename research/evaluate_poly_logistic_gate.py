"""Evaluate Polynomial Logistic Regression Gate (Degree > 1) on Holdout.

Compares:
1. Linear Baseline (Degree = 1)
2. Polynomial Degree = 2 (Quadratic + Pairwise Cross-Terms: x_i * x_j, x_i^2)
3. Polynomial Degree = 2 (Interaction Only)
4. Polynomial Degree = 3 (Cubic S-Curves + Multi-Way Interactions)

Evaluates on the exact same class-disjoint 5-fold CV with operating point pinned at f=0.60.
Completely isolated script - does NOT overwrite any production files or interfere with other jobs.
"""
from __future__ import annotations

import json
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUT = 'outputs'
D = 'data/dl'

SEEN_FRAC = 0.60
W_SEEN, W_UNS = 0.5635, 0.4365
A_COND, B_COND = 0.8726, 0.2761          # v56 real calibration
COEF_TPR = W_SEEN * A_COND               # 0.4917
COEF_TNR = W_UNS * B_COND                # 0.1205
V56_REAL = 51.61643067433057
NFOLD = 5

FEATS = ['sb_max', 'sb_margin', 'sb_lse_gap', 'm_ctftshift', 'm_ftshift', 'm_fullft336shift',
         'tm_ctft', 'tm_full', 'bank_max', 'bank_margin', 'b2f_max', 'b2l_max']


def deploy_weights(y):
    w = np.where(y == 1, W_SEEN / max((y == 1).sum(), 1), W_UNS / max((y == 0).sum(), 1))
    return torch.tensor(w, dtype=torch.float32)


def wz1(v, w):
    mu = (w * v).sum()
    sd = torch.sqrt((w * (v - mu) ** 2).sum()).clamp(min=1e-6)
    return (v - mu) / sd


def standardize(X, w):
    return torch.stack([wz1(X[:, j], w) for j in range(X.shape[1])], dim=1)


def operating_point(score, y, w):
    o = torch.argsort(score, descending=True)
    cw = torch.cumsum(w[o], 0)
    k = int(torch.searchsorted(cw, torch.tensor(SEEN_FRAC * cw[-1].item())).item())
    route_seen = torch.zeros_like(score, dtype=torch.bool)
    route_seen[o[:k + 1]] = True
    ys = torch.tensor(y)
    tpr = (w * route_seen * (ys == 1)).sum().item() / (w * (ys == 1)).sum().item()
    tnr = (w * ~route_seen * (ys == 0)).sum().item() / (w * (ys == 0)).sum().item()
    return tpr, tnr, 100 * (COEF_TPR * tpr + COEF_TNR * tnr)


def auc(score, y, w):
    o = torch.argsort(score)
    ys = torch.tensor(y)[o]
    ws = w[o]
    cum_neg = torch.cumsum(ws * (ys == 0), 0) - ws * (ys == 0) * 0.5
    num = (ws * (ys == 1) * cum_neg).sum().item()
    return num / ((ws * (ys == 1)).sum().item() * (ws * (ys == 0)).sum().item())


def main():
    cache = f'{OUT}/gate_feats_holdout_v77.pt'
    if not os.path.exists(cache):
        print(f"Error: {cache} not found.")
        return

    d = torch.load(cache, weights_only=False)
    X_raw, y, cls = d['X'], d['y'], d['cls']
    w = deploy_weights(y)
    N, D_raw = X_raw.shape
    print(f"Loaded cached holdout features: {N:,} samples x {D_raw} base features")

    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cls])

    # Base reference score (v56 heuristic)
    Z_base = standardize(X_raw, w)
    base_score = Z_base[:, FEATS.index('sb_max')] + 2.0 * Z_base[:, FEATS.index('tm_ctft')]
    b_tpr, b_tnr, b_proj = operating_point(base_score, y, w)
    b_auc = auc(base_score, y, w)

    print("\n" + "=" * 95)
    print("=== POLYNOMIAL LOGISTIC REGRESSION GATE (DEGREE > 1) 5-FOLD CV ===")
    print("=" * 95)
    print(f"{'Model':<38} | {'Dim':<5} | {'AUC':<8} | {'TPR%':<8} | {'TNR%':<8} | {'Proj%':<8} | {'vs v77':<8}")
    print("-" * 95)
    print(f"{'v56 Heuristic Gate (Baseline)':<38} | {2:<5} | {b_auc:8.4f} | {100*b_tpr:8.2f} | {100*b_tnr:8.2f} | {b_proj:8.3f} | {'-':<8}")

    experiments = [
        # (Name, degree, interaction_only, C_list)
        ("Linear (Degree 1, C=1 - v77 Deployed)", 1, False, [1.0]),
        ("Linear (Degree 1, C=100 - v79 Best)", 1, False, [100.0]),
        ("Poly Degree 2 (Full, C=1)", 2, False, [1.0]),
        ("Poly Degree 2 (Full, C=10)", 2, False, [10.0]),
        ("Poly Degree 2 (Full, C=100)", 2, False, [100.0]),
        ("Poly Degree 2 (Interaction-Only, C=10)", 2, True, [10.0]),
        ("Poly Degree 2 (Interaction-Only, C=100)", 2, True, [100.0]),
        ("Poly Degree 3 (Full, C=1)", 3, False, [1.0]),
        ("Poly Degree 3 (Full, C=10)", 3, False, [10.0]),
    ]

    v77_proj = 56.973  # v77 baseline proj

    results = []
    for exp_name, deg, interact, c_vals in experiments:
        for C in c_vals:
            if deg == 1:
                poly = None
                X_poly = X_raw.numpy()
            else:
                poly = PolynomialFeatures(degree=deg, interaction_only=interact, include_bias=False)
                X_poly = poly.fit_transform(X_raw.numpy())

            D_poly = X_poly.shape[1]
            oof = torch.zeros(len(y))

            for f in range(NFOLD):
                tr, va = (folds != f), (folds == f)
                w_tr = deploy_weights(y[tr])
                w_va = deploy_weights(y[va])

                # Standardize within fold to prevent leakage
                Z_tr = standardize(torch.tensor(X_poly[tr], dtype=torch.float32), w_tr).numpy()
                Z_va = standardize(torch.tensor(X_poly[va], dtype=torch.float32), w_va).numpy()

                clf = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=2500)
                clf.fit(Z_tr, y[tr], sample_weight=w_tr.numpy())
                oof[torch.tensor(va)] = torch.tensor(clf.predict_proba(Z_va)[:, 1], dtype=torch.float32)

            tpr, tnr, proj = operating_point(oof, y, w)
            score_auc = auc(oof, y, w)
            delta = proj - v77_proj

            label = f"{exp_name} (C={C})" if len(c_vals) > 1 else exp_name
            print(f"{label:<38} | {D_poly:<5} | {score_auc:8.4f} | {100*tpr:8.2f} | {100*tnr:8.2f} | {proj:8.3f} | {delta:+8.3f}")
            results.append({
                'name': label, 'degree': deg, 'dim': D_poly, 'C': C,
                'auc': score_auc, 'tpr': tpr, 'tnr': tnr, 'proj': proj, 'delta': delta
            })

    print("=" * 95)
    best_exp = max(results, key=lambda r: r['proj'])
    print(f"\n★ BEST CONFIGURATION: {best_exp['name']}")
    print(f"  ├─ Feature Dimension : {best_exp['dim']}D")
    print(f"  ├─ Validation ROC-AUC : {best_exp['auc']:.4f}")
    print(f"  ├─ Projection Score  : {best_exp['proj']:.3f}% (Delta vs v77: {best_exp['delta']:+.3f}pt)")
    print(f"  └─ Delta vs v56 Base : {best_exp['proj'] - b_proj:+.3f}pt")


if __name__ == '__main__':
    main()
