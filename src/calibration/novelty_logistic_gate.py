"""Novelty Logistic Regression Gate for Calibrated Novelty Detection.

Replaces heuristic threshold combinations (z1(seen) + 2*z1(text) >= f=0.60)
with a trained Logistic Regression model outputting sample-dependent probability P(seen | x).
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple


def extract_gating_features_tensor(
    q: torch.Tensor,
    T_seen: torch.Tensor,
    T_unseen: torch.Tensor,
    I_unseen: Optional[torch.Tensor] = None,
    aspect_ratios: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Extracts 9-10 discriminative features for query embeddings q [B, d].
    
    Args:
        q: Query embeddings [B, d]
        T_seen: Seen text / prototype embeddings [S, d]
        T_unseen: Unseen text embeddings [U, d]
        I_unseen: Optional unseen image prototype embeddings [U, d]
        aspect_ratios: Optional aspect ratio per image [B]
    """
    B = q.size(0)
    device = q.device
    
    # 1. Cosine similarity over seen classes
    sims_s = q @ T_seen.t() # [B, S]
    s_max_seen = sims_s.max(dim=1).values
    
    # 2. Top-1 vs Top-5 seen margin
    top5_seen = sims_s.topk(min(5, sims_s.size(1)), dim=1).values
    margin_1_5_seen = top5_seen[:, 0] - top5_seen[:, -1]
    
    # 3. Softmax entropy over seen classes (temperature = 0.07)
    probs_seen = F.softmax(sims_s / 0.07, dim=1)
    entropy_seen = -(probs_seen * (probs_seen + 1e-9).log()).sum(dim=1) / math.log(sims_s.size(1))
    
    # 4. Energy score over seen classes
    energy_seen = -0.07 * torch.logsumexp(sims_s / 0.07, dim=1)
    
    # 5. Cosine similarity over unseen text / image prototypes
    sims_u_txt = q @ T_unseen.t()
    if I_unseen is not None:
        sims_u_img = q @ I_unseen.t()
        sims_u = torch.maximum(sims_u_txt, sims_u_img)
    else:
        sims_u = sims_u_txt
    s_max_unseen = sims_u.max(dim=1).values
    
    # 6. Contrast margin: seen max - unseen max
    diff_seen_unseen = s_max_seen - s_max_unseen
    
    # 7. Confidence ratio: seen max / unseen max
    ratio_seen_unseen = s_max_seen / (s_max_unseen.clamp(min=1e-4))
    
    # 8. Top-2 margin on seen classes
    top2_seen = sims_s.topk(min(2, sims_s.size(1)), dim=1).values
    margin_1_2_seen = top2_seen[:, 0] - top2_seen[:, 1]
    
    # 9. Top-2 margin on unseen classes
    top2_unseen = sims_u.topk(min(2, sims_u.size(1)), dim=1).values
    margin_1_2_unseen = top2_unseen[:, 0] - top2_unseen[:, 1]
    
    feats = [
        s_max_seen,
        margin_1_5_seen,
        -entropy_seen,
        -energy_seen,
        s_max_unseen,
        diff_seen_unseen,
        ratio_seen_unseen,
        margin_1_2_seen,
        margin_1_2_unseen
    ]
    
    if aspect_ratios is not None:
        feats.append(aspect_ratios.to(device))
        
    return torch.stack(feats, dim=1)


class CalibratedNoveltyLogisticGate:
    """Logistic Regression Gate with feature scaling and risk-optimal threshold calibration."""
    def __init__(self, C: float = 1.0, optimal_threshold: float = 0.50):
        self.clf = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=1000)
        self.scaler = StandardScaler()
        self.optimal_threshold = optimal_threshold
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, w_seen: float = 0.5635, w_unseen: float = 0.4365):
        """Fits StandardScaler and Logistic Regression, then optimizes classification threshold."""
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.is_fitted = True
        
        # Optimize threshold over validation probabilities
        probs = self.clf.predict_proba(X_scaled)[:, 1] # P(y = 1) = P(seen)
        best_thr = 0.50
        best_score = -1.0
        
        for thr in np.linspace(0.20, 0.80, 61):
            preds = (probs >= thr).astype(int)
            # Weighted accuracy surrogate
            acc_seen = np.mean(preds[y == 1] == 1) if np.sum(y == 1) > 0 else 0.0
            acc_unseen = np.mean(preds[y == 0] == 0) if np.sum(y == 0) > 0 else 0.0
            score = w_seen * acc_seen + w_unseen * acc_unseen
            if score > best_score:
                best_score = score
                best_thr = thr
                
        self.optimal_threshold = float(best_thr)
        return self.optimal_threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(seen | x) for each sample in X."""
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)[:, 1]

    def predict_route(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Returns boolean array: True for seen route, False for unseen route."""
        thr = self.optimal_threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return probs >= thr

    def state_dict(self) -> dict:
        return {
            'coef': self.clf.coef_,
            'intercept': self.clf.intercept_,
            'classes': self.clf.classes_,
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'optimal_threshold': self.optimal_threshold
        }

    def load_state_dict(self, state: dict):
        self.clf.coef_ = state['coef']
        self.clf.intercept_ = state['intercept']
        self.clf.classes_ = state['classes']
        self.scaler.mean_ = state['scaler_mean']
        self.scaler.scale_ = state['scaler_scale']
        self.optimal_threshold = state['optimal_threshold']
        self.is_fitted = True
