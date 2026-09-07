"""Unified Single-Pipeline Recognition Model (Learned, No DBNorm).

Complies with Competition Rule §2.1:
- Uniform single-pipeline rule over all evaluation images.
- Unbiased argmax over the entire 17,393-class space.
- Strictly inductive (per-image) without transductive batch normalization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.calibration.polynomial_calibrator import MultiModalPolynomialCalibrator
from src.calibration.novelty_logistic_gate import CalibratedNoveltyLogisticGate, extract_gating_features_tensor
from src.pipeline.multi_modal_fusion import MultiModalFusionScorer, score_bank_multiview_max, score_bank_topk_mean


class UnifiedSinglePipelineModel:
    """Full Learning-Based Open-Set Recognition Pipeline."""
    def __init__(
        self,
        classes: List[str],
        seen_classes: List[str],
        calibrators: MultiModalPolynomialCalibrator,
        fusion_scorer: MultiModalFusionScorer,
        gate: CalibratedNoveltyLogisticGate,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.classes = classes
        self.seen_classes = sorted(seen_classes)
        self.ci = {c: i for i, c in enumerate(classes)}
        self.s2i = {c: i for i, c in enumerate(self.seen_classes)}
        self.S = len(self.seen_classes)
        self.NCLS = len(classes)
        
        self.kept_idx = torch.tensor([self.ci[c] for c in self.seen_classes], device=device)
        self.other_idx = torch.tensor([i for i in range(self.NCLS) if classes[i] not in set(self.seen_classes)], device=device)
        
        self.calibrators = calibrators.to(device)
        self.fusion_scorer = fusion_scorer.to(device)
        self.gate = gate
        self.device = device

    def predict_batch(
        self,
        queries: Dict[str, torch.Tensor],
        seen_block: torch.Tensor,
        text_full: torch.Tensor,
        unseen_legs_raw: Dict[str, torch.Tensor],
        T_seen: torch.Tensor,
        T_unseen: torch.Tensor,
        I_unseen: Optional[torch.Tensor] = None
    ) -> List[str]:
        """Predicts class labels for a batch of query images.
        
        All operations are inductive per-sample without dbnorm.
        """
        N = seen_block.size(0)
        
        # 1. Calibrate unseen legs using Monotonic Degree-3 Polynomials
        calibrated_legs = self.calibrators(unseen_legs_raw)
        
        # 2. Fuse unseen legs
        unseen_scores = self.fusion_scorer(calibrated_legs) # [N, Cu]
        
        # 3. Extract novelty gating features & compute P(seen | x) via Logistic Regression
        q_gate = queries['ctftshift']
        X_gate = extract_gating_features_tensor(q_gate, T_seen, T_unseen, I_unseen)
        p_seen = self.gate.predict_proba(X_gate.cpu().numpy())
        route_seen = (p_seen >= self.gate.optimal_threshold)
        
        # 4. Predict
        pred_idx = torch.empty(N, dtype=torch.long, device=self.device)
        
        # Route to seen specialist
        idx_seen = torch.from_numpy(route_seen).nonzero(as_tuple=True)[0].to(self.device)
        if len(idx_seen) > 0:
            pred_idx[idx_seen] = self.kept_idx[seen_block[idx_seen].argmax(dim=1)]
            
        # Route to unseen specialist (inductive, strictly per-image argmax)
        idx_uns = torch.from_numpy(~route_seen).nonzero(as_tuple=True)[0].to(self.device)
        if len(idx_uns) > 0:
            pred_idx[idx_uns] = self.other_idx[unseen_scores[idx_uns].argmax(dim=1)]
            
        predictions = [self.classes[idx] for idx in pred_idx.tolist()]
        return predictions
