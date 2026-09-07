"""Learned Multi-Modal Fusion Module (No DBNorm).

Extracts similarity scores across all visual and textual legs, applies learned
monotonic degree-3 polynomial calibration, and combines them using learned fusion weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MultiModalFusionScorer(nn.Module):
    """Learned Multi-Modal Scorer with Monotonic Degree-3 Polynomial Calibration.
    
    Computes calibrated scores for unseen classes without transductive batch normalization.
    """
    def __init__(self, leg_names: List[str]):
        super().__init__()
        self.leg_names = leg_names
        # Modality fusion weights (learned via regularized regression/cross entropy)
        self.weights = nn.Parameter(torch.ones(len(leg_names), dtype=torch.float32), requires_grad=True)

    def forward(self, calibrated_legs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combines calibrated similarity legs into a single score tensor [Nq, Cu].
        
        Args:
            calibrated_legs: dict mapping leg_name -> tensor [Nq, Cu]
        """
        w_norm = F.softmax(self.weights, dim=0)
        total_score = None
        for i, name in enumerate(self.leg_names):
            if name in calibrated_legs:
                leg_val = calibrated_legs[name]
                weighted_leg = w_norm[i] * leg_val
                if total_score is None:
                    total_score = weighted_leg
                else:
                    total_score = total_score + weighted_leg
        return total_score


def score_bank_topk_mean(
    Q: torch.Tensor,
    bank: dict,
    other_list: list[int],
    topm: int = 4,
    device: str = 'cuda'
) -> torch.Tensor:
    """Computes Top-K mean cosine similarity over an image photo bank [Nq, Cu]."""
    Nq = Q.size(0)
    Cu = len(other_list)
    Sraw = torch.full((Nq, Cu), -1.0, device=device)
    
    for j, gidx in enumerate(other_list):
        photos = bank.get(gidx)
        if photos is None:
            continue
        if not isinstance(photos, torch.Tensor):
            photos = torch.stack(photos)
        if hasattr(photos, 'numel') and photos.numel() == 0:
            continue
        photos = F.normalize(photos.float(), dim=-1).to(device)
        sim = Q @ photos.t() # [Nq, n_photos]
        k = min(topm, sim.size(1))
        Sraw[:, j] = sim.topk(k, dim=1).values.mean(dim=1)
        
    return Sraw


def score_bank_multiview_max(
    view_tensors: list[torch.Tensor],
    bank: dict,
    other_list: list[int],
    topm: int = 4,
    device: str = 'cuda'
) -> torch.Tensor:
    """Computes max cosine similarity across multi-crop views against an image photo bank."""
    view_scores = [score_bank_topk_mean(vq, bank, other_list, topm=topm, device=device) for vq in view_tensors]
    return torch.stack(view_scores, dim=0).max(dim=0).values
