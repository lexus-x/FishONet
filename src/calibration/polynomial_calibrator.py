"""Monotonic 3rd-Degree Polynomial & Spearman Rank Correlation Calibrator.

Replaces transductive double normalization (dbnorm) with strictly inductive,
per-sample polynomial calibration mapping raw similarity scores to calibrated log-odds.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from typing import Tuple, Optional


class MonotonicDegree3Polynomial(nn.Module):
    """Monotonic Degree-3 Polynomial Calibrator:
    
    f(s) = c3 * (s - s0)^3 + c1 * (s - s0) + c0
    
    With constraints c3 >= 0, c1 >= 0 ensuring strictly non-negative derivative:
    f'(s) = 3 * c3 * (s - s0)^2 + c1 >= 0
    """
    def __init__(self, s0: float = 0.0):
        super().__init__()
        self.s0 = nn.Parameter(torch.tensor(s0, dtype=torch.float32), requires_grad=True)
        # Parameterize via softplus/exp to strictly guarantee monotonicity:
        self.log_c3 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        self.log_c1 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        self.c0 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)

    @property
    def c3(self) -> torch.Tensor:
        return F.softplus(self.log_c3)

    @property
    def c1(self) -> torch.Tensor:
        return F.softplus(self.log_c1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Applies monotonic degree-3 polynomial calibration to tensor s of any shape."""
        centered = s - self.s0
        return self.c3 * (centered ** 3) + self.c1 * centered + self.c0

    def fit_spearman_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        num_epochs: int = 200,
        lr: float = 0.05,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> float:
        """Fits polynomial parameters using a smooth rank surrogate (temperature-scaled softmax cross entropy)
        which directly optimizes Spearman rank correlation.
        """
        self.to(device)
        scores = scores.to(device)
        labels = labels.to(device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_loss = float('inf')
        best_state = None
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            calibrated = self.forward(scores)
            loss = F.cross_entropy(calibrated, labels)
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                
        if best_state is not None:
            self.load_state_dict(best_state)
            
        with torch.no_grad():
            calibrated_np = self.forward(scores).cpu().numpy()
            labels_np = labels.cpu().numpy()
            ranks_pred = np.argmax(calibrated_np, axis=-1)
            rho, _ = spearmanr(ranks_pred, labels_np)
            
        return float(rho) if not np.isnan(rho) else 0.0


class MultiModalPolynomialCalibrator(nn.Module):
    """Container for managing independent 3rd-degree polynomial calibrators across multiple modalities."""
    def __init__(self, modality_names: list[str]):
        super().__init__()
        self.calibrators = nn.ModuleDict({
            name: MonotonicDegree3Polynomial() for name in modality_names
        })

    def forward(self, modality_scores: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Calibrates each modality independently."""
        calibrated = {}
        for name, scores in modality_scores.items():
            if name in self.calibrators:
                calibrated[name] = self.calibrators[name](scores)
            else:
                calibrated[name] = scores
        return calibrated
