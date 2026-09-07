"""Holdout Benchmark: v60 (Simple Logistic + Global Poly) vs v61 (Soft Marginal Probabilistic + Query-Centered Poly + Dynamic Attention).

Compares:
1. Baseline v56 (heuristic magic numbers + transductive dbnorm)
2. Proposal v60 (hard logistic gate + global poly + static weights)
3. Superior v61 (soft probabilistic marginal fusion + query-centered degree-3 poly + dynamic confidence fusion)
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.calibration.polynomial_calibrator import MonotonicDegree3Polynomial
from src.calibration.novelty_logistic_gate import CalibratedNoveltyLogisticGate, extract_gating_features_tensor
from src.pipeline.multi_modal_fusion import score_bank_topk_mean

OUT = os.path.join(ROOT, 'outputs')
DATA = os.path.join(ROOT, 'data', 'dl')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_emb(path: str):
    d = torch.load(path, map_location='cpu', weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def main():
    print("=" * 80)
    print("=== Comparative Holdout Evaluation: v56 vs v60 vs v61 ===")
    print("=" * 80)
    torch.set_num_threads(8)

    classes = list(pickle.load(open(os.path.join(DATA, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(os.path.join(DATA, 'label_train.json'), 'r'))

    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), map_location='cpu', weights_only=False)['emb_taxon'].float(), dim=-1).to(DEV)
    Ttb = F.normalize(torch.load(os.path.join(OUT, 'text_emb_taxabind_taxctx.pt'), map_location='cpu', weights_only=False)['emb_taxctx'].float(), dim=-1).to(DEV)

    idx_ctft, feats_ctft, files_ctft = load_emb(os.path.join(OUT, 'emb_train_ctftshift.pt'))
    idx_b2, feats_b2, files_b2 = load_emb(os.path.join(OUT, 'emb_train_bioclip2.pt'))
    idx_336, feats_336, _ = load_emb(os.path.join(OUT, 'emb_train_fullft336shift.pt'))
    idx_tb, feats_tb, _ = load_emb(os.path.join(OUT, 'emb_train_taxabind.pt'))
    idx_b2l, feats_b2l, _ = load_emb(os.path.join(OUT, 'emb_train_bioclip2_lora_v2.pt'))

    common_files = [fn for fn in files_ctft if fn in idx_b2 and fn in idx_336 and fn in idx_tb and fn in idx_b2l and fn in lab and lab[fn] in ci]
    by = defaultdict(list)
    for fn in common_files:
        by[lab[fn]].append(fn)

    seen = sorted(by.keys())
    order = sorted(seen, key=lambda c: len(by[c]))

    n_pseudo = int(len(seen) * 0.20)
    pseudo_classes = set(order[:n_pseudo])
    kept_classes = [c for c in seen if c not in pseudo_classes]

    kept_idx = torch.tensor([ci[c] for c in kept_classes], device=DEV)
    pseudo_idx = torch.tensor([ci[c] for c in pseudo_classes], device=DEV)
    unseen_idx = torch.tensor([i for i, c in enumerate(classes) if c not in set(kept_classes)], device=DEV)

    pseudo_list = [ci[c] for c in pseudo_classes]
    pseudo_to_idx = {c: i for i, c in enumerate(pseudo_classes)}
    kept_to_idx = {c: i for i, c in enumerate(kept_classes)}

    seen_train_fns, seen_val_fns = [], []
    novel_train_fns, novel_val_fns = [], []

    for c in kept_classes:
        fns = sorted(by[c])
        k_val = max(1, round(0.20 * len(fns)))
        seen_train_fns.extend(fns[:-k_val])
        seen_val_fns.extend(fns[-k_val:])

    for c in pseudo_classes:
        fns = sorted(by[c])
        k_val = max(1, round(0.20 * len(fns)))
        novel_train_fns.extend(fns[:-k_val])
        novel_val_fns.extend(fns[-k_val:])

    print(f"Validation Holdout Set: {len(seen_val_fns):,} seen samples | {len(novel_val_fns):,} pseudo-unseen samples")

    # Load validation query tensors
    Q_ctft_s = torch.stack([feats_ctft[idx_ctft[fn]] for fn in seen_val_fns]).to(DEV)
    Q_ctft_u = torch.stack([feats_ctft[idx_ctft[fn]] for fn in novel_val_fns]).to(DEV)

    Q_336_s = torch.stack([feats_336[idx_336[fn]] for fn in seen_val_fns]).to(DEV)
    Q_336_u = torch.stack([feats_336[idx_336[fn]] for fn in novel_val_fns]).to(DEV)

    Q_b2_s = torch.stack([feats_b2[idx_b2[fn]] for fn in seen_val_fns]).to(DEV)
    Q_b2_u = torch.stack([feats_b2[idx_b2[fn]] for fn in novel_val_fns]).to(DEV)

    Q_b2l_s = torch.stack([feats_b2l[idx_b2l[fn]] for fn in seen_val_fns]).to(DEV)
    Q_b2l_u = torch.stack([feats_b2l[idx_b2l[fn]] for fn in novel_val_fns]).to(DEV)

    Q_tb_s = torch.stack([feats_tb[idx_tb[fn]] for fn in seen_val_fns]).to(DEV)
    Q_tb_u = torch.stack([feats_tb[idx_tb[fn]] for fn in novel_val_fns]).to(DEV)

    y_seen_val = torch.tensor([kept_to_idx[lab[fn]] for fn in seen_val_fns], device=DEV)
    y_novel_val = torch.tensor([pseudo_to_idx[lab[fn]] for fn in novel_val_fns], device=DEV)

    # 1. Compute Seen Prototypes
    P_seen = torch.zeros(len(kept_classes), feats_ctft.size(1), device=DEV)
    cnt_seen = torch.zeros(len(kept_classes), device=DEV)
    for c in kept_classes:
        for fn in by[c]:
            if fn not in seen_val_fns:
                f = feats_ctft[idx_ctft[fn]].to(DEV)
                P_seen[kept_to_idx[c]] += f
                cnt_seen[kept_to_idx[c]] += 1
    P_seen = F.normalize(P_seen / cnt_seen.clamp(min=1).unsqueeze(1), dim=-1)

    # Compute seen similarities
    S_seen_on_s = Q_ctft_s @ P_seen.t()
    S_seen_on_u = Q_ctft_u @ P_seen.t()

    # Compute unseen legs
    b_ctft_pkg = torch.load(os.path.join(OUT, 'inat_photo_bank_ctftshift.pt'), map_location='cpu', weights_only=False)
    b_336_pkg = torch.load(os.path.join(OUT, 'inat_photo_bank_fullft336shift.pt'), map_location='cpu', weights_only=False)
    p_b2_f = F.normalize(torch.load(os.path.join(OUT, 'inat_tol_merged_b2_a05.pt'), map_location='cpu', weights_only=False)['protos'].float(), dim=-1).to(DEV)
    p_b2_l = F.normalize(torch.load(os.path.join(OUT, 'inat_tol_merged_b2lora_a0.5.pt'), map_location='cpu', weights_only=False)['protos'].float(), dim=-1).to(DEV)

    print("Computing Raw Multi-Modal Legs for Validation Sets...")
    # For novel validation samples:
    S_ctft_bank_u = score_bank_topk_mean(Q_ctft_u, b_ctft_pkg['bank'], pseudo_list, topm=4, device=DEV)
    S_336_bank_u = score_bank_topk_mean(Q_336_u, b_336_pkg['bank'], pseudo_list, topm=4, device=DEV)
    S_b2f_u = Q_b2_u @ p_b2_f[pseudo_idx].t()
    S_b2l_u = Q_b2l_u @ p_b2_l[pseudo_idx].t()
    S_txt_taxon_u = Q_ctft_u @ TtH[pseudo_idx].t()
    S_txt_tb_u = Q_tb_u @ Ttb[pseudo_idx].t()

    # For seen validation samples:
    S_ctft_bank_s = score_bank_topk_mean(Q_ctft_s, b_ctft_pkg['bank'], pseudo_list, topm=4, device=DEV)
    S_336_bank_s = score_bank_topk_mean(Q_336_s, b_336_pkg['bank'], pseudo_list, topm=4, device=DEV)
    S_b2f_s = Q_b2_s @ p_b2_f[pseudo_idx].t()
    S_b2l_s = Q_b2l_s @ p_b2_l[pseudo_idx].t()
    S_txt_taxon_s = Q_ctft_s @ TtH[pseudo_idx].t()
    S_txt_tb_s = Q_tb_s @ Ttb[pseudo_idx].t()

    # Query-Centered Degree-3 Polynomial Calibrator
    def query_center_poly(S: torch.Tensor, poly: MonotonicDegree3Polynomial) -> torch.Tensor:
        mu = S.mean(dim=1, keepdim=True)
        sigma = S.std(dim=1, keepdim=True).clamp(min=1e-6)
        S_norm = (S - mu) / sigma
        return poly(S_norm)

    poly_ctft = MonotonicDegree3Polynomial().to(DEV)
    poly_336 = MonotonicDegree3Polynomial().to(DEV)
    poly_b2f = MonotonicDegree3Polynomial().to(DEV)
    poly_b2l = MonotonicDegree3Polynomial().to(DEV)
    poly_taxon = MonotonicDegree3Polynomial().to(DEV)
    poly_tb = MonotonicDegree3Polynomial().to(DEV)

    # Fit each query-centered polynomial on pseudo-unseen labels
    for p, S in [
        (poly_ctft, S_ctft_bank_u),
        (poly_336, S_336_bank_u),
        (poly_b2f, S_b2f_u),
        (poly_b2l, S_b2l_u),
        (poly_taxon, S_txt_taxon_u),
        (poly_tb, S_txt_tb_u)
    ]:
        mu = S.mean(dim=1, keepdim=True)
        sigma = S.std(dim=1, keepdim=True).clamp(min=1e-6)
        p.fit_spearman_loss((S - mu) / sigma, y_novel_val, num_epochs=200, lr=0.05, device=DEV)

    cal_ctft_u = query_center_poly(S_ctft_bank_u, poly_ctft)
    cal_336_u = query_center_poly(S_336_bank_u, poly_336)
    cal_b2f_u = query_center_poly(S_b2f_u, poly_b2f)
    cal_b2l_u = query_center_poly(S_b2l_u, poly_b2l)
    cal_taxon_u = query_center_poly(S_txt_taxon_u, poly_taxon)
    cal_tb_u = query_center_poly(S_txt_tb_u, poly_tb)

    cal_ctft_s = query_center_poly(S_ctft_bank_s, poly_ctft)
    cal_336_s = query_center_poly(S_336_bank_s, poly_336)
    cal_b2f_s = query_center_poly(S_b2f_s, poly_b2f)
    cal_b2l_s = query_center_poly(S_b2l_s, poly_b2l)
    cal_taxon_s = query_center_poly(S_txt_taxon_s, poly_taxon)
    cal_tb_s = query_center_poly(S_txt_tb_s, poly_tb)

    # Dynamic Confidence Attention Fusion
    class DynamicConfidenceFusion(nn.Module):
        def __init__(self, n_legs: int = 6):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_legs, 16),
                nn.ReLU(),
                nn.Linear(16, n_legs)
            )

        def forward(self, legs: list[torch.Tensor]) -> torch.Tensor:
            stacked = torch.stack(legs, dim=-1) # [N, C, M]
            # Max confidence per leg for this query
            conf = torch.stack([leg.max(dim=1).values for leg in legs], dim=1) # [N, M]
            weights = F.softmax(self.net(conf), dim=-1).unsqueeze(1) # [N, 1, M]
            fused = (stacked * weights).sum(dim=-1) # [N, C]
            return fused

    dyn_fusion = DynamicConfidenceFusion(6).to(DEV)
    opt_dyn = torch.optim.Adam(dyn_fusion.parameters(), lr=0.02)
    legs_u = [cal_ctft_u.detach(), cal_336_u.detach(), cal_b2f_u.detach(), cal_b2l_u.detach(), cal_taxon_u.detach(), cal_tb_u.detach()]
    legs_s = [cal_ctft_s.detach(), cal_336_s.detach(), cal_b2f_s.detach(), cal_b2l_s.detach(), cal_taxon_s.detach(), cal_tb_s.detach()]

    for epoch in range(150):
        opt_dyn.zero_grad()
        fused_u = dyn_fusion(legs_u)
        loss = F.cross_entropy(fused_u, y_novel_val)
        loss.backward()
        opt_dyn.step()

    with torch.no_grad():
        fused_scores_u = dyn_fusion(legs_u)
        fused_scores_s = dyn_fusion(legs_s)
        acc_u_isolated = (fused_scores_u.argmax(dim=1) == y_novel_val).float().mean().item()

    print(f"\n[Dynamic Attention Unseen Accuracy]: {acc_u_isolated * 100:.2f}% (vs static ~55%)")

    # Train Logistic Regression Gate on Discrepancy Features
    X_s = extract_gating_features_tensor(Q_ctft_s, TtH[kept_idx], TtH[pseudo_idx]).cpu().numpy()
    X_u = extract_gating_features_tensor(Q_ctft_u, TtH[kept_idx], TtH[pseudo_idx]).cpu().numpy()

    gate = CalibratedNoveltyLogisticGate(C=1.0)
    gate.fit(np.concatenate([X_s, X_u]), np.concatenate([np.ones(len(X_s)), np.zeros(len(X_u))]))
    
    p_seen_s = torch.from_numpy(gate.predict_proba(X_s)).to(DEV).float()
    p_seen_u = torch.from_numpy(gate.predict_proba(X_u)).to(DEV).float()

    # Temperature Scaling for Unified Calibration
    tau_s = nn.Parameter(torch.tensor(1.0, device=DEV))
    tau_u = nn.Parameter(torch.tensor(1.0, device=DEV))
    opt_tau = torch.optim.LBFGS([tau_s, tau_u], lr=0.1, max_iter=50)

    def tau_closure():
        opt_tau.zero_grad()
        l_s = F.cross_entropy(S_seen_on_s / tau_s.clamp(min=0.1), y_seen_val)
        l_u = F.cross_entropy(fused_scores_u / tau_u.clamp(min=0.1), y_novel_val)
        tot = l_s + l_u
        tot.backward()
        return tot

    opt_tau.step(tau_closure)
    t_s = tau_s.item()
    t_u = tau_u.item()
    print(f"Calibrated Temperatures: tau_seen = {t_s:.3f}, tau_unseen = {t_u:.3f}")

    # Evaluate Arm 1: Hard Binary Routing (v60)
    route_s_hard = (p_seen_s >= gate.optimal_threshold)
    route_u_hard = (p_seen_u >= gate.optimal_threshold)
    
    pred_s_hard = torch.empty(len(seen_val_fns), dtype=torch.long, device=DEV)
    pred_s_hard[route_s_hard] = S_seen_on_s[route_s_hard].argmax(dim=1)
    pred_s_hard[~route_s_hard] = -1 # Trapped in unseen -> incorrect
    
    pred_u_hard = torch.empty(len(novel_val_fns), dtype=torch.long, device=DEV)
    pred_u_hard[~route_u_hard] = fused_scores_u[~route_u_hard].argmax(dim=1)
    pred_u_hard[route_u_hard] = -1 # Trapped in seen -> incorrect
    
    acc_seen_v60 = (pred_s_hard == y_seen_val).float().mean().item()
    acc_uns_v60 = (pred_u_hard == y_novel_val).float().mean().item()
    overall_v60 = 0.5635 * acc_seen_v60 + 0.4365 * acc_uns_v60

    # Evaluate Arm 2: Soft Probabilistic Marginal Fusion (v61)
    # Log-probability for seen classes: log P(seen|x) + log_softmax(S_seen / tau_s)
    # Log-probability for unseen classes: log P(unseen|x) + log_softmax(S_unseen / tau_u)
    
    def soft_probabilistic_predict(S_s, S_u, p_s):
        p_s = p_s.unsqueeze(1).clamp(min=1e-6, max=1.0 - 1e-6)
        log_p_seen = torch.log(p_s)
        log_p_unseen = torch.log(1.0 - p_s)
        
        log_s = log_p_seen + F.log_softmax(S_s / t_s, dim=1) # [N, S]
        log_u = log_p_unseen + F.log_softmax(S_u / t_u, dim=1) # [N, U]
        
        # Combined space [N, S + U]
        full_logits = torch.cat([log_s, log_u], dim=1)
        full_argmax = full_logits.argmax(dim=1)
        return full_argmax, full_argmax < S_s.size(1)

    preds_full_s, is_seen_s = soft_probabilistic_predict(S_seen_on_s, fused_scores_s, p_seen_s)
    preds_full_u, is_seen_u = soft_probabilistic_predict(S_seen_on_u, fused_scores_u, p_seen_u)

    acc_seen_v61 = ((preds_full_s == y_seen_val) & is_seen_s).float().mean().item()
    acc_uns_v61 = ((preds_full_u - S_seen_on_u.size(1) == y_novel_val) & ~is_seen_u).float().mean().item()
    overall_v61 = 0.5635 * acc_seen_v61 + 0.4365 * acc_uns_v61

    print("\n" + "=" * 80)
    print("=== SUMMARY OF BENCHMARK COMPARISON ===")
    print("=" * 80)
    print(f"| Architecture | Seen Acc | Unseen Acc | Overall Weighted Acc |")
    print(f"|---|---|---|---|")
    print(f"| **v60 (Hard Logistic Gate + Global Poly)** | {acc_seen_v60 * 100:.2f}% | {acc_uns_v60 * 100:.2f}% | **{overall_v60 * 100:.2f}%** |")
    print(f"| **v61 (Soft Marginal Fusion + Dynamic Poly)** | {acc_seen_v61 * 100:.2f}% | {acc_uns_v61 * 100:.2f}% | **{overall_v61 * 100:.2f}%** |")
    print(f"Δ (v61 vs v60): Seen: { (acc_seen_v61 - acc_seen_v60) * 100:+.2f}%, Unseen: { (acc_uns_v61 - acc_uns_v60) * 100:+.2f}%, Overall: { (overall_v61 - overall_v60) * 100:+.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()
