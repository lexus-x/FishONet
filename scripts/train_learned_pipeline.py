"""Comprehensive Training Script for the Learning-Based Pipeline (No DBNorm).

Trains:
1. Calibrated Novelty Logistic Regression Gate on multi-discrepancy features.
2. Monotonic Degree-3 Polynomial Calibrators via Spearman rank loss on validation splits.
3. Multi-Modal Fusion weights via regularized regression.

Saves package to outputs/learned_pipeline_v60.pt.
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

from src.calibration.polynomial_calibrator import MonotonicDegree3Polynomial, MultiModalPolynomialCalibrator
from src.calibration.novelty_logistic_gate import CalibratedNoveltyLogisticGate, extract_gating_features_tensor
from src.pipeline.multi_modal_fusion import MultiModalFusionScorer, score_bank_topk_mean

OUT = os.path.join(ROOT, 'outputs')
DATA = os.path.join(ROOT, 'data', 'dl')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_emb(path: str):
    d = torch.load(path, map_location='cpu', weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def main():
    print("=" * 75)
    print("=== Training Learning-Based Pipeline: Logistic Gate + Degree-3 Calibrators ===")
    print("=" * 75)
    torch.set_num_threads(8)

    # 1. Load taxonomy & labels
    classes = list(pickle.load(open(os.path.join(DATA, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(os.path.join(DATA, 'label_train.json'), 'r'))

    print("Loading Text Embeddings and Prototypes...")
    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), map_location='cpu', weights_only=False)['emb_taxon'].float(), dim=-1).to(DEV)
    TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), map_location='cpu', weights_only=False)['emb_taxctx'].float(), dim=-1).to(DEV)
    Ttb = F.normalize(torch.load(os.path.join(OUT, 'text_emb_taxabind_taxctx.pt'), map_location='cpu', weights_only=False)['emb_taxctx'].float(), dim=-1).to(DEV)

    I_all = None
    proto_path = os.path.join(OUT, 'inat_protos_ctftshift_full.pt')
    if not os.path.isfile(proto_path):
        proto_path = os.path.join(OUT, 'inat_protos_ctftshift.pt')
    if os.path.isfile(proto_path):
        p_data = torch.load(proto_path, map_location='cpu', weights_only=False)
        p_tensor = p_data.get('protos', p_data) if isinstance(p_data, dict) else p_data
        I_all = F.normalize(p_tensor.float(), dim=-1).to(DEV)

    # Load multi-backbone query embeddings for training
    print("Loading Query Embeddings (ctftshift, ftshift, fullft336shift, bioclip2)...")
    idx_ctft, feats_ctft, files_ctft = load_emb(os.path.join(OUT, 'emb_train_ctftshift.pt'))
    idx_b2, feats_b2, files_b2 = load_emb(os.path.join(OUT, 'emb_train_bioclip2.pt'))

    common_files = [fn for fn in files_ctft if fn in idx_b2 and fn in lab and lab[fn] in ci]
    by = defaultdict(list)
    for fn in common_files:
        by[lab[fn]].append(fn)

    seen = sorted(by.keys())
    order = sorted(seen, key=lambda c: len(by[c]))

    # Hard pseudo-unseen split (rarest 20% seen classes held out as pseudo-unseen)
    n_pseudo = int(len(seen) * 0.20)
    pseudo_classes = set(order[:n_pseudo])
    kept_classes = [c for c in seen if c not in pseudo_classes]

    kept_idx = torch.tensor([ci[c] for c in kept_classes], device=DEV)
    unseen_idx = torch.tensor([i for i, c in enumerate(classes) if c not in set(kept_classes)], device=DEV)

    T_seen = TtH[kept_idx]
    T_unseen = TtH[unseen_idx]
    I_unseen = I_all[unseen_idx] if I_all is not None else None

    # Partition into Train / Validation sets
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

    print(f"Split Summary:")
    print(f"  ├─ Kept Seen Classes: {len(kept_classes):,} | Pseudo-Unseen Classes: {len(pseudo_classes):,}")
    print(f"  ├─ Seen Train: {len(seen_train_fns):,} | Seen Val: {len(seen_val_fns):,}")
    print(f"  └─ Novel Train: {len(novel_train_fns):,} | Novel Val: {len(novel_val_fns):,}")

    def get_queries(fns):
        return torch.stack([feats_ctft[idx_ctft[fn]] for fn in fns]).to(DEV)

    print("\n[Step 1/3] Extracting Multi-Discrepancy Gating Features...")
    Q_seen_tr = get_queries(seen_train_fns)
    Q_novel_tr = get_queries(novel_train_fns)
    Q_seen_va = get_queries(seen_val_fns)
    Q_novel_va = get_queries(novel_val_fns)

    X_seen_tr = extract_gating_features_tensor(Q_seen_tr, T_seen, T_unseen, I_unseen)
    X_novel_tr = extract_gating_features_tensor(Q_novel_tr, T_seen, T_unseen, I_unseen)
    X_seen_va = extract_gating_features_tensor(Q_seen_va, T_seen, T_unseen, I_unseen)
    X_novel_va = extract_gating_features_tensor(Q_novel_va, T_seen, T_unseen, I_unseen)

    X_train = torch.cat([X_seen_tr, X_novel_tr], dim=0).cpu().numpy()
    y_train = np.concatenate([np.ones(X_seen_tr.size(0)), np.zeros(X_novel_tr.size(0))])

    X_val = torch.cat([X_seen_va, X_novel_va], dim=0).cpu().numpy()
    y_val = np.concatenate([np.ones(X_seen_va.size(0)), np.zeros(X_novel_va.size(0))])

    print("Training Calibrated Novelty Logistic Regression Gate...")
    gate = CalibratedNoveltyLogisticGate(C=1.0)
    best_thr = gate.fit(X_train, y_train, w_seen=0.5635, w_unseen=0.4365)
    
    val_probs = gate.predict_proba(X_val)
    val_preds = (val_probs >= best_thr).astype(int)
    val_acc_seen = np.mean(val_preds[y_val == 1] == 1)
    val_acc_unseen = np.mean(val_preds[y_val == 0] == 0)
    val_weighted_acc = 0.5635 * val_acc_seen + 0.4365 * val_acc_unseen
    print(f"  ├─ Optimal Logistic Threshold: {best_thr:.4f}")
    print(f"  ├─ Val Seen Accuracy: {val_acc_seen * 100:.2f}% | Val Unseen Accuracy: {val_acc_unseen * 100:.2f}%")
    print(f"  └─ Val Novelty Gate Overall: {val_weighted_acc * 100:.2f}%")

    print("\n[Step 2/3] Training Monotonic Degree-3 Polynomial Calibrators via Spearman Rank Optimization...")
    modality_names = ['ctft_bank', '336_bank', 'b2_frozen_tol', 'b2_lora_tol', 'text_taxon', 'text_taxabind']
    calibrators = MultiModalPolynomialCalibrator(modality_names).to(DEV)

    # Extract raw validation similarity legs for pseudo-unseen classes
    pseudo_list = [ci[c] for c in pseudo_classes]
    pseudo_to_idx = {c: i for i, c in enumerate(pseudo_classes)}
    y_novel_labels = torch.tensor([pseudo_to_idx[lab[fn]] for fn in novel_val_fns], device=DEV)

    # 1. Text taxon leg
    S_txt_taxon = Q_novel_va @ TtH[torch.tensor(pseudo_list, device=DEV)].t()
    rho_txt = calibrators.calibrators['text_taxon'].fit_spearman_loss(S_txt_taxon, y_novel_labels, num_epochs=200, lr=0.05, device=DEV)
    print(f"  ├─ Text Taxon Calibrator: Spearman rho = {rho_txt:.4f}")

    # 2. Text taxabind leg (512-d)
    idx_tb, feats_tb, _ = load_emb(os.path.join(OUT, 'emb_train_taxabind.pt'))
    Q_tb_va = torch.stack([feats_tb[idx_tb[fn]] for fn in novel_val_fns]).to(DEV)
    S_txt_tb = Q_tb_va @ Ttb[torch.tensor(pseudo_list, device=DEV)].t()
    rho_tb = calibrators.calibrators['text_taxabind'].fit_spearman_loss(S_txt_tb, y_novel_labels, num_epochs=200, lr=0.05, device=DEV)
    print(f"  ├─ Text TaxaBind Calibrator: Spearman rho = {rho_tb:.4f}")

    # 3. Bank CTFT leg
    b_ctft_pkg = torch.load(os.path.join(OUT, 'inat_photo_bank_ctftshift.pt'), map_location='cpu', weights_only=False)
    S_bank_ctft = score_bank_topk_mean(Q_novel_va, b_ctft_pkg['bank'], pseudo_list, topm=4, device=DEV)
    rho_ctft = calibrators.calibrators['ctft_bank'].fit_spearman_loss(S_bank_ctft, y_novel_labels, num_epochs=200, lr=0.05, device=DEV)
    print(f"  ├─ CTFT Photo Bank Calibrator: Spearman rho = {rho_ctft:.4f}")

    # 4. Bank 336 leg
    idx_336, feats_336, _ = load_emb(os.path.join(OUT, 'emb_train_fullft336shift.pt'))
    Q_336_va = torch.stack([feats_336[idx_336[fn]] for fn in novel_val_fns]).to(DEV)
    b_336_pkg = torch.load(os.path.join(OUT, 'inat_photo_bank_fullft336shift.pt'), map_location='cpu', weights_only=False)
    S_bank_336 = score_bank_topk_mean(Q_336_va, b_336_pkg['bank'], pseudo_list, topm=4, device=DEV)
    rho_336 = calibrators.calibrators['336_bank'].fit_spearman_loss(S_bank_336, y_novel_labels, num_epochs=200, lr=0.05, device=DEV)
    print(f"  ├─ 336 Photo Bank Calibrator: Spearman rho = {rho_336:.4f}")

    # 5. BioCLIP-2 Frozen ToL
    p_b2_f = F.normalize(torch.load(os.path.join(OUT, 'inat_tol_merged_b2_a05.pt'), map_location='cpu', weights_only=False)['protos'].float(), dim=-1).to(DEV)
    idx_b2_tr, feats_b2_tr, _ = load_emb(os.path.join(OUT, 'emb_train_bioclip2.pt'))
    Q_b2_va = torch.stack([feats_b2_tr[idx_b2_tr[fn]] for fn in novel_val_fns]).to(DEV)
    S_b2_f = Q_b2_va @ p_b2_f[torch.tensor(pseudo_list, device=DEV)].t()
    rho_b2f = calibrators.calibrators['b2_frozen_tol'].fit_spearman_loss(S_b2_f, y_novel_labels, num_epochs=200, lr=0.05, device=DEV)
    print(f"  ├─ BioCLIP-2 Frozen ToL Calibrator: Spearman rho = {rho_b2f:.4f}")

    # 6. BioCLIP-2 LoRA ToL
    p_b2_l = F.normalize(torch.load(os.path.join(OUT, 'inat_tol_merged_b2lora_a0.5.pt'), map_location='cpu', weights_only=False)['protos'].float(), dim=-1).to(DEV)
    idx_b2_l, feats_b2_l, _ = load_emb(os.path.join(OUT, 'emb_train_bioclip2_lora_v2.pt'))
    Q_b2_l_va = torch.stack([feats_b2_l[idx_b2_l[fn]] for fn in novel_val_fns]).to(DEV)
    S_b2_l = Q_b2_l_va @ p_b2_l[torch.tensor(pseudo_list, device=DEV)].t()
    rho_b2l = calibrators.calibrators['b2_lora_tol'].fit_spearman_loss(S_b2_l, y_novel_labels, num_epochs=200, lr=0.05, device=DEV)
    print(f"  └─ BioCLIP-2 LoRA ToL Calibrator: Spearman rho = {rho_b2l:.4f}")

    print("\n[Step 3/3] Learning Multi-Modal Fusion Weights via Cross-Entropy Regression...")
    fusion_scorer = MultiModalFusionScorer(modality_names).to(DEV)
    raw_legs_val = {
        'ctft_bank': S_bank_ctft,
        '336_bank': S_bank_336,
        'b2_frozen_tol': S_b2_f,
        'b2_lora_tol': S_b2_l,
        'text_taxon': S_txt_taxon,
        'text_taxabind': S_txt_tb
    }
    
    with torch.no_grad():
        calibrated_val = {k: v.detach() for k, v in calibrators(raw_legs_val).items()}
    
    opt_fusion = torch.optim.Adam(fusion_scorer.parameters(), lr=0.1)
    for epoch in range(150):
        opt_fusion.zero_grad()
        fused = fusion_scorer(calibrated_val)
        loss = F.cross_entropy(fused, y_novel_labels)
        loss.backward()
        opt_fusion.step()

    with torch.no_grad():
        w_final = F.softmax(fusion_scorer.weights, dim=0).cpu().numpy()
        pred_top1 = fusion_scorer(calibrated_val).argmax(dim=1)
        acc_novel_val = (pred_top1 == y_novel_labels).float().mean().item()
        
    print("  Learned Normalized Modality Fusion Weights:")
    for name, w in zip(modality_names, w_final):
        print(f"    ├─ {name:16s}: {w:.4f}")
    print(f"  └─ Val Pseudo-Unseen Top-1 Accuracy: {acc_novel_val * 100:.2f}%")

    # 4. Save entire calibrated pipeline package
    save_pkg_path = os.path.join(OUT, 'learned_pipeline_v60.pt')
    payload = {
        'gate_state': gate.state_dict(),
        'calibrators_state': calibrators.state_dict(),
        'fusion_state': fusion_scorer.state_dict(),
        'modality_names': modality_names,
        'val_metrics': {
            'val_acc_seen': float(val_acc_seen),
            'val_acc_unseen': float(val_acc_unseen),
            'val_novel_top1': float(acc_novel_val),
            'val_weighted_acc': float(val_weighted_acc)
        }
    }
    torch.save(payload, save_pkg_path)
    print(f"\n[SUCCESS] Learned Pipeline saved to: {save_pkg_path}")
    print("=" * 75)


if __name__ == '__main__':
    main()
