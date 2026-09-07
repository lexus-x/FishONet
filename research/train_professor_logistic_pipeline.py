"""Implementation of Professor's Proposed Pipeline from Handwritten Diagram.

Architecture:
1. Shared Embedding Space:
   - Vision embedding: z_i = BioCLIP_vision(x) in R^d (L2 normalized)
   - Text embeddings: z_T for all 17,393 seen + unseen classes (prefixed)
   - Seen Class Prototypes P_c: mean of training image embeddings with L2-norm before & after mean.

2. Two Initial Classifier Decisions:
   - best seen candidate:   c* = argmax_{c in seen} cos(z_i, P_c)
   - best unseen candidate: u* = argmax_{u in unseen} cos(z_i, z_{Tu})

3. Feature Engineering for Soft Gate Debate:
   - max cos(z_i, P_seen)
   - sum(cos(z_i, P_seen)) / (1 + C_seen)
   - max cos(z_i, z_{T,seen})
   - max cos(z_i, z_{T,unseen})
   - margin: max cos(z_i, P_seen) - max cos(z_i, z_{T,unseen})
   - margin: max cos(z_i, z_{T,seen}) - max cos(z_i, z_{T,unseen})
   - Shannon entropy over seen classes: H(p) = -sum p_i log2(p_i)
   - Top-1 vs Top-2 seen margin
   - Top-1 vs Top-5 seen margin

4. 2-Class Logistic Regression Gate:
   - Trained on the iNaturalist validation set (labeled seen y=1, unseen y=0)
   - Output probability score s = sigma(W^T x_i + b) in [0, 1]
   - Final Decision: if score >= t -> best seen class, else -> best unseen class.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dl')
OUT = os.path.join(ROOT, 'outputs')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_entropy_bits(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p_safe = p.clamp(min=eps)
    return -torch.sum(p_safe * torch.log2(p_safe), dim=-1)


def extract_features_vector(
    z: torch.Tensor,            # [B, d] normalized image embeddings
    P_seen: torch.Tensor,       # [C_seen, d] seen prototypes
    T_seen: torch.Tensor,       # [C_seen, d] seen text embeddings
    T_unseen: torch.Tensor,     # [C_unseen, d] unseen text embeddings
    temp: float = 0.0300
) -> torch.Tensor:
    """Extracts the rich debate features specified in professor's handwritten notes."""
    C_s = P_seen.size(0)
    C_u = T_unseen.size(0)

    # 1. Seen Prototype Similarities
    sim_P_s = z @ P_seen.t() # [B, C_s]
    top5_P_s = sim_P_s.topk(min(5, C_s), dim=1).values
    max_P_s = top5_P_s[:, 0]
    margin_1_2_Ps = top5_P_s[:, 0] - top5_P_s[:, 1]
    margin_1_5_Ps = top5_P_s[:, 0] - top5_P_s[:, -1]
    mean_P_s = sim_P_s.sum(dim=1) / (1.0 + C_s)

    # 2. Shannon Entropy over Seen Classes
    p_s = F.softmax(sim_P_s / temp, dim=1)
    H_s = compute_entropy_bits(p_s)

    # 3. Seen Text Similarities
    sim_T_s = z @ T_seen.t() # [B, C_s]
    max_T_s = sim_T_s.max(dim=1).values
    mean_T_s = sim_T_s.sum(dim=1) / (1.0 + C_s)

    # 4. Unseen Text Similarities
    sim_T_u = z @ T_unseen.t() # [B, C_u]
    top2_T_u = sim_T_u.topk(min(2, C_u), dim=1).values
    max_T_u = top2_T_u[:, 0]
    margin_1_2_Tu = top2_T_u[:, 0] - top2_T_u[:, 1]
    mean_T_u = sim_T_u.sum(dim=1) / (1.0 + C_u)

    # 5. Debate Contrast Margins
    diff_P_seen_T_unseen = max_P_s - max_T_u
    diff_T_seen_T_unseen = max_T_s - max_T_u
    ratio_P_seen_T_unseen = max_P_s / max_T_u.clamp(min=1e-4)

    feature_matrix = torch.stack([
        max_P_s,               # 1. Top-1 Seen Prototype Similarity
        mean_P_s,              # 2. Normalized Seen Similarity Sum / (1 + C)
        -H_s,                  # 3. Negative Seen Entropy (high -> concentrated on seen)
        margin_1_2_Ps,         # 4. Top-1 vs Top-2 Seen Margin
        margin_1_5_Ps,         # 5. Top-1 vs Top-5 Seen Margin
        max_T_s,               # 6. Top-1 Seen Text Similarity
        mean_T_s,              # 7. Mean Seen Text Similarity / (1 + C)
        max_T_u,               # 8. Top-1 Unseen Text Similarity
        mean_T_u,              # 9. Mean Unseen Text Similarity / (1 + C)
        margin_1_2_Tu,         # 10. Top-1 vs Top-2 Unseen Margin
        diff_P_seen_T_unseen,  # 11. Core Debate: max Seen Proto - max Unseen Text
        diff_T_seen_T_unseen,  # 12. Core Debate: max Seen Text - max Unseen Text
        ratio_P_seen_T_unseen, # 13. Seen / Unseen Ratio
    ], dim=1)

    return feature_matrix


def main():
    print("=" * 85)
    print("=== Training Professor's 2-Class Logistic Regression Soft Gate ===")
    print("=" * 85)

    # 1. Load taxonomy & training labels
    classes = list(pickle.load(open(os.path.join(DATA, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(os.path.join(DATA, 'label_train.json'), 'r'))

    seen_classes = sorted(set(lab.values()))
    unseen_classes = sorted(set(classes) - set(seen_classes))

    seen_idx = torch.tensor([ci[c] for c in seen_classes], device=DEV)
    unseen_idx = torch.tensor([ci[c] for c in unseen_classes], device=DEV)

    seen_to_pos = {ci[c]: j for j, c in enumerate(seen_classes)}

    print(f"Candidate Space: {len(classes):,} Classes ({len(seen_classes):,} Seen, {len(unseen_classes):,} Unseen)")

    # 2. Build Class Prototypes P_seen (L2 norm before and after mean operation)
    print("\nComputing Seen Class Prototypes (P_seen) with L2 norm before & after mean...")
    d_tr = torch.load(os.path.join(OUT, 'emb_train_ctftshift.pt'), map_location='cpu', weights_only=False)
    idx_tr = {fn: i for i, fn in enumerate(d_tr['files'])}
    feats_tr = F.normalize(d_tr['feats'].float(), dim=-1) # L2 norm before mean

    P_seen = torch.zeros(len(seen_classes), feats_tr.size(1), device=DEV)
    cnt_seen = torch.zeros(len(seen_classes), device=DEV)
    for fn, cname in lab.items():
        if fn in idx_tr and cname in ci:
            pos = seen_to_pos[ci[cname]]
            P_seen[pos] += feats_tr[idx_tr[fn]].to(DEV)
            cnt_seen[pos] += 1
    P_seen = F.normalize(P_seen / cnt_seen.clamp(min=1).unsqueeze(1), dim=-1) # L2 norm after mean

    # 3. Load Prefixed Text Embeddings z_T (Both Seen and Unseen Classes)
    print("Loading Prefixed Text Embeddings (z_T)...")
    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), map_location='cpu', weights_only=False)['emb_taxon'].float(), dim=-1).to(DEV)
    T_seen = TtH[seen_idx]
    T_unseen = TtH[unseen_idx]

    # 4. Extract Features on the Labeled iNaturalist Validation Dataset (y in {0, 1})
    manifest_path = os.path.join(OUT, 'inat_val_manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print("\nLoading iNaturalist Validation Images to form training set (X, Y)...")
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2.5-vith14')

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
            super().__init__()
            self.base = base
            for p in self.base.parameters(): p.requires_grad_(False)
            self.r = r; self.scaling = alpha / max(r, 1)
            self.A = nn.Parameter(torch.zeros(r, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.drop = nn.Identity(); self.lora_scale = 1.0
        def forward(self, x): return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scaling * self.lora_scale

    blocks = model.visual.transformer.resblocks
    for i in range(len(blocks) - 12, len(blocks)):
        blk = blocks[i]
        blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, 16, 32)
        blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, 16, 32)

    lora_path = os.path.join(OUT, 'ctft_lora.pt')
    if os.path.isfile(lora_path):
        ckpt = torch.load(lora_path, map_location='cpu', weights_only=False)
        state = ckpt.get('state', ckpt)
        model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
        print(f"Loaded LoRA weights from {lora_path}")

    model = model.to(DEV).eval()

    class INatDataset(Dataset):
        def __init__(self, items, transform):
            self.items = [it for it in items if os.path.isfile(it['image_path'])]
            self.transform = transform
        def __len__(self): return len(self.items)
        def __getitem__(self, idx):
            it = self.items[idx]
            img = Image.open(it['image_path']).convert('RGB')
            # y = 1 for seen, y = 0 for unseen
            y = 0 if it.get('is_unseen', False) else 1
            cname = it['class_name']
            c_idx = ci.get(cname, -1)
            return self.transform(img), y, c_idx

    dataset = INatDataset(manifest['items'], preprocess)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    all_z = []
    all_y = []
    all_c_idx = []
    print(f"Encoding {len(dataset):,} iNaturalist images for logistic regression training...")
    with torch.no_grad():
        for imgs, y_batch, c_batch in loader:
            imgs = imgs.to(DEV)
            with torch.amp.autocast('cuda'):
                z = F.normalize(model.encode_image(imgs).float(), dim=-1)
            all_z.append(z.cpu())
            all_y.append(y_batch)
            all_c_idx.append(c_batch)

    all_z = torch.cat(all_z).to(DEV)
    all_y = torch.cat(all_y).numpy()
    all_c_idx = torch.cat(all_c_idx).to(DEV)

    print(f"Extracted features for {len(all_y):,} samples ({all_y.sum():,} seen, {(1-all_y).sum():,} unseen).")

    # 5. Extract Feature Matrix X
    X_tensor = extract_features_vector(all_z, P_seen, T_seen, T_unseen, temp=0.0300)
    X_np = X_tensor.cpu().numpy()

    # 6. Train 2-Class Logistic Regression Model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)

    clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced')
    clf.fit(X_scaled, all_y)

    probs_seen = clf.predict_proba(X_scaled)[:, 1] # P(seen | x)
    auc = roc_auc_score(all_y, probs_seen)
    print(f"\n[Logistic Regression Trained Successfully]")
    print(f"  ├─ Feature Dimension K : {X_np.shape[1]}")
    print(f"  ├─ Validation ROC AUC  : {auc:.4f}")
    print(f"  ├─ Feature Weights (W) :")
    feature_names = [
        "max_cos(z, P_seen)",
        "mean_cos(z, P_seen)/(1+C)",
        "-H(p) (Seen Entropy)",
        "margin_1_2(P_seen)",
        "margin_1_5(P_seen)",
        "max_cos(z, T_seen)",
        "mean_cos(z, T_seen)/(1+C)",
        "max_cos(z, T_unseen)",
        "mean_cos(z, T_unseen)/(1+C)",
        "margin_1_2(T_unseen)",
        "diff(P_seen - T_unseen)",
        "diff(T_seen - T_unseen)",
        "ratio(P_seen / T_unseen)",
    ]
    for name, w in zip(feature_names, clf.coef_[0]):
        print(f"     * {name:<28}: {w:+.4f}")
    print(f"     * Intercept (b)               : {clf.intercept_[0]:+.4f}")

    # 7. Calibrate Threshold t on iNaturalist Validation Set
    sim_P_s = all_z @ P_seen.t()
    sim_T_u = all_z @ T_unseen.t()

    best_seen_pred = seen_idx[sim_P_s.argmax(dim=-1)]
    best_unseen_pred = unseen_idx[sim_T_u.argmax(dim=-1)]

    best_t = 0.50
    best_overall = 0.0
    results_t = []

    print("\n" + "=" * 80)
    print("=== SWEEPING DECISION THRESHOLD t (P(seen|x) >= t) ===")
    print("=" * 80)
    print(f"{'Threshold t':<12} | {'Seen Retention':<16} | {'Novel Yield':<16} | {'Overall Val Acc':<16}")
    print("-" * 80)

    for t in np.linspace(0.10, 0.90, 17):
        route_seen = (probs_seen >= t)
        preds = torch.where(torch.from_numpy(route_seen).to(DEV), best_seen_pred, best_unseen_pred)

        acc = (preds == all_c_idx).float().mean().item()
        seen_ret = (route_seen[all_y == 1]).mean()
        novel_yield = ((~route_seen)[all_y == 0]).mean()

        if acc > best_overall:
            best_overall = acc
            best_t = t

        print(f"{t:<12.2f} | {seen_ret*100:>14.2f}% | {novel_yield*100:>14.2f}% | **{acc*100:>14.2f}%**")
    print("=" * 80)
    print(f"Optimal Threshold: t = {best_t:.2f} (Overall Val Acc = {best_overall*100:.2f}%)")

    # 8. Save Pipeline Model Package
    gate_package = {
        'model_name': 'BioCLIP-2.5 ViT-H/14',
        'clf_coef': clf.coef_.tolist(),
        'clf_intercept': clf.intercept_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'optimal_threshold': float(best_t),
        'auc': float(auc),
        'feature_names': feature_names,
    }
    pkg_path = os.path.join(OUT, 'professor_logistic_pipeline.json')
    with open(pkg_path, 'w') as f:
        json.dump(gate_package, f, indent=2)
    print(f"\n[SAVED] Pipeline package saved to {pkg_path}")


if __name__ == '__main__':
    main()
