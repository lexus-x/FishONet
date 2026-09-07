"""Train a Learned Probabilistic Novelty Gate (MLP) and Calibrate Route Temperatures.

Replaces hand-engineered thresholds (f=0.60, tau=1.8, count exponent gamma=0.25)
with a learned, sample-dependent probability p(seen|x) and analytically optimized
temperatures (tau_seen*, tau_unseen*).

Usage:
  python research/train_learned_novelty_gate.py
"""
from __future__ import annotations

import json
import math
import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
DATA = os.path.join(ROOT, 'data', 'dl')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_emb(path: str):
    d = torch.load(path, map_location='cpu', weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def z_norm(v: torch.Tensor, mean=None, std=None):
    if mean is None:
        mean = v.mean(dim=0, keepdim=True)
    if std is None:
        std = v.std(dim=0, keepdim=True) + 1e-6
    return (v - mean) / std, mean, std


class NoveltyMLPGate(nn.Module):
    """Calibrated 2-layer MLP for predicting p(seen | x)."""
    def __init__(self, in_dim: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logit."""
        return self.net(x).squeeze(-1)

    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Returns calibrated probability in [0, 1]."""
        return torch.sigmoid(self.forward(x))


def extract_gating_features(
    q: torch.Tensor,
    T_seen: torch.Tensor,
    T_unseen: torch.Tensor,
    I_unseen: torch.Tensor | None = None
) -> torch.Tensor:
    """Extracts 7 discriminative gating signals for a batch of query embeddings q [B, d]."""
    # 1. Cosine similarity over seen classes
    sims_s = q @ T_seen.t() # [B, S]
    s_max_seen = sims_s.max(dim=1).values # [B]

    # 2. Top-1 vs Top-5 margin on seen classes
    top5_seen = sims_s.topk(min(5, sims_s.size(1)), dim=1).values
    margin_1_5_seen = top5_seen[:, 0] - top5_seen[:, -1]

    # 3. Softmax entropy over seen classes
    probs_seen = F.softmax(sims_s / 0.07, dim=1)
    entropy_seen = -(probs_seen * (probs_seen + 1e-9).log()).sum(dim=1) / math.log(sims_s.size(1))

    # 4. Energy score over seen classes
    energy_seen = -0.07 * torch.logsumexp(sims_s / 0.07, dim=1)

    # 5. Cosine similarity over unseen text & visual prototypes
    sims_u_txt = q @ T_unseen.t()
    if I_unseen is not None:
        sims_u_img = q @ I_unseen.t()
        sims_u = torch.maximum(sims_u_txt, sims_u_img)
    else:
        sims_u = sims_u_txt
    s_max_unseen = sims_u.max(dim=1).values # [B]

    # 6. Contrast difference (Seen Max - Unseen Max)
    diff_seen_unseen = s_max_seen - s_max_unseen

    # 7. Confidence Ratio
    ratio_seen_unseen = s_max_seen / (s_max_unseen.clamp(min=1e-4))

    features = torch.stack([
        s_max_seen,
        margin_1_5_seen,
        -entropy_seen, # higher => more confident
        -energy_seen,  # higher => more seen-like
        s_max_unseen,
        diff_seen_unseen,
        ratio_seen_unseen
    ], dim=1)

    return features


def compute_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    s = scores.float().cpu()
    y = labels.float().cpu()
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Pairwise AUC
    cmp = (pos.unsqueeze(1) > neg.unsqueeze(0)).float() + 0.5 * (pos.unsqueeze(1) == neg.unsqueeze(0)).float()
    return cmp.mean().item()


def main():
    print("=== Training Learned Probabilistic Novelty Gate & Temperature Calibration ===")

    # 1. Load taxonomy and labels
    classes = list(pickle.load(open(os.path.join(DATA, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(os.path.join(DATA, 'label_train.json'), 'r'))

    print("Loading Text Embeddings and Prototypes...")
    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), map_location='cpu', weights_only=False)['emb_taxon'].float(), dim=-1).to(DEV)
    
    # Load visual prototypes if available (1024-d matching ctftshift)
    I_all = None
    proto_path = os.path.join(OUT, 'inat_protos_ctftshift_full.pt')
    if not os.path.isfile(proto_path):
        proto_path = os.path.join(OUT, 'inat_protos_ctftshift.pt')
    if os.path.isfile(proto_path):
        p_data = torch.load(proto_path, map_location='cpu', weights_only=False)
        p_tensor = p_data.get('protos', p_data) if isinstance(p_data, dict) else p_data
        I_all = F.normalize(p_tensor.float(), dim=-1).to(DEV)

    # Load multi-backbone query embeddings for training images
    print("Loading Training Query Embeddings (ctftshift + bioclip2)...")
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

    # Partition training samples into seen (kept) and novel (pseudo-unseen)
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

    print(f"Dataset Split Statistics:")
    print(f"  ├─ Kept Seen Classes: {len(kept_classes):,} | Pseudo-Unseen Classes: {len(pseudo_classes):,}")
    print(f"  ├─ Seen Train: {len(seen_train_fns):,} | Seen Val: {len(seen_val_fns):,}")
    print(f"  └─ Novel Train: {len(novel_train_fns):,} | Novel Val: {len(novel_val_fns):,}")

    # Extract query feature tensors
    def get_queries(fns):
        q_ctft = torch.stack([feats_ctft[idx_ctft[fn]] for fn in fns]).to(DEV)
        q_b2 = torch.stack([feats_b2[idx_b2[fn]] for fn in fns]).to(DEV)
        # Average / fuse queries for maximum signal strength
        return q_ctft

    print("\nExtracting 7-Signal Discrepancy Features...")
    Q_seen_tr = get_queries(seen_train_fns)
    Q_novel_tr = get_queries(novel_train_fns)

    Q_seen_va = get_queries(seen_val_fns)
    Q_novel_va = get_queries(novel_val_fns)

    X_seen_tr = extract_gating_features(Q_seen_tr, T_seen, T_unseen, I_unseen)
    X_novel_tr = extract_gating_features(Q_novel_tr, T_seen, T_unseen, I_unseen)

    X_seen_va = extract_gating_features(Q_seen_va, T_seen, T_unseen, I_unseen)
    X_novel_va = extract_gating_features(Q_novel_va, T_seen, T_unseen, I_unseen)

    X_train_raw = torch.cat([X_seen_tr, X_novel_tr], dim=0)
    Y_train = torch.cat([torch.ones(X_seen_tr.size(0)), torch.zeros(X_novel_tr.size(0))], dim=0).to(DEV)

    X_val_raw = torch.cat([X_seen_va, X_novel_va], dim=0)
    Y_val = torch.cat([torch.ones(X_seen_va.size(0)), torch.zeros(X_novel_va.size(0))], dim=0).to(DEV)

    # Normalize feature columns
    X_train, feat_mean, feat_std = z_norm(X_train_raw)
    X_val, _, _ = z_norm(X_val_raw, mean=feat_mean, std=feat_std)

    # 2. Train NoveltyMLPGate
    print("\nTraining Calibrated Novelty MLP Gate...")
    gate_model = NoveltyMLPGate(in_dim=7, hidden_dim=32).to(DEV)
    opt = torch.optim.AdamW(gate_model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, 301):
        gate_model.train()
        logits = gate_model(X_train)
        loss = loss_fn(logits, Y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 25 == 0 or epoch == 300:
            gate_model.eval()
            with torch.no_grad():
                val_logits = gate_model(X_val)
                val_probs = torch.sigmoid(val_logits)
                val_auc = compute_auc(val_probs, Y_val)
                train_auc = compute_auc(torch.sigmoid(logits), Y_train)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in gate_model.state_dict().items()}

            print(f"  Epoch {epoch:3d}/300 | Train Loss: {loss.item():.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

    print(f"\n[*] Best Validation Gate AUC: {best_val_auc:.4f}")
    gate_model.load_state_dict({k: v.to(DEV) for k, v in best_state.items()})
    gate_model.eval()

    # 3. Analytical Temperature Calibration via NLL
    print("\nOptimizing Analytical Route Temperatures (tau_seen*, tau_unseen*)...")
    
    # We optimize tau_seen on seen validation samples and tau_unseen on pseudo-unseen validation samples
    tau_seen_param = nn.Parameter(torch.tensor([1.8], device=DEV))
    tau_unseen_param = nn.Parameter(torch.tensor([1.8], device=DEV))
    temp_opt = torch.optim.LBFGS([tau_seen_param, tau_unseen_param], lr=0.1, max_iter=50)

    # Query similarities
    S_seen_val = Q_seen_va @ T_seen.t()
    Y_seen_val = torch.tensor([kept_classes.index(lab[fn]) for fn in seen_val_fns], device=DEV)

    S_novel_val = Q_novel_va @ T_unseen.t()
    Y_novel_val = torch.tensor([pseudo_classes_order := list(pseudo_classes), [pseudo_classes_order.index(lab[fn]) for fn in novel_val_fns]][1], device=DEV)

    def temp_closure():
        temp_opt.zero_grad()
        t_s = tau_seen_param.clamp(min=0.1, max=5.0)
        t_u = tau_unseen_param.clamp(min=0.1, max=5.0)
        loss_s = F.cross_entropy(S_seen_val * t_s, Y_seen_val)
        loss_u = F.cross_entropy(S_novel_val * t_u, Y_novel_val)
        total_loss = loss_s + loss_u
        total_loss.backward()
        return total_loss

    temp_opt.step(temp_closure)

    tau_seen_opt = tau_seen_param.item()
    tau_unseen_opt = tau_unseen_param.item()
    print(f"  Optimized Temperature Scaling:")
    print(f"    ├─ tau_seen*   = {tau_seen_opt:.4f}")
    print(f"    └─ tau_unseen* = {tau_unseen_opt:.4f}")

    # 4. Save Gate & Calibration Package
    save_path = os.path.join(OUT, 'learned_novelty_gate.pt')
    save_payload = {
        'model_state': best_state,
        'feat_mean': feat_mean.cpu(),
        'feat_std': feat_std.cpu(),
        'val_auc': best_val_auc,
        'tau_seen': tau_seen_opt,
        'tau_unseen': tau_unseen_opt,
        'in_dim': 7,
        'hidden_dim': 32
    }
    torch.save(save_payload, save_path)
    print(f"\n[SUCCESS] Trained Learned Novelty Gate & Calibration saved to: {save_path}")


if __name__ == '__main__':
    main()
