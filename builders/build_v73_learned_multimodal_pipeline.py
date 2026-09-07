"""v73: Fully-Learned Open-Set Species Recognition Pipeline.

Architecture (100% Learning-Based, Zero Handcrafted Batch Quantiles):
1. Representation:
   - BioCLIP-2.5 ViT-H/14 with Aspect-Shift adaptation.
   - Seen Prototypes P_c: Empirical centroids of seen training features.
   - Multimodal Unseen Anchors E_u: Fused taxonomic text and visual prototype anchors
     from iNaturalist + TreeOfLife.

2. Candidate Decisions:
   - Best seen candidate:   c* = argmax_{c in seen} cos(z_i, P_c)
   - Best unseen candidate: u* = argmax_{u in unseen} cos(z_i, E_u)

3. Learned Logistic Soft Gate:
   - 13 debate features extracted per sample (similarities, entropy, contrast margins).
   - Parameters (W, b) fitted via Maximum Likelihood Estimation on the validation set.
   - Strictly inductive, per-sample inference: P(seen | x_i) = sigma(W^T Phi(x_i) + b)
   - Standard Bayes-optimal decision threshold: P(seen | x_i) >= 0.50 -> c*, else -> u*.

Usage:
  /home/ubuntu/miniconda3/envs/onet/bin/python builders/build_v73_learned_multimodal_pipeline.py
"""
from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dl')
OUT = os.path.join(ROOT, 'outputs')
SUB = os.path.join(ROOT, 'submissions')
os.makedirs(SUB, exist_ok=True)


def compute_entropy_bits(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p_safe = p.clamp(min=eps)
    return -torch.sum(p_safe * torch.log2(p_safe), dim=-1)


def extract_debate_features(
    z: torch.Tensor,            # [N, d]
    P_seen: torch.Tensor,       # [C_seen, d]
    T_seen: torch.Tensor,       # [C_seen, d]
    T_unseen: torch.Tensor,     # [C_unseen, d]
    temp: float = 0.0300
) -> torch.Tensor:
    """Extracts the 13 debate features for the learned logistic regression gate."""
    C_s = P_seen.size(0)
    C_u = T_unseen.size(0)

    # 1. Seen Prototype Similarities
    sim_P_s = z @ P_seen.t()
    top5_P_s = sim_P_s.topk(min(5, C_s), dim=1).values
    max_P_s = top5_P_s[:, 0]
    margin_1_2_Ps = top5_P_s[:, 0] - top5_P_s[:, 1]
    margin_1_5_Ps = top5_P_s[:, 0] - top5_P_s[:, -1]
    mean_P_s = sim_P_s.sum(dim=1) / (1.0 + C_s)

    # 2. Shannon Entropy over Seen Classes
    p_s = F.softmax(sim_P_s / temp, dim=1)
    H_s = compute_entropy_bits(p_s)

    # 3. Seen Text Similarities
    sim_T_s = z @ T_seen.t()
    max_T_s = sim_T_s.max(dim=1).values
    mean_T_s = sim_T_s.sum(dim=1) / (1.0 + C_s)

    # 4. Unseen Text Similarities
    sim_T_u = z @ T_unseen.t()
    top2_T_u = sim_T_u.topk(min(2, C_u), dim=1).values
    max_T_u = top2_T_u[:, 0]
    margin_1_2_Tu = top2_T_u[:, 0] - top2_T_u[:, 1]
    mean_T_u = sim_T_u.sum(dim=1) / (1.0 + C_u)

    # 5. Debate Contrast Margins
    diff_P_seen_T_unseen = max_P_s - max_T_u
    diff_T_seen_T_unseen = max_T_s - max_T_u
    ratio_P_seen_T_unseen = max_P_s / max_T_u.clamp(min=1e-4)

    features = torch.stack([
        max_P_s,
        mean_P_s,
        -H_s,
        margin_1_2_Ps,
        margin_1_5_Ps,
        max_T_s,
        mean_T_s,
        max_T_u,
        mean_T_u,
        margin_1_2_Tu,
        diff_P_seen_T_unseen,
        diff_T_seen_T_unseen,
        ratio_P_seen_T_unseen,
    ], dim=1)

    return features


def load_emb(path: str):
    d = torch.load(path, map_location='cpu', weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def main():
    print("=" * 85)
    print("=== v73: Building 100% Learned Multimodal Open-Set Recognition Pipeline ===")
    print("=" * 85)

    # 1. Load taxonomy & training labels
    classes = list(pickle.load(open(os.path.join(DATA, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(os.path.join(DATA, 'label_train.json'), 'r'))

    seen_classes = sorted(set(lab.values()))
    unseen_classes = sorted(set(classes) - set(seen_classes))

    seen_idx = torch.tensor([ci[c] for c in seen_classes], device=DEV)
    unseen_idx = torch.tensor([ci[c] for c in unseen_classes], device=DEV)

    seen_to_pos = {ci[c]: j for j, c in enumerate(seen_classes)}

    print(f"Candidate Space: {NCLS:,} Total Classes ({len(seen_classes):,} Seen, {len(unseen_classes):,} Unseen)")

    # 2. Build Seen Class Prototypes (P_seen)
    print("\nComputing Seen Class Prototypes (P_seen)...")
    d_tr = torch.load(os.path.join(OUT, 'emb_train_ctftshift.pt'), map_location='cpu', weights_only=False)
    idx_tr = {fn: i for i, fn in enumerate(d_tr['files'])}
    feats_tr = F.normalize(d_tr['feats'].float(), dim=-1)

    P_seen = torch.zeros(len(seen_classes), feats_tr.size(1), device=DEV)
    cnt_seen = torch.zeros(len(seen_classes), device=DEV)
    for fn, cname in lab.items():
        if fn in idx_tr and cname in ci:
            pos = seen_to_pos[ci[cname]]
            P_seen[pos] += feats_tr[idx_tr[fn]].to(DEV)
            cnt_seen[pos] += 1
    P_seen = F.normalize(P_seen / cnt_seen.clamp(min=1).unsqueeze(1), dim=-1)

    # 3. Load Prefixed Text Embeddings z_T
    print("Loading Prefixed Text Embeddings (z_T)...")
    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), map_location='cpu', weights_only=False)['emb_taxon'].float(), dim=-1).to(DEV)
    T_seen = TtH[seen_idx]
    T_unseen = TtH[unseen_idx]

    # 4. Build Multimodal Unseen Anchor Matrix (Visual Prototypes + Taxonomic Text)
    print("Loading Visual Prototype Banks (iNaturalist + TreeOfLife)...")
    proto_candidates = [
        os.path.join(OUT, 'inat_protos_ctftshift_full.pt'),
        os.path.join(OUT, 'inat_protos_ctftshift.pt'),
    ]
    proto_path = None
    for p in proto_candidates:
        if os.path.isfile(p):
            proto_path = p
            break

    protoC = torch.zeros(NCLS, TtH.shape[1], device=DEV)
    hasp = torch.zeros(NCLS, dtype=torch.bool, device=DEV)

    if proto_path:
        pd = torch.load(proto_path, map_location='cpu', weights_only=False)
        p_classes = pd['classes']
        p_protos = F.normalize(pd['protos'].float(), dim=-1)
        name2row = {c: i for i, c in enumerate(p_classes)}
        for i, c in enumerate(classes):
            r = name2row.get(c)
            if r is not None and float(p_protos[r].norm()) > 0.5:
                protoC[i] = p_protos[r].to(DEV)
                hasp[i] = True

        P_unseen = protoC[unseen_idx]
        has_proto = hasp[unseen_idx].unsqueeze(-1)
        print(f"  Visual Prototypes mapped for {int(hasp[unseen_idx].sum()):,}/{len(unseen_classes):,} unseen classes.")
        
        # Fused Multimodal Anchor: alpha * Text + (1 - alpha) * Visual Prototype
        E_unseen = torch.where(has_proto, 0.50 * T_unseen + 0.50 * P_unseen, T_unseen)
        E_unseen = F.normalize(E_unseen, dim=-1)
    else:
        print("  Visual prototypes not found, falling back to text.")
        E_unseen = T_unseen

    # 5. Load Evaluation Images (35,665 blind test images)
    print("\nLoading 35,665 Evaluation Images...")
    idx_te, feats_te, files_te = load_emb(os.path.join(OUT, 'emb_test_ctftshift.pt'))
    idx_un, feats_un, files_un = load_emb(os.path.join(OUT, 'emb_unseen_ctftshift.pt'))

    all_files = files_te + files_un
    assert len(all_files) == 35665
    Q_eval = torch.cat([feats_te, feats_un]).to(DEV)

    # 6. Compute Candidate Decisions
    print("Computing candidate decisions (Seen Head vs Multimodal Unseen Head)...")
    sim_P_s = Q_eval @ P_seen.t()
    sim_E_u = Q_eval @ E_unseen.t()

    best_seen_pred = seen_idx[sim_P_s.argmax(dim=-1)]
    best_unseen_pred = unseen_idx[sim_E_u.argmax(dim=-1)]

    # 7. Extract Debate Features & Evaluate Learned Logistic Gate
    print("Extracting debate features for logistic soft gate...")
    X_tensor = extract_debate_features(Q_eval, P_seen, T_seen, T_unseen, temp=0.0300)
    X_np = X_tensor.cpu().numpy()

    pkg_path = os.path.join(OUT, 'professor_logistic_pipeline.json')
    pkg = json.load(open(pkg_path, 'r'))
    mean = np.array(pkg['scaler_mean'])
    scale = np.array(pkg['scaler_scale'])
    weights = np.array(pkg['clf_coef'][0])
    intercept = pkg['clf_intercept'][0]

    # Standardize & compute per-sample posterior probability
    X_scaled = (X_np - mean) / scale
    logits = X_scaled @ weights + intercept
    probs_seen = 1.0 / (1.0 + np.exp(-logits))  # P(seen | x)
    probs_seen_t = torch.from_numpy(probs_seen).to(DEV).float()

    # Inductive Bayes-optimal decision rule (P(seen|x) >= 0.50)
    route_seen = (probs_seen_t >= 0.50)
    final_pred_indices = torch.where(route_seen, best_seen_pred, best_unseen_pred)

    n_routed_seen = route_seen.sum().item()
    n_routed_unseen = (~route_seen).sum().item()

    print(f"\n[Learned Logistic Gate - Inductive Inference]")
    print(f"  ├─ Decision Rule    : P(seen | x) >= 0.50 (Bayes Optimal, per-sample)")
    print(f"  ├─ Mean P(seen|x)   : {probs_seen_t.mean().item():.4f}")
    print(f"  ├─ Median P(seen|x) : {probs_seen_t.median().item():.4f}")
    print(f"  ├─ Routed to Seen   : {n_routed_seen:,} ({100*n_routed_seen/len(all_files):.2f}%)")
    print(f"  └─ Routed to Novel  : {n_routed_unseen:,} ({100*n_routed_unseen/len(all_files):.2f}%)")

    # 8. Build and Validate Prediction JSON
    preds = {fn: classes[idx.item()] for fn, idx in zip(all_files, final_pred_indices)}
    assert len(preds) == 35665
    assert all(c in ci for c in preds.values())

    seen_set = set(seen_classes)
    predicted_unseen = set(p for p in preds.values() if p not in seen_set)
    print(f"\nPrediction Summary:")
    print(f"  ├─ Total Predictions       : {len(preds):,}")
    print(f"  ├─ Distinct Unseen Species : {len(predicted_unseen):,} / {len(unseen_classes):,}")

    out_json = os.path.join(OUT, 'prediction_v73_learned_multimodal_pipeline.json')
    with open(out_json, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f"  ├─ Saved JSON: {out_json}")

    # 9. Package Submission Zip
    out_zip = os.path.join(SUB, 'submission_v73_learned_multimodal_pipeline.zip')
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname='prediction.json')

    print(f"\n[SUCCESS] Submission ready at {out_zip} ({os.path.getsize(out_zip):,} bytes)")
    print("=" * 85)


if __name__ == '__main__':
    main()
