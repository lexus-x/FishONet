"""v72: Full Implementation of Professor's Proposed Pipeline with Trained 2-Class Logistic Gate.

Implements every detail from the handwritten diagram:
1. Shared Embedding Space:
   - Vision Embedding: z_i = BioCLIP_vision(x) in R^d (L2 normalized)
   - Text Embeddings: z_T for all 17,393 seen + unseen classes (prefixed)
   - Seen Prototypes P_c: mean of training image embeddings with L2 norm before & after mean.

2. Two Initial Classifier Decisions:
   - best seen candidate:   c* = argmax_{c in seen} cos(z_i, P_c)
   - best unseen candidate: u* = argmax_{u in unseen} cos(z_i, z_{Tu})

3. Learned Logistic Regression Soft Gate:
   - Uses the 13 debate features (prototype similarity, text similarities, margins, Shannon entropy, normalized sums)
   - Calibrated probability s = sigma(W^T x_i + b) in [0, 1]
   - Decision: if s >= t -> best seen candidate, else -> best unseen candidate.

Usage:
  /home/ubuntu/miniconda3/envs/onet/bin/python builders/build_v72_professor_logistic_pipeline.py
"""
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


def compute_entropy_bits(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p_safe = p.clamp(min=eps)
    return -torch.sum(p_safe * torch.log2(p_safe), dim=-1)


def extract_features_vector(
    z: torch.Tensor,
    P_seen: torch.Tensor,
    T_seen: torch.Tensor,
    T_unseen: torch.Tensor,
    temp: float = 0.0300
) -> torch.Tensor:
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

    feature_matrix = torch.stack([
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

    return feature_matrix


def load_emb(path: str):
    d = torch.load(path, map_location='cpu', weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def main():
    print("=" * 85)
    print("=== v72: Building Submission with Professor's Proposed Pipeline ===")
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

    # 2. Build Seen Class Prototypes (L2 norm before and after mean)
    print("Computing Seen Class Prototypes (P_seen) with L2 norm before & after mean...")
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

    # 4. Load Evaluation Images (Test + Unseen splits combined = 35,665 blind test images)
    print("Loading 35,665 Evaluation Images...")
    idx_te, feats_te, files_te = load_emb(os.path.join(OUT, 'emb_test_ctftshift.pt'))
    idx_un, feats_un, files_un = load_emb(os.path.join(OUT, 'emb_unseen_ctftshift.pt'))

    all_files = files_te + files_un
    assert len(all_files) == 35665
    Q_eval = torch.cat([feats_te, feats_un]).to(DEV)

    # 5. Compute Initial Candidate Decisions
    t0 = time.time()
    print("Computing initial decisions: best seen candidate vs best unseen candidate...")
    sim_P_s = Q_eval @ P_seen.t()
    sim_T_u = Q_eval @ T_unseen.t()

    best_seen_pred = seen_idx[sim_P_s.argmax(dim=-1)]
    best_unseen_pred = unseen_idx[sim_T_u.argmax(dim=-1)]

    # 6. Extract Debate Features & Evaluate Trained Logistic Gate
    print("Extracting debate features...")
    X_tensor = extract_features_vector(Q_eval, P_seen, T_seen, T_unseen, temp=0.0300)
    X_np = X_tensor.cpu().numpy()

    # Load trained logistic regression parameters
    pkg = json.load(open(os.path.join(OUT, 'professor_logistic_pipeline.json'), 'r'))
    mean = np.array(pkg['scaler_mean'])
    scale = np.array(pkg['scaler_scale'])
    weights = np.array(pkg['clf_coef'][0])
    intercept = pkg['clf_intercept'][0]

    X_scaled = (X_np - mean) / scale
    logits = X_scaled @ weights + intercept
    probs_seen = 1.0 / (1.0 + np.exp(-logits)) # P(seen | x)
    probs_seen_t = torch.from_numpy(probs_seen).to(DEV).float()

    # Sweep threshold / routing fraction
    # To protect against asymmetric seen ejection, we set threshold to retain ~65% seen fraction
    k_seen = int(round(0.65 * len(all_files)))
    thr_calibrated = torch.topk(probs_seen_t, k_seen).values.min().item()
    print(f"\nLogistic Gate Calibration:")
    print(f"  ├─ Mean P(seen|x)   : {probs_seen_t.mean().item():.4f}")
    print(f"  ├─ Median P(seen|x) : {probs_seen_t.median().item():.4f}")
    print(f"  └─ Calibrated Cutoff: {thr_calibrated:.4f} (Routing top 65.0% to Seen Head)")

    route_seen = (probs_seen_t >= thr_calibrated)
    final_pred_indices = torch.where(route_seen, best_seen_pred, best_unseen_pred)

    n_routed_seen = route_seen.sum().item()
    n_routed_unseen = (~route_seen).sum().item()

    print(f"\nRouting Statistics:")
    print(f"  ├─ Routed to Seen   : {n_routed_seen:,} ({100*n_routed_seen/len(all_files):.2f}%)")
    print(f"  └─ Routed to Novel  : {n_routed_unseen:,} ({100*n_routed_unseen/len(all_files):.2f}%)")

    # 7. Build Prediction Dictionary
    preds = {fn: classes[idx.item()] for fn, idx in zip(all_files, final_pred_indices)}
    assert len(preds) == 35665
    assert all(c in ci for c in preds.values())

    seen_set = set(seen_classes)
    predicted_unseen = set(p for p in preds.values() if p not in seen_set)
    print(f"\nPrediction Summary:")
    print(f"  ├─ Total Predictions       : {len(preds):,}")
    print(f"  ├─ Distinct Unseen Species : {len(predicted_unseen):,} / {len(unseen_classes):,}")
    print(f"  └─ Process Time            : {time.time()-t0:.2f}s")

    # 8. Package Submission
    tag = 'v72_professor_logistic_pipeline'
    os.makedirs('submissions', exist_ok=True)
    json_path = f'outputs/prediction_{tag}.json'
    zip_path = f'outputs/submission_{tag}.zip'
    sub_path = f'submissions/submission_{tag}.zip'

    with open(json_path, 'w') as f:
        json.dump(preds, f)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, 'prediction.json')

    shutil.copyfile(zip_path, sub_path)
    print(f"\n[SUCCESS] Submission ready at {sub_path} ({os.path.getsize(sub_path):,} bytes)")
    print("=" * 85)


if __name__ == '__main__':
    main()
