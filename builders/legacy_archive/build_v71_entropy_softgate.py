"""v71: Clean Single-Backbone BioCLIP-2.5 with Simple Entropy-Based Soft Gate.

Implements the exact formula requested by Prof. Ryu (류.正.열) & Sai:
1. Encoder: LoRA-tuned BioCLIP-2.5 ViT-H/14 ('ctftshift')
2. Two initial decisions:
   - best seen class:   argmax_c cos(z, P_seen[c])
   - best unseen class: argmax_u cos(z, P_unseen[u])
3. Entropy-based soft gate:
   - p = softmax(cos(z, P_seen) / T)
   - H(p) = - sum_i p_i * log2(p_i)
4. Decision rule:
   - if H(p) > tau: prediction = best unseen class
   - else:          prediction = best seen class

Calibrated parameters (from validation holdout & iNat val):
  - T = 0.0300
  - tau = 1.9455 bits

Usage:
  /home/ubuntu/miniconda3/envs/onet/bin/python builders/build_v71_entropy_softgate.py
"""
import json
import math
import os
import pickle
import shutil
import time
import zipfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dl')
OUT = os.path.join(ROOT, 'outputs')

# Calibrated Hyperparameters
T_TEMP = 0.0300
TAU_BITS = 1.9455


def load_emb(path: str):
    d = torch.load(path, map_location='cpu', weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def compute_entropy_bits(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute Shannon entropy H(p) in bits: -sum_i p_i * log2(p_i)."""
    p_safe = p.clamp(min=eps)
    return -torch.sum(p_safe * torch.log2(p_safe), dim=-1)


def main():
    print("=" * 85)
    print("=== v71: Clean BioCLIP-2.5 Pipeline with Simple Entropy Soft Gate ===")
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

    print(f"Candidate Space: {NCLS:,} Total Classes")
    print(f"  ├─ Seen Classes in Training : {len(seen_classes):,}")
    print(f"  └─ Unseen Novel Classes     : {len(unseen_classes):,}")

    # 2. Load BioCLIP-2.5 Taxonomic Text Embeddings (T_unseen)
    print("\nLoading BioCLIP-2.5 Taxonomic Text Embeddings...")
    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), map_location='cpu', weights_only=False)['emb_taxon'].float(), dim=-1).to(DEV)
    P_unseen = TtH[unseen_idx]

    # 3. Load Training Features to compute P_seen
    print("Loading Seen Class Image Prototypes (P_seen)...")
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

    # 4. Load Evaluation Images (Test + Unseen splits combined = 35,665 blind test images)
    print("\nLoading Evaluation Image Embeddings (BioCLIP-2.5 LoRA)...")
    idx_te, feats_te, files_te = load_emb(os.path.join(OUT, 'emb_test_ctftshift.pt'))
    idx_un, feats_un, files_un = load_emb(os.path.join(OUT, 'emb_unseen_ctftshift.pt'))

    all_files = files_te + files_un
    assert len(all_files) == 35665, f"Expected 35,665 total eval files, got {len(all_files)}"

    Q_eval = torch.cat([feats_te, feats_un]).to(DEV)
    print(f"Total Evaluation Samples: {Q_eval.size(0):,}")

    # 5. Compute Seen and Unseen Similarities
    t0 = time.time()
    print("\nComputing similarities and initial predictions...")
    sim_seen = Q_eval @ P_seen.t()       # [N, C_seen]
    sim_unseen = Q_eval @ P_unseen.t()   # [N, C_unseen]

    top1_seen_pred = seen_idx[sim_seen.argmax(dim=-1)]
    top1_unseen_pred = unseen_idx[sim_unseen.argmax(dim=-1)]

    # 6. Compute Shannon Entropy Soft Gate
    print(f"Applying Shannon Entropy Soft Gate (T={T_TEMP:.4f}, tau={TAU_BITS:.4f} bits)...")
    p_seen = F.softmax(sim_seen / T_TEMP, dim=-1)
    H_p = compute_entropy_bits(p_seen)

    route_unseen = (H_p > TAU_BITS)
    final_pred_indices = torch.where(route_unseen, top1_unseen_pred, top1_seen_pred)

    n_routed_seen = (~route_unseen).sum().item()
    n_routed_unseen = route_unseen.sum().item()

    print(f"\nRouting Statistics:")
    print(f"  ├─ Mean Entropy H(p) : {H_p.mean().item():.3f} +/- {H_p.std().item():.3f} bits")
    print(f"  ├─ Median Entropy    : {H_p.median().item():.3f} bits")
    print(f"  ├─ Routed to Seen    : {n_routed_seen:,} ({100*n_routed_seen/len(all_files):.2f}%)")
    print(f"  └─ Routed to Unseen  : {n_routed_unseen:,} ({100*n_routed_unseen/len(all_files):.2f}%)")

    # 7. Format Prediction Dictionary & Verify Compliance
    preds = {fn: classes[idx.item()] for fn, idx in zip(all_files, final_pred_indices)}
    assert len(preds) == 35665
    assert all(c in ci for c in preds.values())

    seen_set = set(seen_classes)
    predicted_unseen_classes = set(p for p in preds.values() if p not in seen_set)

    print(f"\nPrediction Breakdown:")
    print(f"  ├─ Total Predictions       : {len(preds):,}")
    print(f"  ├─ Distinct Unseen Classes : {len(predicted_unseen_classes):,} / {len(unseen_classes):,}")
    print(f"  └─ Total Process Time      : {time.time()-t0:.2f}s")

    # 8. Package Submission Zip
    tag = 'v71_entropy_softgate'
    os.makedirs('submissions', exist_ok=True)
    json_path = f'outputs/prediction_{tag}.json'
    zip_path = f'outputs/submission_{tag}.zip'
    sub_path = f'submissions/submission_{tag}.zip'

    with open(json_path, 'w') as f:
        json.dump(preds, f)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, 'prediction.json')

    shutil.copyfile(zip_path, sub_path)
    print(f"\n[SUCCESS] Saved submission to {sub_path} ({os.path.getsize(sub_path):,} bytes)")
    print("=" * 85)


if __name__ == '__main__':
    main()
