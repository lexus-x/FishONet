"""Empirical Evaluation and Optimization of Simple Entropy-Based Soft Gate.

Requested by Prof. Ryu (류.正.열) & Sai:
1. Replace handcrafted multi-score/learned gate with pure Shannon entropy H(p).
2. Two initial decisions: best seen class, best unseen class.
3. Compute seen probabilities: p = softmax(cos(z, P_seen) / T).
4. Compute entropy: H(p) = -sum(p_i * log2(p_i)).
5. Decision: if H(p) > tau -> best unseen pred, else -> best seen pred.
6. Calibrate (T, tau) on validation set and evaluate vs 46.3% baseline.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dl')
OUT = os.path.join(ROOT, 'outputs')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_emb(path: str):
    d = torch.load(path, map_location='cpu', weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def compute_entropy_bits(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute Shannon entropy H(p) in bits: -sum_i p_i * log2(p_i)."""
    p_safe = p.clamp(min=eps)
    return -torch.sum(p_safe * torch.log2(p_safe), dim=-1)


def evaluate_entropy_gate_on_split(
    Q_seen_val: torch.Tensor,
    y_seen_val: torch.Tensor,
    Q_unseen_val: torch.Tensor,
    y_unseen_val: torch.Tensor,
    P_seen: torch.Tensor,
    P_unseen: torch.Tensor,
    seen_class_indices: torch.Tensor,
    unseen_class_indices: torch.Tensor,
    T_sweep: list[float],
    tau_percentiles: list[float],
    w_seen_pop: float = 0.5635,
    w_unseen_pop: float = 0.4365,
):
    """Evaluates the entropy soft gate across a 2D grid of temperatures T and thresholds tau."""
    print("\n" + "=" * 80)
    print("=== SWEEPING TEMPERATURE T AND ENTROPY THRESHOLD TAU ===")
    print("=" * 80)
    print(f"Candidate Space: {len(seen_class_indices):,} Seen Classes | {len(unseen_class_indices):,} Unseen Classes")
    print(f"Validation Size: {Q_seen_val.size(0):,} Seen Val Samples | {Q_unseen_val.size(0):,} Novel Val Samples")
    print(f"Population Weights: w_seen = {w_seen_pop:.4f}, w_unseen = {w_unseen_pop:.4f}")

    # 1. Cosine similarities
    sim_seen_s = Q_seen_val @ P_seen.t()     # [N_s, C_seen]
    sim_unseen_s = Q_seen_val @ P_unseen.t() # [N_s, C_unseen]

    sim_seen_u = Q_unseen_val @ P_seen.t()     # [N_u, C_seen]
    sim_unseen_u = Q_unseen_val @ P_unseen.t() # [N_u, C_unseen]

    # 2. Initial decisions: best seen class & best unseen class
    top1_seen_s = seen_class_indices[sim_seen_s.argmax(dim=-1)]       # [N_s]
    top1_unseen_s = unseen_class_indices[sim_unseen_s.argmax(dim=-1)] # [N_s]

    top1_seen_u = seen_class_indices[sim_seen_u.argmax(dim=-1)]       # [N_u]
    top1_unseen_u = unseen_class_indices[sim_unseen_u.argmax(dim=-1)] # [N_u]

    # Baseline 1: Flat zero-shot across all classes (no gating)
    sim_all_s = torch.cat([sim_seen_s, sim_unseen_s], dim=1)
    sim_all_u = torch.cat([sim_seen_u, sim_unseen_u], dim=1)
    all_indices = torch.cat([seen_class_indices, unseen_class_indices])

    acc_s_flat = (all_indices[sim_all_s.argmax(dim=1)] == y_seen_val).float().mean().item()
    acc_u_flat = (all_indices[sim_all_u.argmax(dim=1)] == y_unseen_val).float().mean().item()
    overall_flat = w_seen_pop * acc_s_flat + w_unseen_pop * acc_u_flat

    # Baseline 2: Pure seen-head oracle accuracy on seen queries & unseen-head oracle on unseen queries
    oracle_seen_head_acc = (top1_seen_s == y_seen_val).float().mean().item()
    oracle_unseen_head_acc = (top1_unseen_u == y_unseen_val).float().mean().item()
    upper_bound_oracle = w_seen_pop * oracle_seen_head_acc + w_unseen_pop * oracle_unseen_head_acc

    print("\n--- Benchmark Reference Baselines ---")
    print(f"Flat Zero-Shot (No Gate): Seen {acc_s_flat*100:.2f}% | Unseen {acc_u_flat*100:.2f}% | Overall: {overall_flat*100:.2f}%")
    print(f"Specialist Head Upper Bound (Oracle Gate): Seen {oracle_seen_head_acc*100:.2f}% | Unseen {oracle_unseen_head_acc*100:.2f}% | Ceiling: {upper_bound_oracle*100:.2f}%")

    # Grid search results
    best_res = None
    results_table = []

    for T in T_sweep:
        # Softmax probabilities over seen classes
        p_s = F.softmax(sim_seen_s / T, dim=-1)
        p_u = F.softmax(sim_seen_u / T, dim=-1)

        # Shannon entropy H(p) in bits
        H_s = compute_entropy_bits(p_s)
        H_u = compute_entropy_bits(p_u)

        # Theoretical max entropy = log2(C_seen)
        H_max = math.log2(len(seen_class_indices))

        mean_Hs, std_Hs = H_s.mean().item(), H_s.std().item()
        mean_Hu, std_Hu = H_u.mean().item(), H_u.std().item()

        # Combine entropy to pick threshold percentiles
        all_H = torch.cat([H_s, H_u])
        
        best_T_res = None

        for q in tau_percentiles:
            tau = torch.quantile(all_H, q).item()

            # Gate decision: H(p) > tau -> route to unseen, else seen
            route_unseen_s = (H_s > tau)
            route_unseen_u = (H_u > tau)

            # Final prediction
            pred_s = torch.where(route_unseen_s, top1_unseen_s, top1_seen_s)
            pred_u = torch.where(route_unseen_u, top1_unseen_u, top1_seen_u)

            acc_s = (pred_s == y_seen_val).float().mean().item()
            acc_u = (pred_u == y_unseen_val).float().mean().item()
            overall = w_seen_pop * acc_s + w_unseen_pop * acc_u

            seen_to_seen_route = (~route_unseen_s).float().mean().item()
            novel_to_novel_route = route_unseen_u.float().mean().item()

            rec = {
                'T': T,
                'tau': tau,
                'tau_q': q,
                'mean_Hs': mean_Hs,
                'mean_Hu': mean_Hu,
                'H_max': H_max,
                'acc_s': acc_s,
                'acc_u': acc_u,
                'overall': overall,
                'seen_route_purity': seen_to_seen_route,
                'novel_route_purity': novel_to_novel_route,
            }

            if best_T_res is None or overall > best_T_res['overall']:
                best_T_res = rec

            if best_res is None or overall > best_res['overall']:
                best_res = rec

        results_table.append(best_T_res)

    print("\n" + "=" * 80)
    print(f"{'T (Temp)':<8} | {'Opt tau (bits)':<14} | {'Seen H (bits)':<14} | {'Novel H (bits)':<14} | {'Seen Acc':<10} | {'Unseen Acc':<11} | {'Overall Acc':<12}")
    print("-" * 80)
    for r in results_table:
        print(f"{r['T']:<8.3f} | {r['tau']:<14.3f} | {r['mean_Hs']:<14.3f} | {r['mean_Hu']:<14.3f} | {r['acc_s']*100:>8.2f}% | {r['acc_u']*100:>9.2f}% | {r['overall']*100:>9.2f}%")
    print("=" * 80)

    print(f"\n[OPTIMAL ENTROPY CONFIGURATION]")
    print(f"  ├─ Best Temperature T     : {best_res['T']:.4f}")
    print(f"  ├─ Best Entropy Threshold : {best_res['tau']:.4f} bits (max capacity: {best_res['H_max']:.2f} bits)")
    print(f"  ├─ Seen Query Entropy      : {best_res['mean_Hs']:.3f} +/- {std_Hs:.3f} bits")
    print(f"  ├─ Novel Query Entropy     : {best_res['mean_Hu']:.3f} +/- {std_Hu:.3f} bits")
    print(f"  ├─ Routing Discrimination  : Seen->Seen {best_res['seen_route_purity']*100:.1f}% | Novel->Novel {best_res['novel_route_purity']*100:.1f}%")
    print(f"  ├─ Validation Seen Acc     : {best_res['acc_s']*100:.2f}%")
    print(f"  ├─ Validation Unseen Acc   : {best_res['acc_u']*100:.2f}%")
    print(f"  └─ Validation Overall Acc  : {best_res['overall']*100:.2f}% (vs Baseline Flat {overall_flat*100:.2f}%, Delta: {(best_res['overall']-overall_flat)*100:+.2f}%)")

    return best_res, results_table, overall_flat


def main():
    print("=" * 85)
    print("=== Entropy-Based Soft Gate Optimization (BioCLIP-2.5 LoRA) ===")
    print("=" * 85)

    # 1. Load taxonomy & training labels
    classes = list(pickle.load(open(os.path.join(DATA, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(os.path.join(DATA, 'label_train.json'), 'r'))

    # Load BioCLIP-2.5 ViT-H Taxonomic Text Embeddings
    print("Loading BioCLIP-2.5 Taxonomic Text Embeddings...")
    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), map_location='cpu', weights_only=False)['emb_taxon'].float(), dim=-1).to(DEV)

    # 2. Load BioCLIP-2.5 LoRA Image Query Embeddings (train only — the eval
    # splits pseudo-novel holdout out of train). EMB_TRAIN env var swaps models.
    emb_train = os.environ.get('EMB_TRAIN', 'emb_train_ctftshift.pt')
    print(f"Loading image embeddings: {emb_train}")
    idx_tr, feats_tr, files_tr = load_emb(os.path.join(OUT, emb_train))

    # 3. Setup Internal Holdout Split (20% Rarest Classes as Pseudo-Unseen)
    by = defaultdict(list)
    for fn in files_tr:
        if fn in lab and lab[fn] in ci:
            by[lab[fn]].append(fn)

    seen = sorted(by.keys())
    order = sorted(seen, key=lambda c: len(by[c]))

    n_pseudo = int(len(seen) * 0.20)
    pseudo_classes = set(order[:n_pseudo])
    kept_classes = [c for c in seen if c not in pseudo_classes]

    kept_idx = torch.tensor([ci[c] for c in kept_classes], device=DEV)
    pseudo_idx = torch.tensor([ci[c] for c in pseudo_classes], device=DEV)

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

    Q_s = torch.stack([feats_tr[idx_tr[fn]] for fn in seen_val_fns]).to(DEV)
    Q_u = torch.stack([feats_tr[idx_tr[fn]] for fn in novel_val_fns]).to(DEV)

    y_s = torch.tensor([ci[lab[fn]] for fn in seen_val_fns], device=DEV)
    y_u = torch.tensor([ci[lab[fn]] for fn in novel_val_fns], device=DEV)

    # 4. Build Seen Prototypes P_seen (from training images) & Novel Prototypes P_unseen (from Text / iNat)
    print(f"Building class prototypes for {len(kept_classes):,} kept seen classes...")
    P_seen = torch.zeros(len(kept_classes), feats_tr.size(1), device=DEV)
    cnt_seen = torch.zeros(len(kept_classes), device=DEV)
    for c in kept_classes:
        for fn in by[c]:
            if fn in seen_train_fns:
                f = feats_tr[idx_tr[fn]].to(DEV)
                P_seen[kept_to_idx[c]] += f
                cnt_seen[kept_to_idx[c]] += 1
    P_seen = F.normalize(P_seen / cnt_seen.clamp(min=1).unsqueeze(1), dim=-1)

    # Unseen anchors: Text embeddings T_unseen
    P_unseen = TtH[pseudo_idx]

    # 5. Run temperature & threshold sweep
    T_sweep = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50]
    tau_qs = list(np.linspace(0.1, 0.9, 33))

    best_res, table, baseline_flat = evaluate_entropy_gate_on_split(
        Q_seen_val=Q_s,
        y_seen_val=y_s,
        Q_unseen_val=Q_u,
        y_unseen_val=y_u,
        P_seen=P_seen,
        P_unseen=P_unseen,
        seen_class_indices=kept_idx,
        unseen_class_indices=pseudo_idx,
        T_sweep=T_sweep,
        tau_percentiles=tau_qs,
    )

    # Save summary report
    out_report = {
        'model': f'BioCLIP-2.5 ViT-H/14 ({emb_train})',
        'baseline_flat_accuracy': baseline_flat,
        'best_temperature': best_res['T'],
        'best_entropy_threshold': best_res['tau'],
        'best_seen_accuracy': best_res['acc_s'],
        'best_unseen_accuracy': best_res['acc_u'],
        'best_overall_accuracy': best_res['overall'],
        'delta_vs_baseline': best_res['overall'] - baseline_flat,
        'seen_entropy_mean': best_res['mean_Hs'],
        'unseen_entropy_mean': best_res['mean_Hu'],
        'entropy_theoretical_max': best_res['H_max'],
        'results_by_temperature': table,
    }

    report_tag = emb_train.removeprefix('emb_train_').removesuffix('.pt')
    report_path = os.path.join(OUT, 'entropy_softgate_evaluation.json' if report_tag == 'ctftshift'
                               else f'entropy_softgate_evaluation_{report_tag}.json')
    with open(report_path, 'w') as f:
        json.dump(out_report, f, indent=2)
    print(f"\nResults saved to {report_path}")


if __name__ == '__main__':
    main()
