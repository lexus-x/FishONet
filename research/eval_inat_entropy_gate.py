"""Evaluate Simple Entropy-Based Soft Gate on the 19,409 iNaturalist Validation Set.

Measures:
1. Seen species (domain transfer) accuracy
2. Unseen species (zero-shot novel classes) accuracy
3. Overall accuracy vs flat baseline
4. Entropy distributions H(p) for seen vs unseen iNat queries.
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dl')
OUT = os.path.join(ROOT, 'outputs')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / max(r, 1)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scaling * self.lora_scale


def inject_lora(model: nn.Module, r: int = 16, alpha: int = 32, top_k: int = 0) -> int:
    blocks = model.visual.transformer.resblocks
    sel = range(len(blocks)) if top_k <= 0 else range(len(blocks) - top_k, len(blocks))
    n = 0
    for i in sel:
        blk = blocks[i]
        if hasattr(blk, 'mlp'):
            blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, r, alpha)
            blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, r, alpha)
            n += 2
    return n


class INatValDataset(Dataset):
    def __init__(self, items: list[dict], class_to_idx: dict[str, int], transform):
        self.valid_items = []
        for it in items:
            cname = it['class_name']
            if cname in class_to_idx:
                self.valid_items.append({
                    'image_path': it['image_path'],
                    'label': class_to_idx[cname],
                    'class_name': cname,
                    'is_unseen': it.get('is_unseen', False),
                })
        self.transform = transform

    def __len__(self) -> int:
        return len(self.valid_items)

    def __getitem__(self, idx: int):
        it = self.valid_items[idx]
        img = Image.open(it['image_path']).convert('RGB')
        return self.transform(img), it['label'], it['is_unseen']


def compute_entropy_bits(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p_safe = p.clamp(min=eps)
    return -torch.sum(p_safe * torch.log2(p_safe), dim=-1)


def main():
    print("=" * 80)
    print("=== iNaturalist Validation Benchmark: Simple Entropy Soft Gate ===")
    print("=" * 80)

    # 1. Load taxonomy & training labels
    classes = list(pickle.load(open(os.path.join(DATA, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(os.path.join(DATA, 'label_train.json'), 'r'))

    seen_classes = sorted(set(lab.values()))
    unseen_classes = sorted(set(classes) - set(seen_classes))

    seen_idx = torch.tensor([ci[c] for c in seen_classes], device=DEV)
    unseen_idx = torch.tensor([ci[c] for c in unseen_classes], device=DEV)

    seen_to_pos = {ci[c]: j for j, c in enumerate(seen_classes)}
    unseen_to_pos = {ci[c]: j for j, c in enumerate(unseen_classes)}

    print(f"Total Competition Classes: {len(classes):,}")
    print(f"  ├─ Seen Classes in Training : {len(seen_classes):,}")
    print(f"  └─ Unseen Novel Classes     : {len(unseen_classes):,}")

    # 2. Load BioCLIP-2.5 Text Embeddings & Precomputed Train Embeddings to build P_seen
    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), map_location='cpu', weights_only=False)['emb_taxon'].float(), dim=-1).to(DEV)
    
    # Load training image embeddings to compute Seen Prototypes
    print("\nLoading seen class image features...")
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

    # Unseen Anchors: Text Embeddings
    P_unseen = TtH[unseen_idx]

    # 3. Load Model (BioCLIP-2.5 ViT-H/14 with ctftshift / LoRA)
    manifest_path = os.path.join(OUT, 'inat_val_manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"\nLoading BioCLIP-2.5 ViT-H/14 base model...")
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2.5-vith14')

    # Load ctft_lora or inject if lora checkpoint exists
    lora_path = os.path.join(OUT, 'ctft_lora.pt')
    if os.path.isfile(lora_path):
        ckpt = torch.load(lora_path, map_location='cpu', weights_only=False)
        rank = ckpt.get('rank', 16)
        alpha = ckpt.get('alpha', 32)
        top_k = ckpt.get('top_k', 12)
        print(f"Injecting LoRA ({rank=}, {alpha=}, {top_k=})...")
        inject_lora(model, r=rank, alpha=alpha, top_k=top_k)
        state = ckpt.get('state', ckpt)
        cleaned_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(cleaned_state, strict=False)
        print(f"Loaded LoRA weights from {lora_path}")
    else:
        print("Using base encoder weights for extraction.")

    model = model.to(DEV)
    model.eval()

    dataset = INatValDataset(manifest['items'], ci, preprocess)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    print(f"iNaturalist Validation Images: {len(dataset):,}")

    # 4. Extract Embeddings & Labels
    all_z = []
    all_y = []
    all_is_unseen = []

    t0 = time.time()
    print("Extracting validation embeddings...")
    with torch.no_grad():
        for imgs, labels, is_uns in loader:
            imgs = imgs.to(DEV)
            with torch.amp.autocast('cuda'):
                z = model.encode_image(imgs)
                z = F.normalize(z.float(), dim=-1)
            all_z.append(z.cpu())
            all_y.append(labels)
            all_is_unseen.append(is_uns)

    all_z = torch.cat(all_z).to(DEV)
    all_y = torch.cat(all_y).to(DEV)
    all_is_unseen = torch.cat(all_is_unseen).to(DEV)
    print(f"Extracted {all_z.size(0):,} embeddings in {time.time()-t0:.1f}s")

    # 5. Evaluate Flat Zero-Shot vs Entropy Soft Gate
    sim_seen = all_z @ P_seen.t()       # [N, C_seen]
    sim_unseen = all_z @ P_unseen.t()   # [N, C_unseen]

    top1_seen_pred = seen_idx[sim_seen.argmax(dim=-1)]
    top1_unseen_pred = unseen_idx[sim_unseen.argmax(dim=-1)]

    # Flat Zero-Shot Baseline (Concatenating all classes)
    sim_all = torch.cat([sim_seen, sim_unseen], dim=-1)
    all_cand_idx = torch.cat([seen_idx, unseen_idx])
    flat_preds = all_cand_idx[sim_all.argmax(dim=-1)]

    uns_mask = all_is_unseen.bool()
    seen_mask = ~uns_mask

    flat_seen_acc = (flat_preds[seen_mask] == all_y[seen_mask]).float().mean().item()
    flat_uns_acc = (flat_preds[uns_mask] == all_y[uns_mask]).float().mean().item()
    flat_overall_acc = (flat_preds == all_y).float().mean().item()

    print("\n--- Baseline Flat Model Performance (No Gate) ---")
    print(f"Seen Accuracy   : {flat_seen_acc*100:.2f}%")
    print(f"Unseen Accuracy : {flat_uns_acc*100:.2f}%")
    print(f"Overall Accuracy: {flat_overall_acc*100:.2f}%")

    # 6. Evaluate Entropy Soft Gate across Temperatures
    print("\n" + "=" * 80)
    print("=== SWEEPING (T, tau) ON iNATURALIST VALIDATION SET ===")
    print("=" * 80)
    print(f"{'T (Temp)':<8} | {'Opt tau (bits)':<14} | {'Seen H (bits)':<14} | {'Unseen H (bits)':<15} | {'Seen Acc':<10} | {'Unseen Acc':<11} | {'Overall Acc':<12}")
    print("-" * 80)

    best_inat_res = None
    T_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20]

    for T in T_list:
        p = F.softmax(sim_seen / T, dim=-1)
        H = compute_entropy_bits(p)

        mean_Hs = H[seen_mask].mean().item()
        mean_Hu = H[uns_mask].mean().item()

        # Sweep tau percentiles
        best_T_res = None
        for q in np.linspace(0.05, 0.95, 37):
            tau = torch.quantile(H, q).item()
            route_unseen = (H > tau)
            preds = torch.where(route_unseen, top1_unseen_pred, top1_seen_pred)

            acc_s = (preds[seen_mask] == all_y[seen_mask]).float().mean().item()
            acc_u = (preds[uns_mask] == all_y[uns_mask]).float().mean().item()
            acc_all = (preds == all_y).float().mean().item()

            rec = {
                'T': T,
                'tau': tau,
                'acc_s': acc_s,
                'acc_u': acc_u,
                'acc_all': acc_all,
                'mean_Hs': mean_Hs,
                'mean_Hu': mean_Hu,
            }

            if best_T_res is None or acc_all > best_T_res['acc_all']:
                best_T_res = rec

            if best_inat_res is None or acc_all > best_inat_res['acc_all']:
                best_inat_res = rec

        print(f"{best_T_res['T']:<8.3f} | {best_T_res['tau']:<14.3f} | {best_T_res['mean_Hs']:<14.3f} | {best_T_res['mean_Hu']:<15.3f} | {best_T_res['acc_s']*100:>8.2f}% | {best_T_res['acc_u']*100:>9.2f}% | **{best_T_res['acc_all']*100:>9.2f}%**")

    print("=" * 80)
    print(f"\n[OPTIMAL iNATURALIST ENTROPY GATE RESULTS]")
    print(f"  ├─ Best Temperature T     : {best_inat_res['T']:.4f}")
    print(f"  ├─ Best Threshold tau     : {best_inat_res['tau']:.4f} bits")
    print(f"  ├─ Seen Entropy           : {best_inat_res['mean_Hs']:.3f} bits")
    print(f"  ├─ Unseen Entropy         : {best_inat_res['mean_Hu']:.3f} bits")
    print(f"  ├─ Seen Accuracy          : {best_inat_res['acc_s']*100:.2f}% (vs Flat {flat_seen_acc*100:.2f}%)")
    print(f"  ├─ Unseen Accuracy        : {best_inat_res['acc_u']*100:.2f}% (vs Flat {flat_uns_acc*100:.2f}%)")
    print(f"  └─ Overall Accuracy       : {best_inat_res['acc_all']*100:.2f}% (vs Flat {flat_overall_acc*100:.2f}%, Delta: +{(best_inat_res['acc_all']-flat_overall_acc)*100:.2f}%)")

    # Save results
    save_dict = {
        'flat_baseline': {
            'seen_acc': flat_seen_acc,
            'unseen_acc': flat_uns_acc,
            'overall_acc': flat_overall_acc,
        },
        'best_entropy_gate': best_inat_res,
    }
    with open(os.path.join(OUT, 'inat_val_entropy_gate_results.json'), 'w') as f:
        json.dump(save_dict, f, indent=2)
    print(f"Saved results to outputs/inat_val_entropy_gate_results.json")


if __name__ == '__main__':
    main()
