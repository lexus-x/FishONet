"""Evaluate Baseline or LoRA Fine-Tuned Encoders on the iNaturalist Validation Set.

Supports:
  - imageomics/bioclip-2 (ViT-L/14, 768-d)
  - imageomics/bioclip-2.5-vith14 (ViT-H/14, 1024-d)
  - Standard Zero-Shot (no LoRA)
  - LoRA Fine-Tunes (Cross-Entropy, CosFace, Margin variants)

Usage:
  # 1. Zero-shot baseline on BioCLIP-2
  python scripts/eval_lora_inat.py --model bioclip-2

  # 2. Evaluate BioCLIP-2 LoRA checkpoint
  python scripts/eval_lora_inat.py --model bioclip-2 --lora_ckpt outputs/b2_shift_lora.pt

  # 3. Quick test on 1,000 images
  python scripts/eval_lora_inat.py --model bioclip-2 --lora_ckpt outputs/b2_shift_lora.pt --max_imgs 1000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_MAP = {
    'bioclip-2': 'hf-hub:imageomics/bioclip-2',
    'bioclip-2.5': 'hf-hub:imageomics/bioclip-2.5-vith14',
    'bioclip-1': 'hf-hub:imageomics/bioclip',
}

DEFAULT_TEXT_EMBS = {
    'bioclip-2': os.path.join(OUT, 'text_emb.pt'),
    'bioclip-2.5': os.path.join(OUT, 'text_emb_h_taxon.pt'),
    'bioclip-1': os.path.join(OUT, 'text_emb.pt'),
}


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
    """Injects LoRALinear into transformer resblocks."""
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
                    'split': it.get('split', 'unseen' if it.get('is_unseen', False) else 'seen')
                })
        self.transform = transform

    def __len__(self) -> int:
        return len(self.valid_items)

    def __getitem__(self, idx: int):
        item = self.valid_items[idx]
        img = Image.open(item['image_path']).convert('RGB')
        tensor = self.transform(img)
        return tensor, item['label'], item['is_unseen']


def load_text_embeddings(model_alias: str, custom_path: Optional[str] = None):
    emb_path = custom_path or DEFAULT_TEXT_EMBS.get(model_alias, os.path.join(OUT, 'text_emb.pt'))
    if not os.path.isfile(emb_path):
        raise FileNotFoundError(f"Text embeddings not found at {emb_path}")

    print(f"Loading text embeddings from {emb_path}...")
    data = torch.load(emb_path, map_location='cpu', weights_only=False)

    classes = data.get('classes', [])
    if not classes and 'classes' in data:
        classes = data['classes']

    # Locate embeddings tensor
    emb_tensor = None
    for k in ['emb_taxon', 'emb_name', 'emb_desc', 'feats', 'emb']:
        if k in data and isinstance(data[k], torch.Tensor):
            emb_tensor = data[k]
            print(f"  Using embedding key: '{k}' (shape {emb_tensor.shape})")
            break

    if emb_tensor is None:
        raise KeyError(f"Could not find valid embedding tensor in {emb_path}. Keys: {list(data.keys())}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    emb_tensor = F.normalize(emb_tensor.float(), dim=-1).to(DEV)
    return emb_tensor, classes, class_to_idx


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA on iNaturalist Validation Set")
    parser.add_argument('--model', default='bioclip-2', choices=['bioclip-2', 'bioclip-2.5', 'bioclip-1'],
                        help="Model architecture alias")
    parser.add_argument('--lora_ckpt', default='', help="Path to LoRA checkpoint (.pt). If empty, runs baseline.")
    parser.add_argument('--manifest', default=os.path.join(OUT, 'inat_val_manifest.json'),
                        help="Path to inat_val_manifest.json")
    parser.add_argument('--text_emb', default='', help="Optional custom text embeddings path")
    parser.add_argument('--batch_size', type=int, default=64, help="Inference batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader workers")
    parser.add_argument('--max_imgs', type=int, default=0, help="Limit number of validation images (0 = all)")
    parser.add_argument('--rank', type=int, default=16, help="LoRA rank")
    parser.add_argument('--alpha', type=int, default=32, help="LoRA alpha")
    parser.add_argument('--top_k', type=int, default=12, help="LoRA top_k blocks")
    parser.add_argument('--save_json', default='', help="Optional path to save JSON metrics")
    args = parser.parse_args()

    model_name = MODEL_MAP[args.model]

    # 1. Load Text Embeddings
    text_embs, classes, class_to_idx = load_text_embeddings(args.model, args.text_emb or None)
    num_classes = len(classes)
    print(f"Total candidate classes in text bank: {num_classes:,}")

    # 2. Load Validation Manifest
    if not os.path.isfile(args.manifest):
        raise FileNotFoundError(f"Manifest not found at {args.manifest}. Run scripts/build_inat_val_split.py first.")

    with open(args.manifest, 'r') as f:
        manifest_data = json.load(f)

    items = manifest_data.get('items', [])
    if args.max_imgs > 0:
        items = items[:args.max_imgs]
        print(f"Limiting evaluation to first {len(items):,} images")

    # 3. Create Model & Preprocessing
    print(f"\nLoading model: {model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)

    # 4. Inject & Load LoRA if specified
    if args.lora_ckpt:
        if not os.path.isfile(args.lora_ckpt):
            raise FileNotFoundError(f"LoRA checkpoint not found at {args.lora_ckpt}")

        ckpt = torch.load(args.lora_ckpt, map_location='cpu', weights_only=False)
        rank = ckpt.get('rank', args.rank)
        alpha = ckpt.get('alpha', args.alpha)
        top_k = ckpt.get('top_k', args.top_k)

        print(f"Injecting LoRA (rank={rank}, alpha={alpha}, top_k={top_k})...")
        num_layers = inject_lora(model, r=rank, alpha=alpha, top_k=top_k)
        print(f"  Injected {num_layers} LoRA Linear modules.")

        # Load weights
        state_dict = ckpt['state'] if 'state' in ckpt else ckpt
        # Clean parameter names if necessary
        cleaned_state = {}
        for k, v in state_dict.items():
            cleaned_state[k.replace('module.', '')] = v

        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        lora_keys_loaded = [k for k in cleaned_state.keys() if '.A' in k or '.B' in k]
        print(f"  Successfully loaded {len(lora_keys_loaded)} LoRA weight tensors from {args.lora_ckpt}")
    else:
        print("Running in zero-shot baseline mode (No LoRA injected).")

    model = model.to(DEV)
    model.eval()

    # 5. Dataset & DataLoader
    dataset = INatValDataset(items, class_to_idx, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Evaluation Dataset: {len(dataset):,} valid images mapped to target classes.")

    # 6. Evaluation Loop
    total_correct_top1 = 0
    total_correct_top5 = 0
    unseen_correct_top1 = 0
    unseen_correct_top5 = 0
    seen_correct_top1 = 0
    seen_correct_top5 = 0

    total_count = 0
    unseen_count = 0
    seen_count = 0

    margins = []

    t0 = time.time()
    print("\nStarting evaluation inference...")

    with torch.no_grad():
        for batch_idx, (images, labels, is_unseen) in enumerate(dataloader):
            images = images.to(DEV)
            labels = labels.to(DEV)
            is_unseen = is_unseen.to(DEV)

            with torch.amp.autocast('cuda'):
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features.float(), dim=-1)

            # Cosine similarity [B, C]
            sims = image_features @ text_embs.t()

            # Top-1 and Top-5 predictions
            top5_preds = sims.topk(5, dim=-1).indices
            top1_preds = top5_preds[:, 0]

            correct1 = (top1_preds == labels)
            correct5 = (top5_preds == labels.unsqueeze(1)).any(dim=-1)

            # Compute margin: positive cosine - max negative cosine
            pos_cos = sims.gather(1, labels.unsqueeze(1)).squeeze(1)
            # Mask out ground truth for max negative
            sims_neg = sims.clone()
            sims_neg.scatter_(1, labels.unsqueeze(1), -float('inf'))
            max_neg_cos = sims_neg.max(dim=1).values
            margin = pos_cos - max_neg_cos
            margins.extend(margin.cpu().tolist())

            # Accumulate overall
            total_correct_top1 += correct1.sum().item()
            total_correct_top5 += correct5.sum().item()
            total_count += labels.size(0)

            # Accumulate split-wise
            unseen_mask = is_unseen.bool()
            seen_mask = ~unseen_mask

            if unseen_mask.any():
                unseen_correct_top1 += correct1[unseen_mask].sum().item()
                unseen_correct_top5 += correct5[unseen_mask].sum().item()
                unseen_count += unseen_mask.sum().item()

            if seen_mask.any():
                seen_correct_top1 += correct1[seen_mask].sum().item()
                seen_correct_top5 += correct5[seen_mask].sum().item()
                seen_count += seen_mask.sum().item()

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(dataloader):
                elapsed = time.time() - t0
                fps = total_count / max(elapsed, 1e-3)
                print(f"  Processed {total_count:,}/{len(dataset):,} images ({total_count/len(dataset)*100:.1f}%) "
                      f"[{fps:.1f} img/s] - Current Top-1: {total_correct_top1/total_count*100:.2f}%", flush=True)

    elapsed = time.time() - t0

    # 7. Compute Summary Metrics
    overall_top1 = (total_correct_top1 / total_count * 100) if total_count else 0.0
    overall_top5 = (total_correct_top5 / total_count * 100) if total_count else 0.0

    unseen_top1 = (unseen_correct_top1 / unseen_count * 100) if unseen_count else 0.0
    unseen_top5 = (unseen_correct_top5 / unseen_count * 100) if unseen_count else 0.0

    seen_top1 = (seen_correct_top1 / seen_count * 100) if seen_count else 0.0
    seen_top5 = (seen_correct_top5 / seen_count * 100) if seen_count else 0.0

    mean_margin = sum(margins) / len(margins) if margins else 0.0

    # 8. Formatted Report Output
    title = f"iNaturalist Validation Benchmark: {args.model}"
    if args.lora_ckpt:
        title += f" (LoRA: {os.path.basename(args.lora_ckpt)})"
    else:
        title += " (Zero-Shot Baseline)"

    print("\n" + "=" * 68)
    print(f" {title.center(66)} ")
    print("=" * 68)
    print(f"{'Split / Metric':<30} | {'Top-1 Acc (%)':<15} | {'Top-5 Acc (%)':<15}")
    print("-" * 68)
    print(f"{'OVERALL (All Valid Images)':<30} | {overall_top1:>13.2f}% | {overall_top5:>13.2f}%")
    print(f"{'UNSEEN (Zero-Shot Novel Species)':<30} | {unseen_top1:>13.2f}% | {unseen_top5:>13.2f}%")
    print(f"{'SEEN (Domain Transfer)':<30} | {seen_top1:>13.2f}% | {seen_top5:>13.2f}%")
    print("-" * 68)
    print(f"{'Mean Cosine Separation Margin':<30} | {mean_margin:>14.4f} | {'-':>15}")
    print(f"{'Total Evaluated Images':<30} | {total_count:>14,d} | {'-':>15}")
    print(f"{'Evaluation Time':<30} | {elapsed:>12.1f}s | {total_count/max(elapsed, 1e-3):>11.1f} img/s")
    print("=" * 68 + "\n")

    # 9. Optional JSON output
    if args.save_json:
        result_dict = {
            "model": args.model,
            "lora_ckpt": args.lora_ckpt,
            "overall_top1": overall_top1,
            "overall_top5": overall_top5,
            "unseen_top1": unseen_top1,
            "unseen_top5": unseen_top5,
            "seen_top1": seen_top1,
            "seen_top5": seen_top5,
            "mean_margin": mean_margin,
            "total_images": total_count,
            "unseen_images": unseen_count,
            "seen_images": seen_count,
            "elapsed_sec": elapsed
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"Results saved to {args.save_json}")


if __name__ == '__main__':
    main()
