"""Train LoRA with Multimodal CosFace using BOTH Text and Image Embeddings.

Formulation (Prof. Ryu):
- Remove standard learnable class weights (W).
- Replace W with multimodal class anchors combining:
    1. L2-normalized Text Embeddings (T) across ALL 17,393 classes.
    2. L2-normalized Visual Prototypes (I) from iNaturalist + TreeOfLife.
- Multimodal Anchor: E_j = L2_norm(alpha * T_j + (1 - alpha) * I_j)
- Apply CosFace additive angular margin (cos(θ_y) - m) on the target class.
- Compute loss over ALL 17,393 candidate classes in the softmax denominator.

Usage:
  python src/train_lora_cosface.py --model bioclip-2 --scale 35.0 --margin 0.20 --alpha_txt 0.5 --max_steps 1000 --save outputs/b2_cosface_multimodal_lora.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
DATA = os.path.join(ROOT, 'data', 'dl')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_MAP = {
    'bioclip-2': 'hf-hub:imageomics/bioclip-2',
    'bioclip-2.5': 'hf-hub:imageomics/bioclip-2.5-vith14',
}

TEXT_EMB_MAP = {
    'bioclip-2': os.path.join(OUT, 'text_emb.pt'),
    'bioclip-2.5': os.path.join(OUT, 'text_emb_h_taxon.pt'),
}

IMG_PROTO_MAP = {
    'bioclip-2': os.path.join(OUT, 'inat_tol_merged_b2_a05.pt'),
    'bioclip-2.5': os.path.join(OUT, 'inat_protos_ctftshift.pt'),
}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
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


def inject_lora(model: nn.Module, r: int = 16, alpha: int = 32, top_k: int = 12) -> int:
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


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith('.A') or n.endswith('.B')}


def index_images(root: str) -> dict[str, str]:
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx[f] = os.path.join(dp, f)
    return idx


class TrainDataset(Dataset):
    def __init__(self, items: list[tuple[str, int]], img_idx: dict[str, str], transform):
        self.items = items
        self.img_idx = img_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        fn, y = self.items[idx]
        img_path = self.img_idx.get(fn, '')
        try:
            img = Image.open(img_path).convert('RGB')
            return self.transform(img), y
        except Exception:
            return torch.zeros(3, 224, 224), y


class ValDataset(Dataset):
    def __init__(self, items: list[dict], class_to_idx: dict[str, int], transform):
        self.valid_items = []
        for it in items:
            cname = it['class_name']
            if cname in class_to_idx:
                self.valid_items.append({
                    'image_path': it['image_path'],
                    'label': class_to_idx[cname],
                    'is_unseen': it.get('is_unseen', False)
                })
        self.transform = transform

    def __len__(self) -> int:
        return len(self.valid_items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, bool]:
        it = self.valid_items[idx]
        img = Image.open(it['image_path']).convert('RGB')
        return self.transform(img), it['label'], it['is_unseen']


def cosface_multimodal_loss(
    sims: torch.Tensor,
    targets: torch.Tensor,
    scale: float = 35.0,
    margin: float = 0.20
) -> torch.Tensor:
    """CosFace loss over multimodal similarity matrix.

    sims: [B, 17,393]
    targets: [B]
    """
    cos_y = sims.gather(1, targets.unsqueeze(1)).squeeze(1)
    cos_y_m = cos_y - margin

    sims_m = sims.clone()
    sims_m.scatter_(1, targets.unsqueeze(1), cos_y_m.unsqueeze(1))

    logits = scale * sims_m
    return F.cross_entropy(logits, targets)


def build_multimodal_anchors(
    text_path: str,
    img_proto_path: str,
    alpha_txt: float = 0.5
) -> tuple[torch.Tensor, list[str], dict[str, int]]:
    print(f"Loading Text Embeddings from {text_path}...")
    t_data = torch.load(text_path, map_location='cpu', weights_only=False)
    classes = t_data.get('classes', [])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    t_emb = None
    for k in ['emb_name', 'emb_taxon', 'emb_desc', 'feats', 'emb']:
        if k in t_data and isinstance(t_data[k], torch.Tensor):
            t_emb = t_data[k]
            break
    T_norm = F.normalize(t_emb.float(), dim=-1)

    print(f"Loading Image Prototypes from {img_proto_path}...")
    if os.path.isfile(img_proto_path):
        i_data = torch.load(img_proto_path, map_location='cpu', weights_only=False)
        i_emb = i_data.get('protos', i_data) if isinstance(i_data, dict) else i_data
        I_norm = F.normalize(i_emb.float(), dim=-1)

        # Detect non-zero image prototypes
        has_proto = (I_norm.norm(dim=-1) > 1e-4).unsqueeze(-1)
        print(f"  Visual Prototypes available for {has_proto.sum().item():,}/{len(classes):,} classes.")

        # Multimodal fusion: alpha * Text + (1 - alpha) * Image (for classes with images)
        E = torch.where(has_proto, alpha_txt * T_norm + (1.0 - alpha_txt) * I_norm, T_norm)
        E_all = F.normalize(E, dim=-1).to(DEV)
    else:
        print("  Warning: Image prototypes file not found, falling back to text embeddings only.")
        E_all = T_norm.to(DEV)

    print(f"  Multimodal Anchor Matrix: shape {E_all.shape}, dtype {E_all.dtype} on {DEV}")
    return E_all, classes, class_to_idx


def main():
    parser = argparse.ArgumentParser(description="Multimodal CosFace LoRA Training")
    parser.add_argument('--model', default='bioclip-2', choices=['bioclip-2', 'bioclip-2.5'],
                        help="Model architecture")
    parser.add_argument('--scale', type=float, default=35.0, help="CosFace scale factor (s)")
    parser.add_argument('--margin', type=float, default=0.20, help="CosFace additive margin (m)")
    parser.add_argument('--alpha_txt', type=float, default=0.50,
                        help="Weight for text embedding in multimodal anchor (1-alpha for image proto)")
    parser.add_argument('--rank', type=int, default=16, help="LoRA rank")
    parser.add_argument('--alpha', type=int, default=32, help="LoRA alpha")
    parser.add_argument('--top_k', type=int, default=12, help="LoRA top_k transformer blocks")
    parser.add_argument('--bs', type=int, default=128, help="Batch size")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--max_steps', type=int, default=1000, help="Max training steps")
    parser.add_argument('--val_every', type=int, default=100, help="Validation frequency")
    parser.add_argument('--workers', type=int, default=4, help="DataLoader workers")
    parser.add_argument('--save', default=os.path.join(OUT, 'b2_cosface_multimodal_lora.pt'),
                        help="Save path")
    parser.add_argument('--manifest', default=os.path.join(OUT, 'inat_val_manifest.json'),
                        help="Path to iNat validation manifest")
    parser.add_argument('--val_max_imgs', type=int, default=2000,
                        help="Max validation images for fast check")
    args = parser.parse_args()

    model_name = MODEL_MAP[args.model]
    text_path = TEXT_EMB_MAP[args.model]
    img_proto_path = IMG_PROTO_MAP[args.model]

    print(f"=== Multimodal CosFace LoRA Training (Both Image & Text Anchors) ===")
    print(f"  Model: {model_name}")
    print(f"  CosFace: scale={args.scale}, margin={args.margin}")
    print(f"  Multimodal Alpha (Text): {args.alpha_txt} | (Image): {1.0 - args.alpha_txt}")

    # 1. Build Multimodal Anchor Matrix (Removing original linear weights W)
    E_all, classes, class_to_idx = build_multimodal_anchors(text_path, img_proto_path, args.alpha_txt)

    # 2. Index Training Photos
    print(f"\nIndexing training dataset from {DATA}...")
    train_labels = json.load(open(os.path.join(DATA, 'label_train.json'), 'r'))
    img_idx = index_images(os.path.join(DATA, 'images'))

    train_items = []
    seen_classes = set()
    for fn, cname in train_labels.items():
        if fn in img_idx and cname in class_to_idx:
            train_items.append((fn, class_to_idx[cname]))
            seen_classes.add(cname)

    print(f"  Loaded {len(train_items):,} training photos across {len(seen_classes):,} seen classes.")

    # 3. Model & LoRA Injection
    print(f"\nInitializing {model_name}...")
    model, _, eval_transform = open_clip.create_model_and_transforms(model_name)
    num_lora_layers = inject_lora(model, r=args.rank, alpha=args.alpha, top_k=args.top_k)
    print(f"  Injected {num_lora_layers} LoRA layers into image encoder.")

    model = model.to(DEV)
    for n, p in model.named_parameters():
        p.requires_grad_(n.endswith('.A') or n.endswith('.B'))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # 4. Transforms & Loaders
    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.35, 1.0), ratio=(0.5, 2.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_dataset = TrainDataset(train_items, img_idx, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    # 5. Validation Setup
    val_items = []
    if os.path.isfile(args.manifest):
        with open(args.manifest, 'r') as f:
            val_data = json.load(f)
        val_items = val_data.get('items', [])
        if args.val_max_imgs > 0:
            val_items = val_items[:args.val_max_imgs]

    val_dataset = ValDataset(val_items, class_to_idx, eval_transform)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    @torch.no_grad()
    def evaluate_val() -> tuple[float, float, float]:
        model.eval()
        unseen_correct, seen_correct = 0, 0
        unseen_total, seen_total = 0, 0
        margins = []

        for images, labels, is_unseen in val_loader:
            images = images.to(DEV)
            labels = labels.to(DEV)
            is_unseen = is_unseen.to(DEV)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                f = F.normalize(model.encode_image(images).float(), dim=-1)
                sims = f @ E_all.t()

            preds = sims.argmax(dim=-1)
            correct = (preds == labels)

            pos_cos = sims.gather(1, labels.unsqueeze(1)).squeeze(1)
            sims_neg = sims.clone()
            sims_neg.scatter_(1, labels.unsqueeze(1), -float('inf'))
            max_neg = sims_neg.max(dim=1).values
            margins.extend((pos_cos - max_neg).cpu().tolist())

            unseen_mask = is_unseen.bool()
            seen_mask = ~unseen_mask

            if unseen_mask.any():
                unseen_correct += correct[unseen_mask].sum().item()
                unseen_total += unseen_mask.sum().item()
            if seen_mask.any():
                seen_correct += correct[seen_mask].sum().item()
                seen_total += seen_mask.sum().item()

        model.train()
        unseen_acc = (unseen_correct / unseen_total * 100) if unseen_total else 0.0
        seen_acc = (seen_correct / seen_total * 100) if seen_total else 0.0
        mean_margin = (sum(margins) / len(margins)) if margins else 0.0
        return unseen_acc, seen_acc, mean_margin

    print("\nEvaluating initial zero-shot validation baseline...")
    base_unseen, base_seen, base_margin = evaluate_val()
    print(f"  [BASELINE] Unseen Top-1: {base_unseen:.2f}% | Seen Top-1: {base_seen:.2f}% | Margin: {base_margin:.4f}")

    warmup_steps = max(1, int(0.08 * args.max_steps))

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return args.lr * step / warmup_steps
        p = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    # 6. Training Loop
    print("\n--- Starting Multimodal CosFace Training Loop ---")
    step = 0
    t0 = time.time()
    best_unseen = -1.0
    best_step = 0
    done = False

    while not done:
        for images, labels in train_loader:
            images = images.to(DEV)
            labels = labels.to(DEV)

            curr_lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = curr_lr

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                image_features = F.normalize(model.encode_image(images).float(), dim=-1)
                # Cosine similarity against ALL 17,393 multimodal anchors
                sims = image_features @ E_all.t()
                loss = cosface_multimodal_loss(sims, labels, scale=args.scale, margin=args.margin)

            loss.backward()
            optimizer.step()
            step += 1

            if step % 25 == 0:
                elapsed = time.time() - t0
                speed = step / max(elapsed, 1e-3)
                print(f"Step {step:4d}/{args.max_steps} | Loss: {loss.item():.4f} | LR: {curr_lr:.2e} | Speed: {speed:.1f} it/s", flush=True)

            if step % args.val_every == 0 or step == args.max_steps:
                val_unseen, val_seen, val_margin = evaluate_val()
                print(f"\n  >>> VAL @ Step {step}: Unseen Top-1 = {val_unseen:.2f}% | Seen Top-1 = {val_seen:.2f}% | Margin = {val_margin:.4f}", flush=True)

                if val_unseen > best_unseen:
                    best_unseen = val_unseen
                    best_step = step
                    save_payload = {
                        'state': lora_state_dict(model),
                        'top_k': args.top_k,
                        'rank': args.rank,
                        'alpha': args.alpha,
                        'model': model_name,
                        'scale': args.scale,
                        'margin': args.margin,
                        'alpha_txt': args.alpha_txt,
                        'unseen_acc': val_unseen,
                        'seen_acc': val_seen,
                        'step': step
                    }
                    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
                    torch.save(save_payload, args.save)
                    print(f"  [*] New Best Unseen Score! Saved checkpoint to {args.save}\n", flush=True)

            if step >= args.max_steps:
                done = True
                break

    total_time = time.time() - t0
    print("\n=================================================")
    print(f" Multimodal CosFace Training Complete in {total_time:.1f}s")
    print(f" Best Checkpoint @ Step {best_step}: Unseen Top-1 = {best_unseen:.2f}%")
    print(f" Checkpoint saved to: {args.save}")
    print("=================================================\n")


if __name__ == '__main__':
    main()
