#!/bin/bash
# Full-network fine-tune BioCLIP-2.5 ViT-H at 336px resolution
# Uses gradient checkpointing + bfloat16 autocast to fit L40S 46GB
# Runs on the onet conda environment

source ~/miniconda3/etc/profile.d/conda.sh
conda activate onet
cd ~/onet

echo "=== FULL-NETWORK FT 336px ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Free VRAM: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

python -c "
import argparse, os, json, math, random
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip
from PIL import Image
import torchvision.transforms as T

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'
DIM = 1024
DEV = 'cuda'
RES = 336
print(f'Full-network FT: BioCLIP-2.5 ViT-H @ {RES}px')

model, _, preprocess = open_clip.create_model_and_transforms(MODEL, force_image_size=RES)
model = model.to(DEV)
# Enable gradient checkpointing to save memory
model.visual.set_grad_checkpointing(True)
print(f'Grad checkpointing enabled. Trainable params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

# Check if the full model fits
x = torch.randn(4, 3, RES, RES).to(DEV)
with torch.autocast('cuda', dtype=torch.bfloat16):
    f = model.encode_image(x)
print(f'Forward pass OK. Feature dim: {f.shape}')

# Quick memory check
mem = torch.cuda.max_memory_allocated() / 1e9
print(f'Peak VRAM for forward: {mem:.1f} GB')
torch.cuda.reset_peak_memory_stats()

# Now backward
loss = f.sum()
loss.backward()
mem = torch.cuda.max_memory_allocated() / 1e9
print(f'Peak VRAM for forward+backward: {mem:.1f} GB')
print('Full-network 336px FT is feasible on this GPU!')
" 2>&1

echo ""
echo "=== Checking memory profile for actual training ==="
python -c "
import torch
import open_clip

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'
RES = 336

# Simulate training setup
model, _, preprocess = open_clip.create_model_and_transforms(MODEL, force_image_size=RES)
model = model.to('cuda')
model.visual.set_grad_checkpointing(True)

# Full network trainable (no frozen params)
for p in model.parameters():
    p.requires_grad_(True)

# Create classifier head
protos = torch.randn(5795, 1024, device='cuda')
Wc = nn.Parameter(protos.clone())
from torch import nn

trainable = list(model.parameters()) + [Wc]
print(f'Trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M')

# Test with batch size 8
bs = 8
x = torch.randn(bs, 3, RES, RES, device='cuda')
y = torch.randint(0, 5795, (bs,), device='cuda')

with torch.autocast('cuda', dtype=torch.bfloat16):
    f = nn.functional.normalize(model.encode_image(x).float(), dim=-1)
    cos = f @ nn.functional.normalize(Wc, dim=1).t()
    loss = nn.functional.cross_entropy(30.0 * cos, y)

loss.backward()
mem = torch.cuda.max_memory_allocated() / 1e9
print(f'BS={bs}: Peak VRAM = {mem:.1f} GB')

torch.cuda.reset_peak_memory_stats()
# Try bs=16
x = torch.randn(16, 3, RES, RES, device='cuda')
y = torch.randint(0, 5795, (16,), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    f = nn.functional.normalize(model.encode_image(x).float(), dim=-1)
    cos = f @ nn.functional.normalize(Wc, dim=1).t()
    loss = nn.functional.cross_entropy(30.0 * cos, y)
loss.backward()
mem = torch.cuda.max_memory_allocated() / 1e9
print(f'BS=16: Peak VRAM = {mem:.1f} GB')

torch.cuda.reset_peak_memory_stats()
# Try bs=32
x = torch.randn(32, 3, RES, RES, device='cuda')
y = torch.randint(0, 5795, (32,), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    f = nn.functional.normalize(model.encode_image(x).float(), dim=-1)
    cos = f @ nn.functional.normalize(Wc, dim=1).t()
    loss = nn.functional.cross_entropy(30.0 * cos, y)
loss.backward()
mem = torch.cuda.max_memory_allocated() / 1e9
print(f'BS=32: Peak VRAM = {mem:.1f} GB')
" 2>&1