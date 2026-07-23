#!/bin/bash
# FishOnet: finish the v2b idea (extract it) + train a full top-24 LoRA variant (v2c)
# then extract both train+test embeddings. Runs sequentially on the L40S.
set -x
source ~/miniconda3/etc/profile.d/conda.sh && conda activate onet
cd ~/onet

# 1. extract the interrupted v2b (top-24/r16, ~2.3 epochs) -- free decorrelated member
python src/ft.py --extract train --ckpt outputs/ft_lora_v2b.pt --out outputs/emb_train_v2b.pt --hflip 1 --bs 256
python src/ft.py --extract test  --ckpt outputs/ft_lora_v2b.pt --out outputs/emb_test_v2b.pt  --hflip 1 --bs 256

# 2. full 6-epoch top-24 variant (v2c) -- lever #1: new decorrelated LoRA fit
python src/ft.py --epochs 6 --top_k_blocks 24 --rank 16 --alpha 32 --bs 128 \
  --out outputs/ft_lora_v2c.pt 2>&1 | tail -400

# 3. extract v2c
python src/ft.py --extract train --ckpt outputs/ft_lora_v2c.pt --out outputs/emb_train_v2c.pt --hflip 1 --bs 256
python src/ft.py --extract test  --ckpt outputs/ft_lora_v2c.pt --out outputs/emb_test_v2c.pt  --hflip 1 --bs 256

echo ALL_DONE
