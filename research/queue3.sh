#!/bin/bash
# Queue 3: after queue2 (ctftclean) frees the GPU ->
#   1) SigLIP2-384 train+test embeddings (seen-member diversity test)
#   2) v2d LoRA variant (top-20, rank-32, dropout 0.08) + extract
source ~/miniconda3/etc/profile.d/conda.sh && conda activate onet
cd ~/onet
echo "queue3 waiting for queue2..."
while [ ! -f outputs/emb_train_ctftclean.pt ]; do sleep 60; done
sleep 30
echo "queue3 starting at $(date)"

# 1. SigLIP2-384 (general web-scale, CLIP-style) as seen-ensemble member candidate
python src/embed_any.py --model 'hf-hub:timm/ViT-SO400M-16-SigLIP2-384' \
  --split train --out outputs/emb_train_siglip2.pt --hflip 1 --batch 128
python src/embed_any.py --model 'hf-hub:timm/ViT-SO400M-16-SigLIP2-384' \
  --split test --out outputs/emb_test_siglip2.pt --hflip 1 --batch 128

# 2. v2d LoRA variant (decorrelated: top-20 / rank-32 / alpha-64 / dropout 0.08)
python src/ft.py --epochs 6 --top_k_blocks 20 --rank 32 --alpha 64 --bs 128 \
  --out outputs/ft_lora_v2d.pt 2>&1 | tail -300
python src/ft.py --extract train --ckpt outputs/ft_lora_v2d.pt --out outputs/emb_train_v2d.pt --hflip 1 --bs 256
python src/ft.py --extract test  --ckpt outputs/ft_lora_v2d.pt --out outputs/emb_test_v2d.pt  --hflip 1 --bs 256

echo QUEUE3_DONE
