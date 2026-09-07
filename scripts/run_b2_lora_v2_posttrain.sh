#!/usr/bin/env bash
set -euo pipefail
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate onet
cd /home/ubuntu/onet
CKPT=outputs/b2_shift_lora_v2.pt
LOG=outputs/b2_lora_v2_posttrain.log
exec > >(tee -a "$LOG") 2>&1
echo "=== posttrain $(date -Is) ==="
for split in train test unseen; do
  python research/extract_b2_lora.py --ckpt "$CKPT" --scale 0.4 --split "$split" \
    --squash_tta 1 --out "outputs/emb_${split}_bioclip2_lora_v2.pt"
done
python research/embed_inat.py --enc b2lora --ckpt "$CKPT" --squash_tta 1 --scale 0.4 \
  --files outputs/inat_image_files.json --out outputs/inat_protos_bioclip2_lora_v2.pt
B2_PROTO=outputs/inat_protos_bioclip2_lora_v2.pt \
B2_QENC=outputs/emb_train_bioclip2_lora_v2.pt \
OUT_TAG=inat_bioclip2_v41_lora_v2_mean_proxy \
  python research/inat_bioclip2_v41_lora_mean_proxy.py
echo "=== done $(date -Is) ==="
