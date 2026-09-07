#!/bin/bash
set -euo pipefail
cd /home/ubuntu/onet
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate onet

LOG=outputs/v38_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "=== v38 pipeline $(date) ==="

# wait for coverage download
while pgrep -f 'download_coverage_images.py' >/dev/null 2>&1; do
  tail -1 outputs/coverage_download.log 2>/dev/null || true
  sleep 30
done
echo "download done"; tail -3 outputs/coverage_download.log

# wait for GPU (taxabind embed)
while pgrep -f 'extract_taxabind_train.py' >/dev/null 2>&1; do
  echo "waiting GPU taxabind $(tail -1 outputs/tb_train_embed.log 2>/dev/null)"
  sleep 60
done
echo "GPU free"

python research/embed_inat.py --enc ctft --ckpt outputs/ctft_shift.pt --squash_tta 1 \
  --files outputs/coverage_image_files.json --out outputs/coverage_protos_ctftshift.pt \
  2>&1 | tee outputs/coverage_embed.log

python research/merge_coverage_protos.py

python research/inat_proto_proxy.py --protos outputs/protos_ctftshift_merged.pt --qenc ctftshift \
  2>&1 | tee outputs/coverage_proto_proxy.log

python research/coverage_audit.py outputs/protos_ctftshift_merged.pt

# build w2 and w3 if proxy improved vs baseline in log
for W in 2.0 3.0; do
  IMG_W=$W PROTO_PATH=outputs/protos_ctftshift_merged.pt python builders/build_v38_coverage.py
done
echo "=== pipeline complete $(date) ==="
