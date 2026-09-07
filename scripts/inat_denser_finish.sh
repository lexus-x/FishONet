#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/onet
source ~/miniconda3/etc/profile.d/conda.sh
conda activate onet
python research/download_inat_only.py | tee -a outputs/inat_download_finish.log
python research/embed_inat.py --enc ctft --ckpt outputs/ctft_shift.pt --squash_tta 1 \
  --files outputs/inat_image_files.json --out outputs/inat_protos_ctftshift_full.pt \
  --bs 64 --workers 4 | tee outputs/inat_embed_finish.log
python research/inat_proto_proxy.py --protos outputs/inat_protos_ctftshift_full.pt --qenc ctftshift \
  | tee outputs/inat_proto_proxy_finish.log
W3=$(grep -oP 'S0 \+ 3\*img: \K[0-9.]+' outputs/inat_proto_proxy_finish.log | head -1)
echo "w3_proxy=$W3 ref=43.53"
python3 -c "import sys; w=float('${W3:-0}'); sys.exit(0 if w>=43.83 else 1)" && \
  PROTO_PATH=outputs/inat_protos_ctftshift_full.pt IMG_W=3 python builders/build_v37_inat_proto.py
