#!/usr/bin/env bash
# Keep the L40S busy: 336 eval crops → iNat 336 crop-bank → holdout proxy → v57 zip.
set -euo pipefail
cd /home/ubuntu/onet
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate onet
WLOG=outputs/gpu_chase53_supervisor.log
exec >>"$WLOG" 2>&1
echo "=== supervisor start $(date -Is) gpu=$(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader) ==="

run_logged() {
  local log="$1"; shift
  echo "$(date -Is) RUN $*"
  python -u "$@" 2>&1 | tee -a "$log"
}

# 1. 336 query crop views (eval + holdout)
if ! grep -q '^DONE$' outputs/extract_eval_336_crops.log 2>/dev/null; then
  run_logged outputs/extract_eval_336_crops.log research/extract_eval_336_crops.py
else
  echo "$(date -Is) skip extract_eval_336_crops (DONE)"
fi

# 2. 336 iNat gallery with the same crops
if ! grep -q '^DONE$' outputs/embed_inat_336_cropbank.log 2>/dev/null; then
  run_logged outputs/embed_inat_336_cropbank.log research/embed_inat_336_cropbank.py
else
  echo "$(date -Is) skip embed_inat_336_cropbank (DONE)"
fi

# 3. holdout proxy (written if missing; skip if already scored)
if [[ -f outputs/v57_336crop_proxy.json ]]; then
  echo "$(date -Is) skip proxy, exists"
elif [[ -f research/v57_336crop_proxy.py ]]; then
  run_logged outputs/v57_336crop_proxy.log research/v57_336crop_proxy.py
else
  echo "$(date -Is) proxy script not ready yet"
fi

# 4. zip if builder exists and crops exist
if [[ -f submissions/submission_v57_336crop_f60_tau18.zip ]]; then
  echo "$(date -Is) skip zip, exists"
elif [[ -f builders/build_v57_336crop.py && -f outputs/emb_test_fullft336_cropviews.pt ]]; then
  run_logged outputs/build_v57_336crop.log builders/build_v57_336crop.py
else
  echo "$(date -Is) builder not ready or crops missing"
fi

echo "=== supervisor end $(date -Is) ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
