#!/usr/bin/env bash
# Keep L40S busy: adaptive/margin proxy (existing views) → part-view LoRA FT → optional extract.
set -euo pipefail
cd /home/ubuntu/onet
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate onet
WLOG=outputs/gpu_research_supervisor.log
exec >>"$WLOG" 2>&1
echo "=== research supervisor start $(date -Is) ==="

run_logged() {
  local log="$1"; shift
  echo "$(date -Is) RUN $*"
  python -u "$@" 2>&1 | tee -a "$log"
}

if [[ ! -f outputs/v58_adaptive_proxy.json ]]; then
  run_logged outputs/v58_adaptive_proxy.log research/v58_adaptive_proxy.py
else
  echo "$(date -Is) skip v58 proxy"
fi

if ! grep -q '^DONE$' outputs/ctft_partview_ft.log 2>/dev/null; then
  run_logged outputs/ctft_partview_ft.log research/ctft_partview_ft.py
else
  echo "$(date -Is) skip partview FT"
fi

echo "=== research supervisor end $(date -Is) ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
