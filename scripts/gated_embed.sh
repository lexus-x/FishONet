#!/usr/bin/env bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate onet
cd /home/islab/test/onet
export HF_HUB_DISABLE_PROGRESS_BARS=1
echo "gate start $(date)"
free=0
while true; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  ts=$(date '+%F %T')
  if [ "${apps:-1}" -eq 0 ]; then
    free=$((free+1))
    echo "$ts FREE-reading $free/2 (mem=${mem}MiB util=${util}%)"
  else
    free=0
    echo "$ts busy apps=$apps mem=${mem}MiB util=${util}%"
  fi
  if [ "$free" -ge 2 ]; then
    echo "=== GPU FREE confirmed at $ts — starting embed ==="
    break
  fi
  sleep 60
done
echo "=== embed TEST ==="   ; python src/embed.py --split test   --batch 256
echo "=== embed UNSEEN ===" ; python src/embed.py --split unseen --batch 256
echo "=== FIRST PREDICTION (zero-shot name) ===" ; python src/baseline.py --alpha 0 --text name --out outputs/prediction_zeroshot_name.json
echo "=== embed TRAIN ===" ; python src/embed.py --split train  --batch 256
echo "GATED_EMBED_DONE $(date)"
