#!/usr/bin/env bash
# Stay on fishonet-gpu (this AWS L40S). Wait for iNat shift-bank embed, then proxy.
set -euo pipefail
cd /home/ubuntu/onet
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate onet

LOG=outputs/embed_inat_shift_banks.log
WLOG=outputs/shiftbank_aws_watcher.log
BANK_FT=outputs/inat_photo_bank_ftshift.pt
BANK_336=outputs/inat_photo_bank_fullft336shift.pt
PROXY=outputs/v50_shiftbank_proxy.json

exec >>"$WLOG" 2>&1
echo "=== watcher start $(date -Is) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

while true; do
  if grep -q '^DONE$' "$LOG" 2>/dev/null && [[ -f "$BANK_FT" ]]; then
    echo "$(date -Is) embed DONE; banks: $(ls -lh "$BANK_FT" "$BANK_336" 2>/dev/null || true)"
    break
  fi
  if ! pgrep -f '[p]ython -u research/embed_inat_shift_banks.py' >/dev/null; then
    echo "$(date -Is) embed process gone before DONE — last log:"
    tail -20 "$LOG" || true
    exit 1
  fi
  echo "$(date -Is) waiting embed; $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) last=$(tail -1 "$LOG" 2>/dev/null)"
  sleep 120
done

if [[ -f "$PROXY" ]]; then
  echo "$(date -Is) proxy already exists $PROXY"
else
  echo "$(date -Is) starting v50_shiftbank_proxy.py on this GPU"
  python -u research/v50_shiftbank_proxy.py 2>&1 | tee -a outputs/v50_shiftbank_proxy.log
fi
echo "=== watcher end $(date -Is) ==="
cat "$PROXY" 2>/dev/null || true
