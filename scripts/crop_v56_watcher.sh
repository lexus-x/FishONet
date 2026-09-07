#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/onet
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate onet
LOG=outputs/extract_eval_overlap.log
WLOG=outputs/crop_v56_watcher.log
ZIP=submissions/submission_v56_overlap_336_f60_tau18.zip
exec >>"$WLOG" 2>&1
echo "=== v56 watcher start $(date -Is) ==="
while true; do
  if grep -q '^DONE$' "$LOG" 2>/dev/null && [[ -f outputs/emb_test_ctft_ostrip.pt && -f outputs/emb_unseen_ctft_ostrip.pt ]]; then
    echo "$(date -Is) overlap extract DONE"
    break
  fi
  if ! pgrep -f 'research/extract_eval_overlap.py' >/dev/null; then
    echo "$(date -Is) overlap extract gone before DONE"
    tail -30 "$LOG" || true
    exit 1
  fi
  echo "$(date -Is) waiting overlap; last=$(tail -1 "$LOG" 2>/dev/null)"
  sleep 90
done
if [[ -f "$ZIP" ]]; then
  echo "$(date -Is) zip exists"
else
  python -u builders/build_v56_overlap_336.py 2>&1 | tee -a outputs/build_v56_overlap_336.log
fi
echo "=== v56 watcher end $(date -Is) ==="
