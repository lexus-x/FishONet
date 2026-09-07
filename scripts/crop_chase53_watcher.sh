#!/usr/bin/env bash
# Wait for eval crop-view extract, then holdout stack proxy, then v55 zip. AWS GPU only.
set -euo pipefail
cd /home/ubuntu/onet
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate onet

LOG=outputs/extract_eval_crop_views.log
WLOG=outputs/crop_chase53_watcher.log
TEST_PT=outputs/emb_test_ctft_cropviews.pt
UNS_PT=outputs/emb_unseen_ctft_cropviews.pt
PROXY=outputs/crop_chase53_proxy.json
ZIP=submissions/submission_v55_cropmax_336_f60_tau18.zip

exec >>"$WLOG" 2>&1
echo "=== watcher start $(date -Is) host=$(hostname) ==="

while true; do
  if grep -q '^DONE$' "$LOG" 2>/dev/null && [[ -f "$TEST_PT" && -f "$UNS_PT" ]]; then
    echo "$(date -Is) extract DONE; $(ls -lh "$TEST_PT" "$UNS_PT")"
    break
  fi
  if ! pgrep -f '[p]ython -u research/extract_eval_crop_views.py' >/dev/null; then
    echo "$(date -Is) extract process gone before DONE — last log:"
    tail -30 "$LOG" || true
    exit 1
  fi
  echo "$(date -Is) waiting extract; $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) last=$(tail -1 "$LOG" 2>/dev/null)"
  sleep 90
done

if [[ -f "$ZIP" ]]; then
  echo "$(date -Is) zip exists $ZIP"
else
  echo "$(date -Is) starting build_v55_cropmax_336.py"
  python -u builders/build_v55_cropmax_336.py 2>&1 | tee -a outputs/build_v55_cropmax_336.log
fi

if [[ -f "$PROXY" ]]; then
  echo "$(date -Is) proxy exists $PROXY"
else
  echo "$(date -Is) starting crop_chase53_proxy.py"
  python -u research/crop_chase53_proxy.py 2>&1 | tee -a outputs/crop_chase53_proxy.log
fi

echo "=== watcher end $(date -Is) ==="
python - <<'PY'
import json, os
p='outputs/crop_chase53_proxy.json'
if os.path.exists(p):
    d=json.load(open(p))
    d.pop('rows', None)
    print(json.dumps(d, indent=2))
print('zip', os.path.exists('submissions/submission_v55_cropmax_336_f60_tau18.zip'))
PY
