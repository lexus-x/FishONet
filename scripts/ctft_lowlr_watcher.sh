#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/onet
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate onet

LOG=outputs/ctft_shift_lowlr_train.log
CKPT=outputs/ctft_shift_lowlr.pt
PID="$1"
WLOG=outputs/ctft_shift_lowlr_watcher.log
DB_KILL=25.06

exec >>"$WLOG" 2>&1
echo "=== lowlr watcher start $(date -Is) pid=$PID db_kill=$DB_KILL ==="

best_db_from_log() {
  python3 - <<PY
import re, pathlib
p = pathlib.Path("$LOG")
t = p.read_text() if p.exists() else ""
best = 24.76
for m in re.finditer(r"\* saved best db=([\d.]+)", t):
    best = max(best, float(m.group(1)))
fm = re.search(r"FINAL raw,db = \([^)]+\)\s+best_db=([\d.]+)", t)
if fm:
    best = max(best, float(fm.group(1)))
print(f"{best:.4f}")
PY
}

while ! grep -q "FINAL raw,db" "$LOG" 2>/dev/null; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "pid $PID exited without FINAL"; break
  fi
  echo "$(date -Is) waiting..."
  sleep 60
done

best=$(best_db_from_log)
echo "best_db=$best"
if python3 - <<PY
import sys
sys.exit(0 if float("$best") + 1e-9 >= $DB_KILL else 1)
PY
then
  echo "PASS db kill — run inat_ft_stack_proxy manually if embeddings added later"
else
  printf 'NO_ZIP\nreason=lowlr_probe best_db=%s < %s\n' "$best" "$DB_KILL" > outputs/ctft_lowlr_zip_decision.txt
  echo "NO_ZIP lowlr probe below kill bar"
fi
echo "=== lowlr watcher done $(date -Is) ==="
