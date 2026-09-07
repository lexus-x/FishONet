#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/onet
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate onet

LOG=outputs/ctft_shift_long_train.log
CKPT=outputs/ctft_shift_long.pt
PID="${1:-}"
WLOG=outputs/ctft_shift_long_watcher.log
BASELINE_DB=24.76
DB_KILL=25.06

exec >>"$WLOG" 2>&1
echo "=== watcher start $(date -Is) pid=${PID:-none} db_kill=$DB_KILL ==="

parse_metrics() {
  python3 - <<PY
import re, pathlib
p = pathlib.Path("outputs/ctft_shift_long_train.log")
baseline = ${BASELINE_DB}
if not p.exists():
    print(f"0 0 {baseline:.4f} 0"); raise SystemExit
t = p.read_text()
best = baseline
last_imp = 0
last_step = 0
for m in re.finditer(r"\* saved best db=([\d.]+)", t):
    db = float(m.group(1))
    if db > best + 1e-9:
        best = db
        last_imp = last_step
for m in re.finditer(r"VAL @ (\d+): raw,db = \([^,]+, ([\d.]+)\)", t):
    last_step = max(last_step, int(m.group(1)))
fm = re.search(r"FINAL raw,db = \([^)]+\)\s+best_db=([\d.]+)", t)
if fm:
    best = max(best, float(fm.group(1)))
done = "FINAL raw,db" in t
print(f"{int(done)} {last_step} {best:.4f} {last_imp}")
PY
}

db_clears_kill() {
  local best_db="$1"
  python3 - <<PY
best=float("$best_db")
kill=float("$DB_KILL")
import sys
sys.exit(0 if best + 1e-9 >= kill else 1)
PY
}

write_no_zip() {
  local reason="$1"
  {
    echo "NO_ZIP"
    echo "reason=$reason"
    echo "baseline_db=$BASELINE_DB"
    echo "db_kill=$DB_KILL"
    echo "ts=$(date -Is)"
  } > outputs/ctft_long_zip_decision.txt
  echo "$reason" | tee -a "$WLOG"
}

wait_training() {
  local restarted=0
  while true; do
    if [[ -f "$LOG" ]] && grep -q "FINAL raw,db" "$LOG"; then
      echo "training complete (FINAL in log)"; return 0
    fi
    if [[ -n "$PID" ]] && ! kill -0 "$PID" 2>/dev/null; then
      echo "process $PID gone $(date -Is)"
      if grep -q "FINAL raw,db" "$LOG" 2>/dev/null; then return 0; fi
      if [[ $restarted -lt 1 ]]; then
        echo "restarting training once..."
        restarted=1
        nohup python src/contrastive_ft.py --shift_aug 1 --init_ckpt outputs/ctft_shift.pt \
          --max_steps 2400 --lr 3e-4 --val_every 200 --save outputs/ctft_shift_long.pt \
          >>"$LOG" 2>&1 &
        PID=$!
        echo "new PID=$PID"
        sleep 30
        continue
      fi
      echo "training failed after restart"; return 1
    fi
    read -r done last_step best_db last_imp < <(parse_metrics)
    echo "$(date -Is) step=$last_step best_db=$best_db last_imp@=$last_imp running"
    if [[ "$done" == "1" ]]; then return 0; fi
    sleep 150
  done
}

run_extract() {
  local split out
  for split in train test unseen; do
    out="outputs/emb_${split}_ctftshift_long.pt"
    echo "extract $split -> $out"
    python research/extract_ctft_any.py --ckpt "$CKPT" --scale 0.4 --split "$split" \
      --out "$out" --squash_tta 1 --workers 4
  done
}

run_inat_assets() {
  echo "build iNat protos + photo bank"
  python research/embed_inat.py --enc ctft --ckpt "$CKPT" --squash_tta 1 \
    --out outputs/inat_protos_ctftshift_long_full.pt
  python research/inat_maxpool_proxy.py --build-bank \
    --ckpt "$CKPT" --squash_tta 1 \
    --bank outputs/inat_photo_bank_ctftshift_long.pt
}

run_proxy() {
  python3 - <<'PY'
import json, os, sys
import torch
sys.path.insert(0, 'research')
from common import FishData, OUT
from inat_ft_stack_proxy import eval_encoder

D = FishData(); cand = D.cand
bank_base = torch.load(os.path.join(OUT, 'inat_photo_bank_ctftshift.pt'), weights_only=False)['bank']
bank_long = torch.load(os.path.join(OUT, 'inat_photo_bank_ctftshift_long.pt'), weights_only=False)['bank']
ref = eval_encoder(D, cand, 'ctftshift', os.path.join(OUT, 'inat_protos_ctftshift_full.pt'), bank_base)
long = eval_encoder(D, cand, 'ctftshift_long', os.path.join(OUT, 'inat_protos_ctftshift_long_full.pt'), bank_long)
res = {
    'ckpt': 'outputs/ctft_shift_long.pt',
    'ref_ctftshift': {k: ref[k] for k in ref if k not in ('has','S0','gold','q','Smean','Smp4')},
    'long_ft': {k: long[k] for k in long if k not in ('has','S0','gold','q','Smean','Smp4')},
    'delta_mean_w3': long['mean_w3'] - ref['mean_w3'],
    'delta_maxpool_w35': long['maxpool_top4_w35'] - ref['maxpool_top4_w35'],
    'ref_targets': {'mean_w3': 43.49, 'maxpool_w35': 45.17},
    'projected_real_overall_012': {
        'from_mean_delta': 50.49 + 0.12 * (long['mean_w3'] - ref['mean_w3']),
        'from_mp_delta': 50.49 + 0.12 * (long['maxpool_top4_w35'] - ref['maxpool_top4_w35']),
    },
}
out = os.path.join(OUT, 'ctftshift_long_stack_proxy_results.json')
json.dump(res, open(out, 'w'), indent=1)
ev = max(res['projected_real_overall_012']['from_mean_delta'], res['projected_real_overall_012']['from_mp_delta'])
print(f'ref mean@3={ref["mean_w3"]:.2f} mp@3.5={ref["maxpool_top4_w35"]:.2f}')
print(f'long mean@3={long["mean_w3"]:.2f} ({res["delta_mean_w3"]:+.2f}) mp@3.5={long["maxpool_top4_w35"]:.2f} ({res["delta_maxpool_w35"]:+.2f})')
print(f'projected EV {ev:.2f}%')
open('outputs/ctft_long_zip_decision.txt','w').write(f'ZIP_CANDIDATE\nprojected_ev={ev:.4f}\n')
json.dump(res, open(out, 'w'), indent=1)
PY
}

maybe_zip() {
  local ev
  ev=$(grep projected_ev outputs/ctft_long_zip_decision.txt | cut -d= -f2)
  if python3 - <<PY
ev=float("$ev")
import sys
sys.exit(0 if ev > 50.49 else 1)
PY
  then
    echo "ZIP eligible ev=$ev — building v40 long variant"
    CTFT_LONG=1 python builders/build_v40_ctftshift_long.py
  else
    echo "NO_ZIP projected_ev=$ev <= 50.49" | tee -a outputs/ctft_long_zip_decision.txt
  fi
}

wait_training || exit 1
read -r _done _step best_db _imp < <(parse_metrics)
if ! db_clears_kill "$best_db"; then
  write_no_zip "best_ckpt_db=${best_db} < db_kill=${DB_KILL} (baseline ${BASELINE_DB}+0.3); skip extract/proxy/zip"
  echo "=== watcher done (NO_ZIP) $(date -Is) ==="
  exit 0
fi
[[ -f "$CKPT" ]] || { echo "missing $CKPT"; exit 1; }
run_extract
run_inat_assets
run_proxy
maybe_zip
echo "=== watcher done $(date -Is) ==="
