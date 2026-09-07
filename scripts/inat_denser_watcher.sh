#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/onet
while pgrep -f '[p]ython research/inat_opendata.py --stage photos --photo-cap 30' >/dev/null; do
  sleep 120
done
while pgrep -f '[p]ython research/fetch_inat_denser.py' >/dev/null; do
  sleep 120
done
bash scripts/inat_denser_finish.sh >> outputs/inat_denser_finish.log 2>&1
