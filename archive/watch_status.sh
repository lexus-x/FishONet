#!/usr/bin/env bash
F="$HOME/test/onet/outputs/job_status.txt"
while true; do
  clear 2>/dev/null || printf '\033[2J\033[H'
  echo "live monitor @ $(date '+%H:%M:%S')   (refresh 5s, Ctrl-C to stop)"
  echo
  if [ -f "$F" ]; then cat "$F"; else echo "waiting for job to start ($F not found yet)..."; fi
  sleep 5
done
