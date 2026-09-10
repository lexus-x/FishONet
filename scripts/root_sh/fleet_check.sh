#!/bin/bash
HOSTS=(a100 blackwell a6000 blackwell2 a6000-left a6000-mid)
echo "=== FLEET CHECK $(date) ==="
for h in "${HOSTS[@]}"; do
    echo -n "$h: "
    ssh -o ConnectTimeout=4 -o BatchMode=yes "$h" 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null' 2>/dev/null || echo "UNREACHABLE"
done
echo "=== DONE ==="