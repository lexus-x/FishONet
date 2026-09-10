#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate onet
cd ~/onet

echo "=== DIFFUSERS CHECK ==="
python -c "from diffusers import DiffusionPipeline; print('diffusers OK')" 2>&1 || echo "diffusers NOT found"

echo "=== TRANSFORMERS ==="
python -c "import transformers; print(transformers.__version__)" 2>&1

echo "=== DESCRIPTIONS ==="
python -c "
import json
desc=json.load(open('data/dl/descriptions.json'))
keys=list(desc.keys())
print(len(desc), 'species with descriptions')
print('Sample:', desc[keys[0]][:200])
" 2>&1

echo "=== DISK SPACE ==="
df -h / | tail -1

echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader