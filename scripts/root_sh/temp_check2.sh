#!/bin/bash
cd ~/onet
source ~/miniconda3/etc/profile.d/conda.sh
conda activate onet

echo "=== TRAIN IMAGES ==="
python -c "
import json
lab=json.load(open('data/dl/label_train.json'))
print('{} images, {} species'.format(len(lab), len(set(lab.values()))))
"

echo "=== SRC FILES ==="
ls src/

echo "=== OUTPUTS (EMBS) ==="
ls -lh outputs/emb_*

echo "=== CHECKPOINTS ==="
ls -lh outputs/*.pt 2>/dev/null | grep -E '(ft_lora|ctft|cap|state)'

echo "=== GPU VRAM (FREE) ==="
nvidia-smi --query-gpu=memory.free --format=csv,noheader

echo "=== DONE ==="