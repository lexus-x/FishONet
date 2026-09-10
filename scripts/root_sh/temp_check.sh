#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate onet
echo "=== PYTORCH ==="
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "=== DATA ==="
ls ~/onet/data/dl/
echo "=== TRAIN IMAGES ==="
python -c "import json; lab=json.load(open('data/dl/label_train.json')); print(len(lab), 'images,', len(set(lab.values())), 'species')"
echo "=== SPLITS ==="
ls ~/onet/data/dl/splits/
echo "=== IMAGES DIR ==="
ls ~/onet/data/dl/images/ | head -5
echo "=== IMAGES COUNT ==="
find ~/onet/data/dl/images/ -name "*.jpg" | wc -l