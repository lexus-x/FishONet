#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate onet
cd /home/islab/test/onet
echo "=== SCHEDULED EXTRACT started $(date) ===" > outputs/extract_ftrobust.log
python src/ft.py --extract train --ckpt outputs/ft_robust_lora.pt --out outputs/emb_train_ftrobust.pt --hflip 1 --bs 128 --workers 8 >> outputs/extract_ftrobust.log 2>&1
echo "=== train done $(date), starting test ===" >> outputs/extract_ftrobust.log
python src/ft.py --extract test --ckpt outputs/ft_robust_lora.pt --out outputs/emb_test_ftrobust.pt --hflip 1 --bs 128 --workers 8 >> outputs/extract_ftrobust.log 2>&1
echo "=== ALL DONE $(date) ===" >> outputs/extract_ftrobust.log

