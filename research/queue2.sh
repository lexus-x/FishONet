#!/bin/bash
# Queue 2: after the v2c pipeline frees the GPU -> leakage-free ctft retrain + extract
source ~/miniconda3/etc/profile.d/conda.sh && conda activate onet
cd ~/onet
echo "queue2 waiting for v2c pipeline..."
while [ ! -f outputs/emb_test_v2c.pt ]; do sleep 60; done
sleep 30
echo "queue2 starting ctft_cleanse at $(date)"

python - <<'EOF'
s = open('research/ctft_cleanse.py').read()
old = "train_items=[(fn,k2i[lab[fn]]) for c in known for fn in by[c]]"
new = ("train_items=[(fn,k2i[lab[fn]]) for c in known for fn in "
       "(sorted(by[c])[:-max(1,round(0.2*len(by[c])))] if len(by[c])>=3 else sorted(by[c]))]")
assert old in s, 'train_items line not found'
s = s.replace(old, new)
s = s.replace("default='outputs/ctft_lora.pt'", "default='outputs/ctft_cleanse.pt'")
s = s.replace("type=int,default=12", "type=int,default=4")
open('research/ctft_cleanse.py', 'w').write(s)
print('patched ctft_cleanse.py (leakage-free train split)')
EOF

python research/ctft_cleanse.py --top_k 16 --rank 48 --alpha 96 --max_steps 800 --save outputs/ctft_cleanse.pt
python research/extract_ctft_any.py --ckpt outputs/ctft_cleanse.pt --scale 0.4 --split train --out outputs/emb_train_ctftclean.pt
echo QUEUE2_DONE
