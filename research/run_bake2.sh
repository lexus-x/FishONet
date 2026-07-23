#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate onet
cd ~/onet
# wait until v2b test extraction exists (pipe writes it after train extraction)
while [ ! -f outputs/emb_test_v2b.pt ]; do sleep 30; done
python research/ext_ensemble2.py
