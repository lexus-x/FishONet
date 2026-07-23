#!/usr/bin/env bash
# Just run:  bash run.sh
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate onet 2>/dev/null
python make_family_map.py
