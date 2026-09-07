#!/usr/bin/env bash
# One-time environment setup. Idempotent-ish.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
if ! conda env list | grep -q '^onet '; then
  conda create -y -n onet python=3.10
fi
conda activate onet
# CUDA wheels for WSL2 (cu121 works with driver 596.x). Adjust if needed.
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install open_clip_torch>=2.24 timm pillow pandas numpy scikit-learn tqdm pyyaml matplotlib
echo "[setup] done. Activate with: conda activate onet"
