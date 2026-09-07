iNaturalist Validation & LoRA Evaluation Kit
==============================================

Files:
- inat_val_manifest.json : 19,409 validation images (17,508 unseen zero-shot / 1,901 seen domain-transfer)
- eval_lora_inat.py      : Evaluation script for baseline and LoRA checkpoints
- build_inat_val_split.py: Split generator script

Usage:
  # 1. Zero-shot baseline on BioCLIP-2
  python eval_lora_inat.py --model bioclip-2 --manifest inat_val_manifest.json

  # 2. Evaluate LoRA checkpoint
  python eval_lora_inat.py --model bioclip-2 --lora_ckpt path/to/lora.pt --manifest inat_val_manifest.json

  # 3. Quick test on 1,000 images
  python eval_lora_inat.py --model bioclip-2 --lora_ckpt path/to/lora.pt --manifest inat_val_manifest.json --max_imgs 1000
