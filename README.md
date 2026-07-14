# onet — CV4Ecology 2026 Fish Species Open-Set Recognition

Open-set fish species classification. The task is to classify fish images into
17,393 species — a mix of **seen** (known from training, 5,795 classes) and
**unseen** (novel, described only by text, 11,598 classes). Scored on overall
accuracy with seen/unseen breakdown. Submit via Codabench.

## Pipeline

```
data/dl/images/ (99,979 .jpg)
  │
  ├─► src/embed.py          → outputs/emb_train.pt, emb_test.pt       (ViT-L frozen)
  ├─► src/build_text_embeddings.py → outputs/text_emb.pt, text_emb_h*.pt
  ├─► src/ft.py             → outputs/ft_*.pt (LoRA)                  (ViT-H fine-tune)
  ├─► src/ft_cap.py         → outputs/ft_cap*.pt (Caption encoder FT)
  ├─► src/contrastive_ft.py → outputs/ctft_*.pt (contrastive FT)
  │
  └─► src/predict_v20_multienc.py → outputs/prediction_v20_*.json
        ├── SEEN route:  z-scored blend of 3 encoders (ft + cap + L)
        │                  + taxon text discriminative term (LAM=4.0)
        └── UNSEEN route: ctft_big + frozen-L (v15 recipe, text matching)
```

## Accuracy

| Metric | Value |
|---|---|
| v20 holdout (seen only) | 87.94% |
| v20 real test seen (probe) | 79.70% |
| Frozen baseline (ViT-L NCM) | ~75% |
| Full-network FT 336px | 82.77% (training in progress) |

**Key problem:** ~8% train→test distribution shift gap (87.94% holdout vs 79.70% real).

## Key Techniques

- **Backbone:** BioCLIP 2.5 Huge (ViT-H/14), BioCLIP 2 (ViT-L/14) frozen
- **Fine-tuning:** LoRA (rank 16, top-12 blocks), ArcFace margin (m=0.2), balanced-softmax, genus-auxiliary loss
- **Known head:** Multi-encoder z-scored blend (ft + caption + frozen-L) + taxonomic text discriminative term
- **Open-set gate:** Structural split (test→seen, unseen→unseen); MLS gate in progress
- **Unknown head:** Zero-shot text matching via contrastive FT on descriptions
- **TTA:** Horizontal flip + multi-crop (experimental)

## Quickstart

```bash
bash scripts/setup_env.sh          # create conda env 'onet'
conda activate onet
python src/sanity.py               # verify torch+CUDA+BioCLIP
```

## Full Pipeline

```bash
# 1. Extract frozen embeddings (ViT-L)
python src/embed.py

# 2. Build text embeddings
python src/build_text_embeddings.py

# 3. Fine-tune seen head (ViT-H LoRA) — GPU required, ~2-3 hrs
python src/ft.py

# 4. Fine-tune caption encoder
python src/ft_cap.py

# 5. Contrastive FT for unseen text matching
python src/contrastive_ft.py

# 6. Extract embeddings for FT models
python src/extract_embeddings.py   # or individual extract_*.py scripts

# 7. Predict
python src/predict_v20_multienc.py --real
# Output: outputs/prediction_v20_real.json, submission_v20_*.zip
```

Or use the bundled runner:
```bash
bash scripts/run_pipeline.sh
```

## Layout

```
onet/
├── src/                    # pipeline scripts (60+)
│   ├── ft.py              # LoRA fine-tuning (seen head)
│   ├── ft_cap.py          # Caption encoder FT
│   ├── ft_robust.py       # Shift-robust FT (heavy aug)
│   ├── contrastive_ft.py  # CTFT for unseen text matching
│   ├── embed.py           # Frozen embedding extraction
│   ├── build_text_embeddings.py
│   ├── predict_v20_multienc.py  # Current best predictor
│   └── validate.py        # Local validation harness
├── configs/               # YAML configs (WIP)
├── data/dl/               # dataset (gitignored)
│   ├── images/ (99,979 .jpg)
│   ├── label_train.json, descriptions.json
│   └── splits/ (train.pkl, test.pkl, unseen.pkl)
├── outputs/               # embeddings, predictions, submissions
├── submissions/           # final submission files
├── notes/playbook.md      # research strategy
├── Leah's work/           # family/species taxonomy mapping
├── scripts/               # setup + run helpers
├── MEMORY.md              # operating rules
└── environment.yml        # conda environment