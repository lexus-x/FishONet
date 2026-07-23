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
| **v21 unified_ctftbig_ext holdout** (single pipeline, w_text=0.5 gamma=30) | seen=85.10% unseen=20.10% **overall=56.76%** |
| **v21 unified_ctftbig_ext REAL** (Codabench, 2026-07-21) | seen=71.60% unseen=11.10% **overall=45.19%** |

**Key problem:** holdout badly overestimates the real single-pipeline score, worse than the old
v20 gap. v21 real-vs-holdout: seen -13.5pt, unseen -9.0pt, overall -11.6pt. Two separable causes
(see `notes/playbook.md` for the full diagnostic math):
1. **Novelty-gate miscalibration**: real seen-image mis-route rate to the unseen head is ~19.4%,
   vs ~4.9% implied by the holdout simulation -- **3.9x worse**. The gate is far more trigger-happy
   on real photos than on the pseudo-unseen (rarest-20%-of-seen-classes) holdout proxy.
2. **Zero-shot ceiling gap**: conditional accuracy on images correctly routed to the unseen head is
   ~13.9% real vs ~28.3% holdout oracle -- **~0.5x**. True unseen species (zero training images) are
   harder to zero-shot-match via text than the pseudo-unseen holdout (which still has a few images)
   suggests.
Next lever to try: lower gamma (less aggressive novelty penalty) -- given seen is 56% of the real
population and its real mis-route cost is worse than modeled, the gate is probably over-tuned
toward unseen recall relative to what actually pays off in the real world.

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

## Layout

```
onet/
├── src/                    # current pipeline scripts (27) -- embed/FT/predict
│   ├── ft.py              # LoRA fine-tuning (seen head)
│   ├── ft_cap.py          # Caption encoder FT
│   ├── ft_robust.py       # Shift-robust FT (heavy aug)
│   ├── contrastive_ft.py  # CTFT for unseen text matching
│   ├── embed.py           # Frozen embedding extraction
│   ├── build_text_embeddings.py
│   ├── predict_v20_multienc.py     # v20 predictor
│   ├── unified_holdout*.py         # v21 single-pipeline holdout eval harnesses (current)
│   ├── validate.py, validate_gate.py
│   └── archive/            # superseded scripts (v15/v18/v19 eras, cyc_* research branch,
│                            #   one-off sweeps/probes) -- kept for history, not maintained
├── builders/                # scripts that build a final submission zip (build_v21_unified_*.py)
├── experiments/              # diagnostics/verification scripts still in active use
├── archive/                  # superseded root-level one-offs + stale-path shell scripts
├── configs/               # YAML configs (WIP)
├── data/dl/                # dataset (gitignored)
│   ├── images/ (99,979 .jpg)
│   ├── label_train.json, descriptions.json
│   └── splits/ (train.pkl, test.pkl, unseen.pkl)
├── outputs/                # embeddings, predictions, submissions (flat, path-referenced by name)
├── submissions/             # final submission files
├── notes/playbook.md       # research strategy
├── Leah's work/            # family/species taxonomy mapping
├── scripts/                # setup + run helpers (setup_env.sh, gated_embed.sh, v2c_pipeline.sh)
├── MEMORY.md               # operating rules
└── environment.yml         # conda environment