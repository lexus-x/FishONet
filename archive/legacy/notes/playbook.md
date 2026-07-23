# Verified winning playbook (deep-research, 23/25 claims confirmed)

## Backbone
- PRIMARY: BioCLIP 2.5 Huge ViT-H/14 (hf-hub:imageomics/bioclip-2.5-vith14),
  +5.7% zero-shot over BioCLIP 2. Self-reported benchmarks; higher inference cost.
- FALLBACK: BioCLIP 2 ViT-L/14 (hf-hub:imageomics/bioclip-2). Cheaper, well-tested.
- Both pretrained on TreeOfLife-200M incl. FathomNet marine data.
- Optional ensemble member: DINOv2/DINOv3 features (must fine-tune; not zero-shot).

## Open-set recognition (the part that wins/loses it)
- Use Maximum Logit Score (MLS), NOT max-softmax. Magnitude-sensitive familiarity
  scores are best for fine-grained OSR (Vaze ICLR'22; Lang CVPR'24).
- Train the closed-set classifier as hard as possible — strong closed-set => strong OSR.
- Train at the finest taxonomic granularity available.
- Calibrate the known/unknown threshold on a held-out split that SIMULATES unknowns
  (hide some training species, measure unknown accuracy).

## Unknown species via provided text
- Convert scientific -> common English names; 2-5x zero-shot gain (Parashar EMNLP'23).
  F-name rule: use whichever name form is more frequent.
- Feed the provided unknown-species descriptions as labels, NegLabel-style scorer
  (better than plain MCM which ignores given text).

## Squeeze (apply after baseline)
- Multi-crop / flip test-time augmentation (TTA).
- Ensemble fine-tuned-head logits + zero-shot text logits.
- Taxonomic hierarchy fallback: predict genus when species is uncertain.

## Killed claims (do NOT rely on)
- "+8.7% fish-specific FishNet gain for 2.5" -> REFUTED (0-3).
- "frozen DINOv2 works out-of-the-box, no fine-tuning" -> REFUTED (0-3).

## Open gaps to chase next
- Prior FGVC/iNaturalist/GeoLifeCLEF winning-solution specifics (TTA/ensembling
  recipes) were NOT verified — do a targeted follow-up once we have the data.
- OSR results validated on birds/insects, not fish/underwater — verify on our val split.
