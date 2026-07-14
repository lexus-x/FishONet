# onet — CV4Ecology 2026 Fish Species Recognition Challenge

Open-set fish species classification. Scored on overall accuracy, with KNOWN
(seen in training) and UNKNOWN (novel) species reported separately. Text
descriptions are provided for ALL species (known + unknown). Submit via Codabench.

## Strategy (from verified deep-research playbook — see notes/playbook.md)
1. Backbone: BioCLIP 2.5 Huge (ViT-H/14) — hf-hub:imageomics/bioclip-2.5-vith14.
   Fallback / faster: BioCLIP 2 (ViT-L/14) — hf-hub:imageomics/bioclip-2.
2. KNOWN head: fine-tune a strong closed-set classifier at finest taxonomic
   granularity. Train it as hard as possible (this also powers the open-set gate).
3. OPEN-SET gate: Maximum Logit Score (MLS), NOT max-softmax. Calibrate the
   known-vs-unknown threshold on a held-out split that SIMULATES unknowns.
4. UNKNOWN head: zero-shot text matching using the provided descriptions.
   Convert scientific -> common English names (F-name rule). NegLabel-style scoring.
5. Hybrid routing: gate decides known vs unknown, route to the right head.
6. Squeeze: multi-crop TTA, ensemble fine-tuned + zero-shot logits, taxonomic
   hierarchy fallback (genus when unsure).

## Layout
- src/            pipeline code
- configs/        yaml configs
- data/           dataset (gitignored)
- checkpoints/    model weights (gitignored)
- outputs/        embeddings, logits, submissions
- notes/          research playbook + running log
- scripts/        setup + run helpers

## Quickstart
    bash scripts/setup_env.sh         # one-time: conda env 'onet' + deps
    conda activate onet
    python src/sanity.py              # verify torch+CUDA+BioCLIP load
