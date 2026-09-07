# FishONet

Open-set fish species recognition for the CV4Ecology 2026 challenge (Codabench 16815).
17,393 species, 35,665 evaluation images, and 11,598 of the species have no training
photograph at all.

**Official leaderboard result: 53.736% overall** (77.61% on photographed species, 22.91% on
species with none), submission `v109`. The technical report is
[reports/FishONet_Technical_Report_paper.pdf](reports/FishONet_Technical_Report_paper.pdf).
An interactive walkthrough of the system is at https://sai-fishonet.pages.dev/.

Lalith Sai and Cheng Yaw Low, Intelligent Systems Laboratory, Changwon National University.

## What it does

Every evaluation image is scored by five encoders (three BioCLIP-2.5 ViT-H/14 fine-tunes, one
BioCLIP-2 ViT-L/14 with a LoRA adapter, and TaxaBind ViT-B/16), then routed by a learned
12-feature gate to one of two heads. The seen head scores the 5,795 photographed species from
class prototypes, nearest training exemplars, and taxonomic text. The novel head scores the 11,598
unphotographed species from ten fused legs: text descriptions, an iNaturalist photo bank, and
TreeOfLife-200M prototypes. Each head's shortlist is re-ranked by a small logistic model trained
on a leak-free candidate pool. The final label is a single argmax over all 17,393 classes.

The gate never sees the seen/unseen split. It ranks all 35,665 images by a familiarity score
and routes the top 60% to the seen head. `research/verify_compliance.py` checks this and every
other rules claim mechanically; see "Verifying compliance" below.

```
 eval image ──► 5 encoders ──► 12-feature quota gate (f = 0.60)
                                        │
                  ┌─────────────────────┴─────────────────────┐
                  ▼                                           ▼
          seen head, 5,795 classes                  novel head, 11,598 classes
          prototypes + exemplars + taxon text       6 text legs + 2 iNat banks + 2 ToL protos
          32-feature re-rank, K = 10                34-feature re-rank, K = 20
          genus-level backoff
                  └─────────────────────┬─────────────────────┘
                                        ▼
                          one argmax over all 17,393 classes
```

## Results

Every number here is a real leaderboard score. The seen and novel columns are the diagnostics
Codabench reports back; the split weights are 0.5635 and 0.4365.

| build | overall | seen | novel | what changed |
|---|---:|---:|---:|---|
| v22 | 45.39 | – | – | closed-set ensemble, shift-robust augmentation |
| v33 | 47.78 | 78.20 | 8.51 | single-pipeline routing, Sinkhorn on the novel route |
| v36 | 49.04 | 78.95 | 10.42 | shift-augmented LoRA, TaxaBind leg |
| v46 | 50.84 | 78.95 | 14.55 | iNaturalist ∩ TreeOfLife merged prototypes |
| v56 | 51.62 | 75.67 | 20.57 | 7-crop max-pool over a 336 px photo bank |
| v77 | 53.38 | 77.72 | 21.96 | learned 12-feature quota gate |
| v79 | 53.42 | 77.45 | 22.41 | gate refit at C = 100 |
| v81 | 53.43 | 77.45 | 22.42 | shortlist re-ranker (trained on a leaking pool, see report §5.1) |
| v82 | 53.64 | 77.45 | 22.91 | leak-free novel re-ranker |
| v83 | 53.70 | 77.54 | 22.91 | leak-free seen re-ranker |
| **v109** | **53.74** | **77.61** | **22.91** | genus-level backoff on the seen route |

The full ledger, including every build that was tried and dropped, is `HANDOFF.md`. The report's
Figure 1 cites it by line number.

## Reproducing the submission

One NVIDIA L40S (46 GB) is enough. Fine-tuned encoder weights and the embedding caches are not in
the repository; the scripts under `src/` produce them from the provided training images.

```bash
bash scripts/setup_env.sh && conda activate onet
python src/sanity.py                                   # torch, CUDA, BioCLIP load

# training-side artifacts consumed by the builders
GATE_C=100 GATE_TAG=v79 python research/learned_gate_v77.py    # outputs/learned_gate_v79.pkl
python research/rerank_leakfree_v82.py                          # outputs/rerank_unseen_v82_leakfree.pkl
python research/rerank_seen_v83.py                              # outputs/rerank_seen_v83.pkl
python research/genus_rerank_v103_train.py                      # outputs/rerank_seen_genus_v103.pkl

# the chain; each builder reads the previous one's output
python builders/build_v56_overlap_336.py
python builders/build_v77_learned_gate.py
python builders/build_v81_rerank.py
python builders/build_v82_rerank_leakfree.py
python builders/build_v83_rerank_both.py
python builders/build_v109_genus_gamble.py     # submissions/submission_v109_genus_gamble.zip
```

The gate artifact rebuilds bit-identically from the committed script (coefficients, intercept and
CV delta all match the deployed `outputs/learned_gate_v79.pkl`). The small artifacts the chain
loads are committed under `outputs/` so the builders can be run without retraining.

## Verifying compliance

```bash
python research/verify_compliance.py
```

This re-checks, against the submitted zip: that no builder in the chain reads `splits/*.pkl`;
that predictions land on both sides of the split (one argmax over 17,393, not a restricted set);
that the router is not a folder oracle (seen-split images reach a photographed class 89.11% of the
time, unseen-split images an unphotographed class 77.58%; an oracle would score 100/100); and that
the gate artifact reproduces. It writes `outputs/compliance_report.json` and exits non-zero on any
failure. It reads the split files itself in order to run the oracle check; no builder does.

Two strictly per-image variants of v109 are included for organisers who read the rules as
forbidding batch-coupled inference. `builders/build_v109_compliant.py` freezes the gate's
standardisation and replaces the rank cutoff with a holdout-calibrated absolute threshold; it
changes 729 of 35,665 predictions (2.04%). `builders/build_v109_compliant_dbnorm.py` additionally
freezes the novel head's across-image normalisation; 4,213 predictions (11.81%). Their prediction
files are in `outputs/`.

## Layout

```
builders/            the six builders in the v109 chain, plus the two per-image variants
builders/legacy_archive/   earlier builders (v31–v72); v31 is the fully per-image fallback
research/            training scripts for the gate and re-rankers, the compliance checker,
                     and the ablation scripts referenced in HANDOFF.md
src/                 encoder fine-tuning (ft.py, fullft336.py, train_lora_cosface.py,
                     contrastive_ft.py), embedding extraction (embed*.py), text embeddings
reports/             the technical report: paper.html + paper_figs.py + render_paper.py → PDF
submissions/         the scored zips
outputs/             small artifacts the chain loads (gate and re-ranker pickles, predictions)
HANDOFF.md           the working ledger, kept throughout the competition
DISCLOSURE.md        every external dataset and model used, and two that were declined
COMPETITION_RULES.md our cached copy of the rules, with the gray zones we identified
```

## External resources

iNaturalist Open Data (117,225 research-grade images covering 7,364 species), TreeOfLife-200M
embeddings, and the public BioCLIP, BioCLIP-2 and TaxaBind checkpoints. FishBase descriptions
were fetched late in the competition, never used by any builder, and are declared anyway.
No model or dataset built for fish recognition was used; two such repositories were considered
and declined. Details in `DISCLOSURE.md`.

## License

Copyright © 2026 Lalith Sai, Intelligent Systems Laboratory (ISLab, CWNU), and Cheng Yaw Low.
All rights reserved; not open source.

Notwithstanding the above, the CV4Ecology 2026 / Codabench 16815 organisers are granted a
non-exclusive, irrevocable right to receive, inspect, execute, and independently reproduce the
inference code, model weights, and configuration required to verify the submitted result, and to
retain that material for verification and prize adjudication.
