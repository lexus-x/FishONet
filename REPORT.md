# Technical Report — Fish Species Recognition Challenge (Codabench 16815)

**FishONet: Open-Set Fine-Grained Fish Species Recognition via Shift-Augmented Multimodal Ensembles, Learned Quota Routing, and Leak-Free Shortlist Re-ranking**

- **Live Interactive Showcase:** [https://fihonet.lalithsai00.workers.dev/](https://fihonet.lalithsai00.workers.dev/)
- **Code Repository:** [https://github.com/lexus-x/FihOnet](https://github.com/lexus-x/FihOnet)
- **Final Submitted Artifact:** `submissions/submission_v83_rerank_both_f60.zip`
- **Validated Final Score:** **53.69690172437964% Overall** (Seen: **77.544%** · Novel/Unseen: **22.912%**)
- **Full External Data Audit:** `DISCLOSURE.md`

---

## Executive Summary

The CV4Ecology 2026 challenge presents a difficult open-set taxonomic identification problem: classify **35,665 evaluation images** across **17,393 candidate fish species**, where **11,598 species (66.7%) have zero training photographs** and are defined exclusively by scientific taxonomy and morphological text descriptions.

Our solution, **FishONet (v83)**, achieves **53.697% overall accuracy** on the public leaderboard. The system operates strictly without test-set knowledge or split ground truth, utilizing:
1. **Shift-Augmented Vision-Language Ensembles:** Addressing the citizen-science aspect ratio domain shift (training aspect $\sim 1.15$ vs. evaluation $\sim 2.12$) across 5 foundation encoders.
2. **Learned 12-Feature Quota Routing Gate:** Replacing hard thresholding with an invariant quota gate ($f=0.60$) over multimodal consistency signals.
3. **Leak-Free Shortlist Re-ranking:** A calibrated re-ranking stage diagnosed and trained under leak-free candidate pools, unlocking +0.266 real points where standard cross-validation gave a misleading +13.37 point artifact.

---

## 1. Task Definition & Core Challenges

| Component | Seen Head (Known Classes) | Novel/Unseen Head (Zero-Shot) |
| :--- | :--- | :--- |
| **Class Count** | 5,795 species (33.3%) | 11,598 species (66.7%) |
| **Training Data** | Labelled citizen-science photographs | Zero images — text descriptions & taxonomy only |
| **Evaluation Split** | 20,097 test images (56.35%) | 15,568 test images (43.65%) |
| **Available Evidence** | Class image prototypes, training exemplars | Taxonomy text, iNaturalist Open Data photo bank |

### Fundamental Bottlenecks
- **The Zero-Shot Penalty:** Two-thirds of the taxonomic target space has never been seen in training.
- **The Blind Decision Dilemma:** At test time, the model is not told whether an image belongs to a seen or unseen species. Any misrouted image chooses from a candidate set that cannot contain the true label ($\rightarrow$ guaranteed $0\%$ accuracy).
- **Aspect Ratio Shift:** Evaluation fish are horizontally elongated (mean aspect ratio $2.12$) compared to square/standard crops in training datasets ($1.15$).

---

## 2. System Architecture & Methodology

```
                          ┌── Seen Head (5,795 classes) ───┐
                          │   • 3-Encoder Image Protos      │
                          │   • Nearest Exemplar Bank       │
                          │   • Taxonomy Text Anchor        │
                          │   • 32-Feature Shortlist Ranker │
                          │                                 │
Input Image ──► 5 Encoders ┤                                 ├──► Quota Gate (f=0.60) ──► Final Species Prediction
(with Shift               │                                 │    (12 Multimodal Features)
 Augmentation)            │                                 │
                          │                                 │
                          └── Novel Head (11,598 classes) ──┘
                              • 6 Text Legs + TaxaBind
                              • 2 iNat Multi-View Banks
                              • 2 BioCLIP-2 Protos (ToL)
                              • 34-Feature Shortlist Ranker
```

### 2.1 Multi-Encoder Foundation Backbone
All fine-tuning was performed exclusively on provided competition training data using shift-matched augmentations (`scale(0.35, 1.0)`, `ratio(0.5, 2.0)`):
- **BioCLIP-2.5 ViT-H/14 (`ctftshift`):** Contrastive LoRA fine-tuning with taxonomic context.
- **BioCLIP-2.5 ViT-H/14 (`ftshift` & `fullft336shift`):** Standard and high-resolution 336px fine-tuned backbones.
- **BioCLIP-2 ViT-L/14:** Frozen backbone + LoRA v2, merged with TreeOfLife-200M precomputed representations.
- **TaxaBind ViT-B/16:** Multi-modal biomedical taxonomic embedding anchor.

### 2.2 Seen Head Formulation
The seen head computes an ensembled similarity score for each candidate $c \in \mathcal{C}_{\text{seen}}$:
$$\mathcal{S}_{\text{seen}}(x, c) = w_1 \cdot \text{proto}(x, c) + w_2 \cdot \text{cmax}(x, c) + w_3 \cdot \text{taxon\_sim}(x, c)$$
where weights $(1.0, 2.5, 2.5)$ combine mean class prototypes, nearest training exemplars, and taxonomic text descriptions.

### 2.3 Novel (Unseen) Head Formulation
For novel classes $\mathcal{C}_{\text{unseen}}$, ten distinct representations are fused via debiased normalization (`dbnorm`):
- **6 Text Legs:** Multi-template taxonomy, common names, and taxonomic context across four encoders.
- **2 iNaturalist Photo-Bank Legs:** Multi-view max-pooling over 7 crops (center, squashed, 5 overlapping horizontal strips) against 117,225 research-grade iNaturalist images (covering 7,364 species).
- **2 BioCLIP-2 Prototype Legs:** Dense representations merged with TreeOfLife-200M embeddings.

### 2.4 Multimodal Quota Gate
Rather than relying on noisy uncalibrated probability thresholds, the gate ranks all 35,665 test queries using a 12-feature logistic regression model:
- **Features Extracted:** Seen max similarity, top-1/top-2 margin, log-sum-exp entropy gap, per-encoder individual maxima, dual-text margin differences ($\text{sim}_{\text{seen\_text}} - \text{sim}_{\text{unseen\_text}}$), and iNat photo-bank confidence scores.
- **Quota Assignment:** Exactly the top $f = 0.60$ quantile of images sorted by familiarity score route to the Seen Head; the remaining $40\%$ route to the Novel Head.

### 2.5 Leak-Free Shortlist Re-ranking
Top candidates from initial fusion ($K=10$ on seen, $K=20$ on novel) are re-ordered by dedicated secondary classifiers:
- **Novel Head Re-ranker:** 34 within-query features (percentile ranks, gap-to-max, individual leg z-scores).
- **Seen Head Re-ranker:** 32 within-query rank and consistency features.
- All features are invariant to candidate pool size to ensure perfect domain transfer.

---

## 3. Results Progression

| Version | Overall Accuracy | Seen Head | Unseen Head | Key Innovation |
| :--- | :--- | :--- | :--- | :--- |
| **v33** | 47.78% | 75.8% | 11.6% | Baseline single-pipeline Sinkhorn routing |
| **v36** | 49.04% | 78.2% | 11.5% | Shift-augmented LoRA + TaxaBind multimodal leg |
| **v37** | 50.46% | 78.9% | 13.5% | iNaturalist Open Data photo bank prototypes |
| **v41–v46** | 50.84% | 78.9% | 14.5% | BioCLIP-2 dual representation & TreeOfLife-200M merge |
| **v50–v56** | 51.62% | 78.9% | 16.3% | 7-view crop-max pooling + 336px photo bank |
| **v77** | **53.38%** | **78.9%** | **20.3%** | **12-feature learned gate replacing heuristic routing (+1.767pt)** |
| **v79** | 53.42% | 79.1% | 20.3% | Optimal regularisation ($C=100$) refit |
| **v82** | 53.64% | 79.1% | 20.8% | Leak-free trained unseen shortlist re-ranker (+0.213pt) |
| **v83 (Best)** | **53.697%** | **77.54%** | **22.91%** | **Dual seen & unseen leak-free re-ranking system** |

---

## 4. Key Discovery: The Validation Harness Leak & Transfer Calibration

A central scientific contribution of this work is identifying how cross-validation harnesses fail on open-set retrieval tasks.

### 4.1 The Illusion vs. Reality
When developing the initial re-ranker (v81), class-disjoint 5-fold cross-validation on a pseudo-novel holdout produced a massive **+13.374 proxy gain**, yet yielded only **+0.006 on the real leaderboard**.

```
Standard Holdout Pool:    1,159 Pseudo-Novel  +  11,598 Real-Novel (Impossible Distractors)
                          └── Gold was ALWAYS in the 9.1% subpopulation ──┘
                          Model learned: "Detect training subpopulation signature" (Cheating)

Leak-Free Pool:           1,159 Pseudo-Novel Only (All candidates equally gold-eligible)
                          Model forced to: "Rank evidence based on visual & text features"
```

### 4.2 The Diagnosis & Fix
- **Root Cause:** The holdout candidate pool contained 1,159 pseudo-novel classes alongside 11,598 true-novel distractor classes. Because gold labels were strictly drawn from the 1,159 training-derived classes, the classifier learned the distribution signature of training classes rather than true visual-textual relevance.
- **The Remedy:** Candidate pools were restricted strictly to gold-eligible candidates, and all raw score levels were replaced with within-query rank-order statistics.
- **Outcome:** The measured validation gain dropped from $+13.374$ to a realistic $+8.154$, while real leaderboard gain surged from $+0.006$ to **$+0.213$**.

### 4.3 Transfer Anchor Calibration
By calibrating validation gains against real submissions, we established a strict predictive ratio:
$$\Delta \text{Real} \approx 0.026 \times \Delta \text{CV}_{\text{leak-free}}$$
This metric accurately forecasted the performance of v83 ($53.69\%$ projected vs. $53.6969\%$ measured).

---

## 5. Negative Results & System Ceilings

To aid future open-set biodiversity research, we document key saturated directions:

1. **Model-Class Re-ranking Limit:** Replacing Logistic Regression with Gradient Boosting (`HistGradientBoostingClassifier`) yielded $-0.008$ on novel classes and $+0.003$ on seen classes. The ceiling is governed by feature representation quality, not learner complexity.
2. **Alternative Vision-Language Encoders:** Testing $\sim 25$ candidate backbones revealed that BioCLIP's organismal domain pre-training strongly outperforms general-purpose models (SigLIP2 scored $5.13\%$ vs. BioCLIP $21.53\%$; BioTrove $\sim 0\%$).
3. **Routing Error Ceiling:** Aspect ratio shift causes gate performance to drop from $0.981$ AUC on holdout to $0.911$ on evaluation. This mis-routes $15.9\%$ of images, establishing an architectural ceiling at $\sim 54.35\%$.

---

## 6. Rules Compliance & Reproducibility

- **No Split-Leakage:** Zero usage of `splits/*.pkl` at test time. Classification uses a single unified argmax over all 17,393 classes.
- **Foundation Models Only:** No fish-specific third-party fine-tunes were employed.
- **Deterministic Pipeline:** The submission script (`builders/build_v83_rerank_both.py`) was verified to produce bit-identical predictions from clean weights.

```bash
# Complete Reproduction Pipeline
conda activate onet
python research/learned_gate_v77.py         # Trains 12-feature gate -> outputs/learned_gate_v79.pkl
python research/rerank_leakfree_v82.py      # Trains unseen ranker  -> outputs/rerank_unseen_v82_leakfree.pkl
python research/rerank_seen_v83.py          # Trains seen ranker    -> outputs/rerank_seen_v83.pkl
python builders/build_v83_rerank_both.py    # Generates submission  -> submissions/submission_v83_rerank_both_f60.zip
```

---

## 7. Interactive Project & Assets

- **Project Web Demonstration:** [https://fihonet.lalithsai00.workers.dev/](https://fihonet.lalithsai00.workers.dev/)
- **Repository:** [https://github.com/lexus-x/FihOnet](https://github.com/lexus-x/FihOnet)
- **External Dataset Disclosures:** Complete attribution for iNaturalist Open Data, TreeOfLife-200M, and TaxaBind available in `DISCLOSURE.md`.
